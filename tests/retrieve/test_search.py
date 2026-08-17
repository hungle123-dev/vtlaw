"""Tests for retrieve stage - hybrid search + rerank.

Tests cover:
- Hit dataclass construction and properties
- fuse_rrf: RRF fusion logic with various input combinations
- rerank: cross-encoder reranking with mocked model
- HybridRetriever: facade with mocked Neo4j session

No Neo4j needed — all database calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from neo4j import Session

from vtlaw.graph.client import GraphClient
from vtlaw.retrieve.search import (
    Hit,
    HybridRetriever,
    SearchResult,
    _load_cross_encoder,
    bm25_search,
    fuse_rrf,
    rerank,
    vector_search,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_hit(
    uid: str = "test::1",
    score: float = 0.5,
    doc_identity: str = "test",
    label: str = "Clause",
    content: str = "test content",
    title: str | None = None,
) -> Hit:
    return Hit(
        uid=uid,
        score=score,
        doc_identity=doc_identity,
        label=label,  # type: ignore[arg-type]
        content=content,
        title=title,
    )


@pytest.fixture
def mock_session():
    return MagicMock(spec=Session)


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode_query.return_value = [0.1] * 768
    return embedder


def _client_yielding(session):
    """A GraphClient stand-in whose `session()` context manager yields `session`.

    HybridRetriever takes the client by argument, so patching GraphClient.session
    does nothing — the retriever never touches the real class.
    """
    client = MagicMock()
    client.session.return_value.__enter__ = lambda _: session
    client.session.return_value.__exit__ = lambda *a: None
    return client


def _settings(**overrides):
    """Settings stub carrying only what `search` reads."""
    defaults = {
        "fetch_k": 30,
        "rrf_k": 10,
        "rrf_vector_weight": 3.0,
        "rrf_bm25_weight": 1.0,
    }
    return MagicMock(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# Hit dataclass
# ---------------------------------------------------------------------------


class TestHit:
    def test_embedding_text_with_title(self):
        hit = make_hit(title="Article title")
        assert hit.embedding_text == "Article title\ntest content"

    def test_embedding_text_without_title(self):
        hit = make_hit(title=None)
        assert hit.embedding_text == "test content"

    def test_embedding_text_with_empty_content(self):
        hit = make_hit(content="", title="Title only")
        # f"{title}\n{content}".strip() with empty content = "Title only"
        assert hit.embedding_text == "Title only"

    def test_embedding_text_with_none_title(self):
        hit = make_hit(title=None, content="Content only")
        assert hit.embedding_text == "Content only"


# ---------------------------------------------------------------------------
# fuse_rrf
# ---------------------------------------------------------------------------


class TestFuseRRF:
    def test_combines_disjoint_lists(self):
        """Two lists with no overlap should produce union."""
        vec = [make_hit(uid="a"), make_hit(uid="b")]
        bm25 = [make_hit(uid="c"), make_hit(uid="d")]

        fused = fuse_rrf(vec, bm25, k=10)

        assert len(fused) == 4
        uids = {h.uid for h in fused}
        assert uids == {"a", "b", "c", "d"}

    def test_overlapping_hit_ranks_higher(self):
        """A hit appearing in both lists should rank above hits in one list."""
        vec = [make_hit(uid="both"), make_hit(uid="vec_only")]
        bm25 = [make_hit(uid="both"), make_hit(uid="bm25_only")]

        fused = fuse_rrf(vec, bm25, k=10)

        # "both" appears at rank 0 in both lists → highest RRF score
        assert fused[0].uid == "both"
        assert fused[0].score > fused[1].score

    def test_rrf_score_is_sum_of_reciprocals(self):
        """RRF score = sum(1/(rank+1+K)) across lists that contain the hit."""
        from vtlaw.retrieve.search import RRF_K

        vec = [make_hit(uid="x")]
        bm25 = [make_hit(uid="x")]

        fused = fuse_rrf(vec, bm25, k=10)

        expected = 1.0 / (0 + 1 + RRF_K) + 1.0 / (0 + 1 + RRF_K)
        assert abs(fused[0].score - expected) < 1e-10

    def test_empty_lists(self):
        assert fuse_rrf([], [], k=10) == []

    def test_one_empty_list(self):
        vec = [make_hit(uid="a"), make_hit(uid="b")]
        assert len(fuse_rrf(vec, [], k=10)) == 2

    def test_respects_k(self):
        vec = [make_hit(uid=f"v{i}") for i in range(20)]
        bm25 = [make_hit(uid=f"b{i}") for i in range(20)]

        fused = fuse_rrf(vec, bm25, k=5)

        assert len(fused) == 5

    def test_preserves_hit_metadata(self):
        vec = [make_hit(uid="a", doc_identity="doc1", content="content1")]
        bm25 = [make_hit(uid="b", doc_identity="doc2", content="content2")]

        fused = fuse_rrf(vec, bm25, k=10)

        by_uid = {h.uid: h for h in fused}
        assert by_uid["a"].doc_identity == "doc1"
        assert by_uid["a"].content == "content1"
        assert by_uid["b"].doc_identity == "doc2"
        assert by_uid["b"].content == "content2"

    def test_bm25_scored_once_not_twice(self):
        """A BM25-only hit gets exactly one term, not a duplicated one.

        A stray second accumulation in the BM25 loop would double every
        BM25-only score and silently invert the weighting.
        """
        fused = fuse_rrf([], [make_hit(uid="b")], k=10, rrf_k=10, bm25_weight=1.0)

        assert abs(fused[0].score - 1.0 / 11) < 1e-12

    def test_vector_weight_outranks_bm25_at_equal_rank(self):
        """The whole point of the weights: same rank, stronger leg wins."""
        fused = fuse_rrf(
            [make_hit(uid="from_vec")],
            [make_hit(uid="from_bm25")],
            k=10,
            vector_weight=3.0,
            bm25_weight=1.0,
        )

        assert [h.uid for h in fused] == ["from_vec", "from_bm25"]

    def test_smaller_rrf_k_sharpens_the_rank_signal(self):
        """rrf_k damps rank. Smaller k => bigger gap between rank 1 and rank 2."""
        hits = [make_hit(uid="first"), make_hit(uid="second")]

        flat = fuse_rrf(hits, [], k=10, rrf_k=60)
        sharp = fuse_rrf(hits, [], k=10, rrf_k=1)

        assert sharp[0].score / sharp[1].score > flat[0].score / flat[1].score

    def test_weights_default_to_unweighted(self):
        """Omitting the weights keeps the textbook 1:1 behaviour."""
        vec, bm25 = [make_hit(uid="v")], [make_hit(uid="b")]

        fused = fuse_rrf(vec, bm25, k=10, rrf_k=10)

        assert fused[0].score == fused[1].score

    def test_sorts_by_score_descending(self):
        vec = [make_hit(uid="low", score=0.01), make_hit(uid="high", score=0.99)]
        bm25 = []

        fused = fuse_rrf(vec, bm25, k=10)

        # RRF replaces original scores, but the ranking order is preserved
        # because rank 0 gets a higher RRF contribution than rank 1.
        assert fused[0].uid == "low"  # rank 0 in vec → higher RRF
        assert fused[1].uid == "high"


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


class TestRerank:
    @pytest.fixture(autouse=True)
    def _cold_model_cache(self):
        """`rerank` memoises the loaded cross-encoder; each test must start cold.

        Without this, the first test's mock stays cached under "test-model" and
        later tests silently score against it instead of their own mock.
        """
        _load_cross_encoder.cache_clear()
        yield
        _load_cross_encoder.cache_clear()

    def test_with_mocked_cross_encoder(self):
        """rerank should call CrossEncoder.predict and sort by score."""
        hits = [make_hit(uid="a"), make_hit(uid="b"), make_hit(uid="c")]

        with patch("sentence_transformers.CrossEncoder") as mock_ce_class:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.3, 0.9, 0.6]
            mock_ce_class.return_value = mock_model

            reranked = rerank(hits, "query", k=3, reranker_model="test-model")

        assert len(reranked) == 3
        # Sorted by score descending
        assert reranked[0].uid == "b"
        assert reranked[0].score == 0.9
        assert reranked[1].uid == "c"
        assert reranked[1].score == 0.6
        assert reranked[2].uid == "a"
        assert reranked[2].score == 0.3

    def test_respects_k(self):
        hits = [make_hit(uid=f"h{i}") for i in range(10)]

        with patch("sentence_transformers.CrossEncoder") as mock_ce_class:
            mock_model = MagicMock()
            mock_model.predict.return_value = list(range(10, 0, -1))
            mock_ce_class.return_value = mock_model

            reranked = rerank(hits, "query", k=3, reranker_model="test-model")

        assert len(reranked) == 3

    def test_empty_hits(self):
        with patch("sentence_transformers.CrossEncoder") as mock_ce_class:
            mock_model = MagicMock()
            mock_ce_class.return_value = mock_model

            reranked = rerank([], "query", k=3, reranker_model="test-model")

        assert len(reranked) == 0
        mock_model.predict.assert_not_called()

    def test_preserves_metadata(self):
        hits = [
            make_hit(uid="a", doc_identity="doc1", content="content1"),
            make_hit(uid="b", doc_identity="doc2", content="content2"),
        ]

        with patch("sentence_transformers.CrossEncoder") as mock_ce_class:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.8, 0.2]
            mock_ce_class.return_value = mock_model

            reranked = rerank(hits, "query", k=2, reranker_model="test-model")

        by_uid = {h.uid: h for h in reranked}
        assert by_uid["a"].doc_identity == "doc1"
        assert by_uid["a"].content == "content1"
        assert by_uid["b"].doc_identity == "doc2"
        assert by_uid["b"].content == "content2"

    def test_returns_hits_on_import_error(self):
        """If sentence-transformers is not installed, rerank returns hits[:k]."""
        hits = [make_hit(uid="a"), make_hit(uid="b")]

        with patch(
            "builtins.__import__",
            side_effect=ImportError("not installed"),
        ):
            reranked = rerank(hits, "query", k=1, reranker_model="test-model")

        assert len(reranked) == 1
        assert reranked[0].uid == "a"

    def test_uses_embedding_text_for_pairs(self):
        """rerank should build (query, embedding_text) pairs."""
        hits = [
            make_hit(uid="a", content="content a", title="title a"),
            make_hit(uid="b", content="content b", title=None),
        ]

        with patch("sentence_transformers.CrossEncoder") as mock_ce_class:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.5, 0.6]
            mock_ce_class.return_value = mock_model

            rerank(hits, "my query", k=2, reranker_model="test-model")

        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0] == ("my query", "title a\ncontent a")
        assert pairs[1] == ("my query", "content b")


# ---------------------------------------------------------------------------
# vector_search and bm25_search (mocked Neo4j)
# ---------------------------------------------------------------------------


class TestVectorSearch:
    def test_returns_hits_from_session(self, mock_session, mock_embedder):
        # vector_search loops over 3 labels (Article, Clause, Point).
        # Each label call returns the same mocked row, so we get 3 hits.
        mock_session.run.return_value = MagicMock(
            data=lambda: [
                {
                    "uid": "test::article::1",
                    "score": 0.9,
                    "doc_identity": "test",
                    "content": "content",
                    "title": "title",
                }
            ]
        )

        results = vector_search(mock_session, mock_embedder, "query", k=5)

        # 3 labels × 1 row each = 3 hits
        assert len(results) == 3
        assert all(h.uid == "test::article::1" for h in results)
        assert all(h.score == 0.9 for h in results)

    def test_calls_encode_query(self, mock_session, mock_embedder):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        vector_search(mock_session, mock_embedder, "my query", k=5)

        mock_embedder.encode_query.assert_called_once_with("my query")

    def test_empty_results(self, mock_session, mock_embedder):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        results = vector_search(mock_session, mock_embedder, "query", k=5)

        assert results == []


class TestBM25Search:
    def test_returns_hits_from_session(self, mock_session):
        # bm25_search loops over 3 labels (Article, Clause, Point).
        mock_session.run.return_value = MagicMock(
            data=lambda: [
                {
                    "uid": "test::clause::1",
                    "score": 5.5,
                    "doc_identity": "test",
                    "content": "content",
                    "title": None,
                }
            ]
        )

        results = bm25_search(mock_session, "query", k=5)

        # 3 labels × 1 row each = 3 hits
        assert len(results) == 3
        assert all(h.uid == "test::clause::1" for h in results)
        assert all(h.score == 5.5 for h in results)

    def test_tokenizes_query(self, mock_session):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        with patch("vtlaw.retrieve.search.segment") as mock_segment:
            mock_segment.return_value = "segmented"
            bm25_search(mock_session, "my query", k=5)

            mock_segment.assert_called_once_with("my query")

    def test_empty_results(self, mock_session):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        results = bm25_search(mock_session, "query", k=5)

        assert results == []


# ---------------------------------------------------------------------------
# HybridRetriever facade
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def test_search_returns_search_result(self, mock_session, mock_embedder):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        with patch.object(GraphClient, "session") as mock_session_ctx:
            mock_session_ctx.return_value.__enter__ = lambda _: mock_session
            mock_session_ctx.return_value.__exit__ = lambda *a: None

            retriever = HybridRetriever(MagicMock(), mock_embedder)
            result = retriever.search("query", k=5, strategy="vector")

        assert isinstance(result, SearchResult)
        assert result.strategy == "vector"

    def test_search_hybrid_calls_both_strategies(self, mock_session, mock_embedder):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        with patch.object(GraphClient, "session") as mock_session_ctx:
            mock_session_ctx.return_value.__enter__ = lambda _: mock_session
            mock_session_ctx.return_value.__exit__ = lambda *a: None

            retriever = HybridRetriever(MagicMock(), mock_embedder)
            result = retriever.search("query", k=5, strategy="hybrid")

        assert result.strategy == "hybrid"

    def test_search_with_no_results(self, mock_session, mock_embedder):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        with patch.object(GraphClient, "session") as mock_session_ctx:
            mock_session_ctx.return_value.__enter__ = lambda _: mock_session
            mock_session_ctx.return_value.__exit__ = lambda *a: None

            retriever = HybridRetriever(MagicMock(), mock_embedder)
            result = retriever.search("query", k=5)

        assert len(result.hits) == 0
        assert "no results" in result.summary()

    def test_fetches_the_wide_pool_not_just_k(self, mock_session, mock_embedder):
        """Each leg must fetch settings.fetch_k, then the fused list is cut to k.

        Fetching only k per leg starved the fusion — it saw 2k rows and discarded
        the depth where the answer often sat. Measured cost of that bug: recall@5
        0.598 instead of 0.636, with no ranking change.
        """
        rows = [
            {
                "uid": f"doc::article::{i}",
                "doc_identity": "doc",
                "content": f"content {i}",
                "title": None,
                "score": 1.0 - i / 100,
            }
            for i in range(40)
        ]
        mock_session.run.return_value = MagicMock(data=lambda: rows)

        retriever = HybridRetriever(
            _client_yielding(mock_session), mock_embedder, _settings(fetch_k=30)
        )
        result = retriever.search("query", k=5, strategy="vector")

        assert mock_session.run.call_args.kwargs["k"] == 30
        assert len(result.hits) == 5

    def test_pool_never_narrower_than_k(self, mock_session, mock_embedder):
        """A caller asking for more than fetch_k must not get a truncated pool."""
        mock_session.run.return_value = MagicMock(data=lambda: [])

        retriever = HybridRetriever(
            _client_yielding(mock_session), mock_embedder, _settings(fetch_k=30)
        )
        retriever.search("query", k=50, strategy="vector")

        assert mock_session.run.call_args.kwargs["k"] == 50

    def test_explicit_fetch_k_overrides_settings(self, mock_session, mock_embedder):
        mock_session.run.return_value = MagicMock(data=lambda: [])

        retriever = HybridRetriever(
            _client_yielding(mock_session), mock_embedder, _settings(fetch_k=30)
        )
        retriever.search("query", k=5, strategy="vector", fetch_k=100)

        assert mock_session.run.call_args.kwargs["k"] == 100
