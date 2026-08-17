"""Hybrid retrieval: vector search + BM25 + rerank.

Three strategies, each covering what the others miss:

1. **Vector search** — finds provisions by meaning. Query "không đội mũ bảo
   hiểm" matches provisions about "người ngồi trên xe không đội mũ bảo hiểm"
   even though the words differ.

2. **BM25 keyword search** — finds provisions by exact terms. Query "168/2024"
   matches only documents with that identity; vector search would return
   anything topically similar.

3. **Cross-encoder rerank** — scores the top candidates against the query
   directly, reading the full pair rather than comparing two vectors. Catches
   mismatches that vector similarity alone cannot.

Results are fused with **Reciprocal Rank Fusion (RRF)**: each strategy ranks
candidates independently, and the final score is the sum of
``1 / (rank + K)`` across all strategies that returned a candidate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Literal

from neo4j import Session

from vtlaw.config import Settings, get_settings
from vtlaw.embed import Embedder, segment
from vtlaw.graph.client import GraphClient
from vtlaw.graph.schema import PROVISION_LABELS

log = logging.getLogger(__name__)

# Standard RRF damping constant. Larger K makes the ranking more democratic
# (rank differences matter less); smaller K makes it more sensitive to top ranks.
RRF_K = 60

# Cross-encoder input cap. The reranker config advertises 8194 positions; left
# unpinned, every pair pads to that and CPU inference becomes unusable.
RERANK_MAX_LENGTH = 256


@dataclass(frozen=True)
class Hit:
    """One candidate provision."""

    uid: str
    score: float
    doc_identity: str
    label: Literal["Article", "Clause", "Point"]
    content: str
    title: str | None = None

    @property
    def embedding_text(self) -> str:
        """Rebuild the text that was embedded (used for reranking)."""
        if self.title:
            return f"{self.title}\n{self.content}".strip()
        return self.content


@dataclass
class SearchResult:
    """Final retrieval result for one query."""

    query: str
    hits: list[Hit]
    strategy: str = "hybrid"  # "hybrid" | "vector" | "bm25"

    def top_k(self, k: int) -> list[Hit]:
        return self.hits[:k]

    def summary(self) -> str:
        if not self.hits:
            return f"{self.strategy}: no results"
        top = self.hits[0]
        return (
            f"{self.strategy}: {len(self.hits)} hits, "
            f"top={top.uid} score={top.score:.3f}"
        )


@dataclass
class RetrievalResult:
    """Result after rerank."""

    query: str
    hits: list[Hit]
    retrieval_score: str  # "hybrid" | "vector" | "bm25"
    reranked: bool

    def summary(self) -> str:
        tag = "reranked" if self.reranked else self.retrieval_score
        if not self.hits:
            return f"{tag}: no results"
        top = self.hits[0]
        return f"{tag}: {len(self.hits)} hits, top={top.uid} score={top.score:.3f}"


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


def vector_search(
    session: Session,
    embedder: Embedder,
    query: str,
    k: int,
) -> list[Hit]:
    """Vector similarity search against all provision labels."""
    vector = embedder.encode_query(query)
    hits: list[Hit] = []

    for label in PROVISION_LABELS:
        # Vector index has no pre-filter, so overfetch before filtering.
        # A document that is currently expired may still rank high by vector
        # similarity; the filter removes it afterwards.
        rows = session.run(
            """
            CALL db.index.vector.queryNodes($index, $k * $overfetch, $vector)
            YIELD node, score
            MATCH (d:Document {doc_identity: node.doc_identity})
            WHERE d.effect_date <= date() AND (d.expire_date IS NULL OR d.expire_date >= date())
            RETURN node.uid AS uid, node.doc_identity AS doc_identity,
                   node.content AS content, node.title AS title, score
            ORDER BY score DESC
            LIMIT $k
            """,
            index=f"{label.lower()}_embedding",
            k=k,
            overfetch=4,
            vector=vector,
        ).data()

        for row in rows:
            hits.append(
                Hit(
                    uid=row["uid"],
                    score=row["score"],
                    doc_identity=row["doc_identity"],
                    label=label,  # type: ignore[arg-type]
                    content=row["content"] or "",
                    title=row.get("title"),
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def bm25_search(
    session: Session,
    query: str,
    k: int,
) -> list[Hit]:
    """BM25 keyword search against all provision labels."""
    # Tokenize for BM25: same segmentation as the embedding model expects.
    tokenised = segment(query)
    hits: list[Hit] = []

    for label in PROVISION_LABELS:
        # Fulltext index also has no pre-filter, so overfetch before filtering.
        rows = session.run(
            """
            CALL db.index.fulltext.queryNodes($index, $search_text)
            YIELD node, score
            MATCH (d:Document {doc_identity: node.doc_identity})
            WHERE d.effect_date <= date() AND (d.expire_date IS NULL OR d.expire_date >= date())
            RETURN node.uid AS uid, node.doc_identity AS doc_identity,
                   node.content AS content, node.title AS title, score
            ORDER BY score DESC
            LIMIT $k
            """,
            index=f"{label.lower()}_fulltext",
            search_text=tokenised,
            k=k * 4,  # overfetch before filtering
        ).data()

        for row in rows[:k]:
            hits.append(
                Hit(
                    uid=row["uid"],
                    score=row["score"],
                    doc_identity=row["doc_identity"],
                    label=label,  # type: ignore[arg-type]
                    content=row["content"] or "",
                    title=row.get("title"),
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


# ---------------------------------------------------------------------------
# Fusion and reranking
# ---------------------------------------------------------------------------


def fuse_rrf(
    vector_hits: list[Hit],
    bm25_hits: list[Hit],
    k: int,
) -> list[Hit]:
    """Reciprocal Rank Fusion: sum of 1/(rank+K) across strategies."""
    scores: dict[str, float] = {}
    by_uid: dict[str, Hit] = {}

    for rank, hit in enumerate(vector_hits):
        scores[hit.uid] = scores.get(hit.uid, 0.0) + 1.0 / (rank + 1 + RRF_K)
        by_uid.setdefault(hit.uid, hit)

    for rank, hit in enumerate(bm25_hits):
        scores[hit.uid] = scores.get(hit.uid, 0.0) + 1.0 / (rank + 1 + RRF_K)
        by_uid.setdefault(hit.uid, hit)

    fused = [
        Hit(
            uid=uid,
            score=score,
            doc_identity=by_uid[uid].doc_identity,
            label=by_uid[uid].label,
            content=by_uid[uid].content,
            title=by_uid[uid].title,
        )
        for uid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused[:k]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Load and cache the cross-encoder.

    Cached because the caller is per-request: loading XLM-RoBERTa fresh on every
    query cost ~15s and the API segfaulted holding two copies in memory.
    """
    from sentence_transformers import CrossEncoder

    log.info("loading cross-encoder %s", model_name)
    # max_length pinned: the model reports max_position_embeddings=8194, so the
    # default would pad pairs to 8k tokens on CPU.
    return CrossEncoder(model_name, max_length=RERANK_MAX_LENGTH)


def rerank(
    hits: list[Hit],
    query: str,
    k: int,
    reranker_model: str = "AITeamVN/Vietnamese_Reranker",
) -> list[Hit]:
    """Cross-encoder rerank: score (query, provision) pairs directly.

    The cross-encoder reads the full pair rather than comparing two vectors,
    so it catches fine-grained mismatches that a vector similarity score
    cannot. Slower than vector search, so only applied to the top candidates.
    """
    if not hits:
        return hits[:k]

    try:
        model = _load_cross_encoder(reranker_model)
        pairs = [(query, hit.embedding_text) for hit in hits]
        scores = model.predict(pairs)
    except (ImportError, ValueError, OSError) as exc:
        log.warning("reranker unavailable (%s); skipping rerank", exc)
        return hits[:k]

    reranked = [
        Hit(
            uid=hit.uid,
            score=float(score),
            doc_identity=hit.doc_identity,
            label=hit.label,
            content=hit.content,
            title=hit.title,
        )
        for hit, score in zip(hits, scores, strict=False)
    ]
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked[:k]


# ---------------------------------------------------------------------------
# Retriever facade
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Facade for hybrid retrieval with optional rerank."""

    def __init__(
        self,
        client: GraphClient,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = settings or get_settings()

    def search(
        self,
        query: str,
        k: int = 10,
        strategy: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> SearchResult:
        """Run retrieval with the chosen strategy."""
        with self._client.session() as session:
            if strategy == "vector":
                hits = vector_search(session, self._embedder, query, k)
            elif strategy == "bm25":
                hits = bm25_search(session, query, k)
            else:  # hybrid
                vec = vector_search(session, self._embedder, query, k)
                bm25 = bm25_search(session, query, k)
                hits = fuse_rrf(vec, bm25, k)

        return SearchResult(query=query, hits=hits, strategy=strategy)

    def search_and_rerank(
        self,
        query: str,
        k: int = 10,
        strategy: Literal["hybrid", "vector", "bm25"] = "hybrid",
        rerank_top: int | None = None,
        reranker_model: str | None = None,
        heuristic_rerank: bool = False,
        as_of: date | None = None,
    ) -> RetrievalResult:
        """Retrieve, rerank, and optionally apply legal-safety heuristic.

        Args:
            query: The user query.
            k: Number of hits to return.
            strategy: "hybrid", "vector", or "bm25".
            rerank_top: Number of candidates to rerank with cross-encoder.
            reranker_model: Cross-encoder model name.
            heuristic_rerank: If True, apply amendment penalty + recency bonus
                after cross-encoder rerank.
            as_of: Date for recency calculation. Defaults to today.

        Returns:
            RetrievalResult with reranked hits.
        """
        result = self.search(query, k=k, strategy=strategy)

        if not result.hits:
            return RetrievalResult(
                query=query, hits=[], retrieval_score=strategy, reranked=False
            )

        rerank_k = rerank_top or min(k, 20)
        reranker = reranker_model or self._settings.rerank_model

        reranked_hits = rerank(result.hits, query, rerank_k, reranker)

        # Apply legal-safety heuristic after cross-encoder rerank
        if heuristic_rerank:
            from vtlaw.retrieve.heuristics import apply_heuristic_rerank
            reranked_hits = apply_heuristic_rerank(
                reranked_hits, self._client, as_of=as_of
            )

        return RetrievalResult(
            query=query,
            hits=reranked_hits,
            retrieval_score=strategy,
            reranked=True,
        )
