"""Tests for the two /chat paths the mock fixture in test_app.py never reached.

Both crashed in production while the suite stayed green:

  * cache HIT  — `get_retrieval` returns decoded dicts; the old code called
    `json.loads` on each one and raised TypeError on every repeat question.
  * LLM configured — the old code called `state.generator.generate(...)`, a
    method `AnswerGenerator` does not have, so every request AttributeError'd.

The fixture in test_app.py pinned `get_retrieval` to None and `generator` to
None, so neither branch was ever entered. These tests enter both.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import date
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from vtlaw.cache import CachedRetrieval
from vtlaw.retrieve.search import Hit

HIT = Hit(
    uid="168/2024/NĐ-CP::article::6::clause::3::point::a",
    score=0.87,
    doc_identity="168/2024/NĐ-CP",
    label="Point",
    content="Điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h",
    title=None,
)


def _state(*, cached_hits=None, cached_answer=None, llm=False, api_key=""):
    state = MagicMock()
    state.settings.rerank_top = 15
    state.settings.context_k = 8
    state.settings.embed_model = "test-embed"
    state.settings.llm_model = "test-llm"
    state.settings.decompose_queries = False
    # Set explicitly: a MagicMock attribute is truthy, so leaving api_key unset
    # would silently turn auth on for every test with an unusable key.
    state.settings.api_key = api_key
    state.settings.rate_limit_per_minute = 1000
    state.llm_configured = llm
    state.graph = MagicMock()
    state.decomposer = None
    state.rewriter = None
    state.router = None
    state.graph_queries = None

    cache = MagicMock()
    cache.get_retrieval.return_value = cached_hits
    cache.get_answer.return_value = cached_answer
    state.cache = cache

    state.retriever.search.return_value = MagicMock(hits=[HIT], strategy="hybrid")
    state.retriever.search_and_rerank.return_value = MagicMock(
        hits=[HIT], retrieval_score="hybrid", reranked=True
    )

    if llm:
        state.generator.generate_from_hits.return_value = "Phạt tiền từ 800.000 đồng."
    else:
        state.generator = None
    return state


@contextmanager
def client_for(state):
    """TestClient with `get_state` patched for the duration of the request.

    The patch has to stay open while the request runs — returning the client
    from a closed `with` block silently falls through to the real AppState and
    talks to Neo4j.
    """
    with patch("vtlaw.api.app.get_state", return_value=state), patch(
        "vtlaw.api.app._rate_limiter", None
    ):
        from vtlaw.api.app import app

        yield TestClient(app)


class TestRetrievalCacheHit:
    """The path that raised `TypeError: the JSON object must be str ... not dict`."""

    def test_cache_hit_returns_200(self):
        state = _state(cached_hits=[asdict(HIT)])
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "quá tốc độ phạt bao nhiêu"})

        assert response.status_code == 200
        assert response.json()["sources"][0]["uid"] == HIT.uid

    def test_cache_hit_does_not_re_retrieve(self):
        state = _state(cached_hits=[asdict(HIT)])
        with client_for(state) as client:
            client.post("/chat", json={"question": "quá tốc độ phạt bao nhiêu"})

        state.retriever.search.assert_not_called()
        state.retriever.search_and_rerank.assert_not_called()

    def test_cache_hit_keeps_retrieval_metadata(self):
        state = _state(
            cached_hits=CachedRetrieval(
                hits=[asdict(HIT)],
                reranked=True,
                sub_queries=["quá tốc độ phạt bao nhiêu"],
            )
        )
        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "quá tốc độ phạt bao nhiêu", "profile": "quality"}
            )

        assert response.json()["reranked"] is True
        assert response.json()["sub_queries"] == ["quá tốc độ phạt bao nhiêu"]

    def test_repeated_question_stays_200(self):
        """A second identical question is exactly what used to 500."""
        state = _state()
        with client_for(state) as client:
            first = client.post("/chat", json={"question": "x"})

            # What the first request wrote is what the second one reads back.
            state.cache.get_retrieval.return_value = (
                state.cache.set_retrieval.call_args.args[-1]
            )
            second = client.post("/chat", json={"question": "x"})

        assert (first.status_code, second.status_code) == (200, 200)
        assert second.json()["sources"] == first.json()["sources"]

    def test_cached_rows_round_trip_through_hit(self):
        """asdict -> Hit(**row) must survive; the reverse of the old bug."""
        assert Hit(**asdict(HIT)) == HIT


class TestLLMConfigured:
    """The path that raised `AttributeError: 'AnswerGenerator' has no 'generate'`."""

    def test_generates_answer(self):
        state = _state(llm=True)
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "quá tốc độ phạt bao nhiêu"})

        assert response.status_code == 200
        assert response.json()["answer"] == "Phạt tiền từ 800.000 đồng."
        state.generator.generate_from_hits.assert_called_once()

    def test_generator_receives_the_retrieved_hits(self):
        state = _state(llm=True)
        with client_for(state) as client:
            client.post("/chat", json={"question": "q"})

        _, hits = state.generator.generate_from_hits.call_args.args
        assert hits == [HIT]

    def test_generate_is_called_with_cached_hits_too(self):
        """Cache hit + LLM configured: generate from the cached hits, don't re-search."""
        state = _state(cached_hits=[asdict(HIT)], llm=True)
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        assert response.status_code == 200
        state.retriever.search_and_rerank.assert_not_called()
        state.generator.generate_from_hits.assert_called_once()

    def test_answer_cache_hit_skips_generation(self):
        state = _state(cached_hits=[asdict(HIT)], cached_answer="đã lưu", llm=True)
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        assert response.json()["answer"] == "đã lưu"
        state.generator.generate_from_hits.assert_not_called()

    def test_generated_answer_is_cached(self):
        state = _state(llm=True)
        with client_for(state) as client:
            client.post("/chat", json={"question": "q"})

        assert state.cache.set_answer.call_args.args[-1] == "Phạt tiền từ 800.000 đồng."

    def test_reports_an_unverified_generated_citation(self):
        state = _state(llm=True)
        state.generator.generate_from_hits.return_value = (
            "Theo Điều 1 Nghị định 100/2019/NĐ-CP, bị phạt."
        )
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        assert response.json()["citation_status"] == "unsupported"
        assert response.json()["unsupported_citations"] == ["100/2019/NĐ-CP::article::1"]


class TestNoCache:
    def test_works_without_cache(self):
        state = _state(llm=True)
        state.cache = None
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        assert response.status_code == 200


class TestTemporalChatContract:
    def test_as_of_reaches_retrieval_and_cache_without_llm(self):
        state = _state(llm=False)
        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "q", "as_of": "2025-01-01"}
            )

        assert response.status_code == 200
        state.retriever.search_and_rerank.assert_called_once()
        assert state.retriever.search_and_rerank.call_args.kwargs["as_of"] == date(
            2025, 1, 1
        )
        assert state.cache.get_retrieval.call_args.kwargs["as_of"] == date(2025, 1, 1)
        assert state.cache.set_retrieval.call_args.kwargs["as_of"] == date(2025, 1, 1)


class TestQueryDecomposition:
    def test_enabled_decomposition_reaches_retrieval(self):
        state = _state(llm=True)
        state.settings.decompose_queries = True
        state.decomposer = MagicMock()
        state.decomposer.decompose.return_value = [{"query": "vượt đèn đỏ xe mô tô"}]

        with client_for(state) as client:
            response = client.post(
                "/chat",
                json={"question": "vượt đèn đỏ phạt thế nào", "profile": "decomposition"},
            )

        assert response.status_code == 200
        assert state.retriever.search_and_rerank.call_args.kwargs["sub_queries"] == [
            "vượt đèn đỏ phạt thế nào",
            "vượt đèn đỏ xe mô tô",
        ]

    def test_quality_profile_composes_decomposition_and_real_reranking(self):
        state = _state(llm=True)
        state.decomposer = MagicMock()
        state.decomposer.decompose.return_value = [{"query": "vượt đèn đỏ xe mô tô"}]
        with client_for(state) as client:
            response = client.post(
                "/chat",
                json={"question": "vượt đèn đỏ phạt thế nào", "profile": "quality"},
            )

        assert response.status_code == 200
        assert response.json()["profile"] == "quality"
        assert response.json()["sub_queries"] == [
            "vượt đèn đỏ phạt thế nào",
            "vượt đèn đỏ xe mô tô",
        ]
        assert state.retriever.search_and_rerank.call_args.kwargs["rerank_enabled"] is True

    def test_baseline_profile_does_not_inherit_experimental_server_flags(self):
        state = _state(llm=True)
        state.settings.decompose_queries = True
        state.settings.rerank_enabled = True
        state.decomposer = MagicMock()
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "vượt đèn đỏ phạt thế nào"})

        assert response.status_code == 200
        assert response.json()["profile"] == "baseline"
        state.decomposer.decompose.assert_not_called()
        assert state.retriever.search_and_rerank.call_args.kwargs["rerank_enabled"] is False


class TestConversationRewrite:
    def test_follow_up_is_resolved_before_retrieval_and_reported(self):
        state = _state(llm=True)
        state.rewriter = MagicMock()
        state.rewriter.rewrite.return_value = "mức phạt xe ô tô vượt đèn đỏ"
        history = [{"role": "user", "content": "xe máy vượt đèn đỏ phạt bao nhiêu?"}]

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "còn ô tô thì sao?", "history": history}
            )

        assert response.status_code == 200
        assert response.json()["resolved_question"] == "mức phạt xe ô tô vượt đèn đỏ"
        assert state.retriever.search_and_rerank.call_args.args[0] == (
            "mức phạt xe ô tô vượt đèn đỏ"
        )

    def test_generation_receives_the_resolved_question_not_the_raw_turn(self):
        """Retrieval and generation must be asked the same question.

        Retrieval ran on the rewrite; handing the generator "còn ô tô thì sao?"
        makes it answer a question the evidence was never gathered for.
        """
        state = _state(llm=True)
        state.rewriter = MagicMock()
        state.rewriter.rewrite.return_value = "mức phạt xe ô tô vượt đèn đỏ"
        history = [{"role": "user", "content": "xe máy vượt đèn đỏ phạt bao nhiêu?"}]

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "còn ô tô thì sao?", "history": history}
            )

        assert response.status_code == 200
        generated_question = state.generator.generate_from_hits.call_args.args[0]
        assert generated_question == "mức phạt xe ô tô vượt đèn đỏ"
        # The response still echoes what the user actually typed.
        assert response.json()["question"] == "còn ô tô thì sao?"


class TestIntentRouting:
    def test_routes_before_rewriting_a_follow_up(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "retrieve"
        state.rewriter = MagicMock()
        state.rewriter.rewrite.return_value = "mức phạt xe ô tô vượt đèn đỏ"
        history = [{"role": "user", "content": "xe máy vượt đèn đỏ phạt bao nhiêu?"}]

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "còn ô tô thì sao?", "history": history}
            )

        assert response.status_code == 200
        assert response.json()["intent"] == "retrieve"
        state.router.route.assert_called_once_with("còn ô tô thì sao?")
        state.rewriter.rewrite.assert_called_once()

    def test_reject_does_not_call_retrieval_or_rewriter(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "reject"
        state.rewriter = MagicMock()

        with client_for(state) as client:
            response = client.post("/chat", json={"question": "nấu phở thế nào?"})

        assert response.status_code == 200
        assert response.json()["intent"] == "reject"
        assert response.json()["sources"] == []
        state.retriever.search_and_rerank.assert_not_called()
        state.rewriter.rewrite.assert_not_called()

    def test_runs_a_supported_graph_template_without_retrieval(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "cypher_query"
        state.graph_queries = MagicMock()
        state.graph_queries.answer.return_value = MagicMock(
            operation="article_count", answer="Nghị định 168/2024/NĐ-CP có 89 điều."
        )

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "Nghị định 168/2024/NĐ-CP có bao nhiêu điều?"}
            )

        assert response.status_code == 200
        assert response.json()["intent"] == "cypher_query"
        assert response.json()["graph_operation"] == "article_count"
        state.graph_queries.answer.assert_called_once()
        state.retriever.search_and_rerank.assert_not_called()
