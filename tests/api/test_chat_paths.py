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

import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from vtlaw.cache import Cache, CachedAnswer, CachedRetrieval
from vtlaw.config import Settings
from vtlaw.retrieve.search import Hit

HIT = Hit(
    uid="168/2024/NĐ-CP::article::6::clause::3::point::a",
    score=0.87,
    doc_identity="168/2024/NĐ-CP",
    label="Point",
    content="Điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h",
    title=None,
)

CLAUSE = Hit(
    uid="168/2024/NĐ-CP::article::6::clause::3",
    score=0.0,
    doc_identity="168/2024/NĐ-CP",
    label="Clause",
    content="Phạt tiền từ 800.000 đồng",
)

UNCITED_HIT = Hit(
    uid="168/2024/NĐ-CP::article::7",
    score=0.62,
    doc_identity="168/2024/NĐ-CP",
    label="Article",
    content="Một ứng viên truy xuất không được viện dẫn",
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


def _sse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        events.append((name, json.loads(data)))
    return events


@contextmanager
def client_for(state, *, raise_server_exceptions=True):
    """TestClient with `get_state` patched for the duration of the request.

    The patch has to stay open while the request runs — returning the client
    from a closed `with` block silently falls through to the real AppState and
    talks to Neo4j.
    """
    with patch("vtlaw.api.app.get_state", return_value=state), patch(
        "vtlaw.api.app._rate_limiter", None
    ):
        from vtlaw.api.app import app

        yield TestClient(app, raise_server_exceptions=raise_server_exceptions)


class TestRetrievalCacheHit:
    """The path that raised `TypeError: the JSON object must be str ... not dict`."""

    def test_cache_hit_returns_200(self):
        state = _state(cached_hits=[asdict(HIT)])
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "quá tốc độ phạt bao nhiêu"})

        assert response.status_code == 200
        assert response.json()["retrieved_candidates"][0]["uid"] == HIT.uid
        assert response.json()["cited_sources"][0]["uid"] == HIT.uid
        assert "sources" not in response.json()
        assert response.json()["citation_status"] == "verified"

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
        assert second.json()["retrieved_candidates"] == first.json()["retrieved_candidates"]

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

    def test_answer_cache_hit_without_a_full_rendered_record_regenerates(self):
        article = Hit(
            uid="168/2024/NĐ-CP::article::6",
            score=0.9,
            doc_identity="168/2024/NĐ-CP",
            label="Article",
            content="Điều 6",
        )
        state = _state(
            cached_hits=[asdict(article)],
            cached_answer=CachedAnswer(
                text="Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
                evidence_uids=frozenset(
                    {"168/2024/NĐ-CP::article::6::clause::3::point::a"}
                ),
            ),
            llm=True,
        )
        state.generator.generate_from_hits.return_value = "Câu trả lời đã tạo lại an toàn."
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        assert response.json()["answer"] == "Câu trả lời đã tạo lại an toàn."
        assert "Điểm a Khoản 3 Điều 6" not in response.json()["answer"]
        state.generator.generate_from_hits.assert_called_once()

    def test_separates_retrieved_candidates_from_rendered_cited_sources(self):
        state = _state(llm=True)
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[HIT, UNCITED_HIT], retrieval_score="hybrid", reranked=True
        )

        def generate(*_args, **kwargs):
            kwargs["evidence_uids"].update({HIT.uid, UNCITED_HIT.uid, CLAUSE.uid})
            rendered = kwargs.get("rendered_evidence")
            if rendered is not None:
                rendered.update({HIT.uid: HIT, UNCITED_HIT.uid: UNCITED_HIT, CLAUSE.uid: CLAUSE})
            return "Theo Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt."

        state.generator.generate_from_hits.side_effect = generate
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        data = response.json()
        assert "sources" not in data
        assert [source["uid"] for source in data["retrieved_candidates"]] == [
            HIT.uid,
            UNCITED_HIT.uid,
        ]
        assert [source["uid"] for source in data["cited_sources"]] == [CLAUSE.uid]

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

    def test_works_when_redis_is_unavailable(self):
        """An optional cache outage must not turn /chat into a 500."""
        state = _state(llm=True)
        state.settings = Settings(neo4j_password="test", intent_router_enabled=False)
        state.cache = Cache(state.settings)
        state.cache._redis = MagicMock()
        state.cache._redis.get.side_effect = ConnectionError("unavailable")
        state.cache._redis.setex.side_effect = ConnectionError("unavailable")

        with client_for(state, raise_server_exceptions=False) as client:
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
        assert response.json()["retrieved_candidates"] == []
        assert response.json()["cited_sources"] == []
        state.retriever.search_and_rerank.assert_not_called()


class TestExactCitationNoMatch:
    def test_rewrite_cannot_erase_a_raw_unknown_full_citation(self):
        question = "Điểm a Khoản 3 Điều 999 999/2099/NĐ-CP"
        state = _state(cached_hits=[asdict(HIT)], llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "direct_answer"
        state.rewriter = MagicMock()
        state.rewriter.rewrite.return_value = "mức phạt chạy quá tốc độ"
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[], retrieval_score="exact_citation", reranked=False
        )
        history = [{"role": "user", "content": "Cho tôi hỏi một điều luật."}]

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": question, "history": history}
            )

        data = response.json()
        assert "không tìm thấy" in data["answer"].casefold()
        assert "999/2099/NĐ-CP" in data["answer"]
        assert data["retrieved_candidates"] == []
        assert data["cited_sources"] == []
        state.router.route.assert_not_called()
        state.rewriter.rewrite.assert_not_called()
        assert state.retriever.search_and_rerank.call_args.args[0] == question
        state.cache.get_retrieval.assert_not_called()
        state.generator.generate_from_hits.assert_not_called()

    def test_successful_raw_citation_controls_retrieval_generation_and_strategy(self):
        question = "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP"
        answer = f"Theo {question}, phạt tiền từ 800.000 đồng."
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "reject"
        state.rewriter = MagicMock()
        state.rewriter.rewrite.return_value = "một câu hỏi khác"
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[HIT], retrieval_score="exact_citation", reranked=False
        )

        def generate(*_args, **kwargs):
            kwargs["evidence_uids"].add(HIT.uid)
            kwargs["rendered_evidence"][HIT.uid] = HIT
            return answer

        state.generator.generate_from_hits.side_effect = generate
        history = [{"role": "user", "content": "Hãy tra cứu chính xác."}]

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": question, "history": history}
            )

        data = response.json()
        assert data["answer"] == answer
        assert data["strategy"] == "exact_citation"
        assert data["citation_status"] == "verified"
        assert data["retrieved_candidates"][0]["uid"] == HIT.uid
        assert data["cited_sources"][0]["uid"] == HIT.uid
        assert data["resolved_question"] is None
        assert data["intent"] == "retrieve"
        state.router.route.assert_not_called()
        state.rewriter.rewrite.assert_not_called()
        assert state.retriever.search_and_rerank.call_args.args[0] == question
        assert state.generator.generate_from_hits.call_args.args[0] == question

    def test_route_rejection_cannot_preempt_an_unknown_full_citation(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "reject"
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[], retrieval_score="exact_citation", reranked=False
        )

        with client_for(state) as client:
            response = client.post(
                "/chat",
                json={"question": "Điểm a Khoản 3 Điều 999 999/2099/NĐ-CP"},
            )

        data = response.json()
        assert "không tìm thấy" in data["answer"].casefold()
        assert data["retrieved_candidates"] == []
        assert data["cited_sources"] == []
        state.router.route.assert_not_called()
        state.retriever.search_and_rerank.assert_called_once()
        state.generator.generate_from_hits.assert_not_called()

    def test_unknown_full_citation_is_terminal_and_preserves_provenance_contract(self):
        state = _state(llm=True)
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[], retrieval_score="exact_citation", reranked=False
        )

        with client_for(state) as client:
            response = client.post(
                "/chat",
                json={"question": "Điểm a Khoản 3 Điều 999 999/2099/NĐ-CP"},
            )

        data = response.json()
        assert response.status_code == 200
        assert "không tìm thấy" in data["answer"].casefold()
        assert "999/2099/NĐ-CP" in data["answer"]
        assert data["strategy"] == "exact_citation"
        assert data["retrieved_candidates"] == []
        assert data["cited_sources"] == []
        state.generator.generate_from_hits.assert_not_called()
        state.cache.get_answer.assert_not_called()

    def test_unknown_full_citation_ignores_stale_approximate_retrieval_cache(self):
        state = _state(cached_hits=[asdict(HIT)], llm=True)
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[], retrieval_score="exact_citation", reranked=False
        )

        with client_for(state) as client:
            response = client.post(
                "/chat",
                json={"question": "Điểm a Khoản 3 Điều 999 999/2099/NĐ-CP"},
            )

        assert response.json()["strategy"] == "exact_citation"
        assert response.json()["retrieved_candidates"] == []
        state.retriever.search_and_rerank.assert_called_once()
        state.generator.generate_from_hits.assert_not_called()


class TestChatStreaming:
    def test_streams_only_verified_final_answer_after_truthful_statuses(self):
        state = _state(llm=True)
        raw_draft = "RAW PROVIDER DRAFT — NEVER SEND"
        verified = (
            "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, "
            "phạt tiền từ 800.000 đồng."
        )

        def generate(*_args, **kwargs):
            if on_delta := kwargs.get("on_delta"):
                on_delta(raw_draft)
            kwargs["evidence_uids"].add(HIT.uid)
            kwargs["rendered_evidence"][HIT.uid] = HIT
            return verified

        state.generator.generate_from_hits.side_effect = generate
        with (
            client_for(state) as client,
            client.stream("POST", "/chat/stream", json={"question": "q"}) as response,
        ):
            body = response.read().decode()
            events = _sse_events(body)

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        names = [name for name, _ in events]
        statuses = [data["phase"] for name, data in events if name == "status"]
        assert statuses == ["request", "retrieval", "citation_validation"]
        assert names[:3] == ["status", "status", "status"]
        assert set(names[3:-2]) == {"delta"}
        assert names[-2:] == ["final", "done"]
        assert "replace" not in names
        assert raw_draft not in body
        final = next(data for name, data in events if name == "final")
        deltas = "".join(data["text"] for name, data in events if name == "delta")
        assert deltas == final["answer"] == verified
        assert final["citation_status"] == "verified"
        assert final["retrieved_candidates"][0]["uid"] == HIT.uid

    def test_unsupported_output_emits_error_and_done_without_answer_events(self):
        state = _state(llm=True)

        def generate(*_args, **kwargs):
            kwargs["evidence_uids"].add(HIT.uid)
            kwargs["rendered_evidence"][HIT.uid] = HIT
            return "Theo Điều 1 Nghị định 100/2019/NĐ-CP, bị phạt."

        state.generator.generate_from_hits.side_effect = generate
        with (
            client_for(state) as client,
            client.stream("POST", "/chat/stream", json={"question": "q"}) as response,
        ):
            events = _sse_events(response.read().decode())

        assert [name for name, _ in events][-2:] == ["error", "done"]
        assert not {"delta", "final", "replace"} & {name for name, _ in events}

    def test_exact_citation_no_match_emits_no_final_answer(self):
        state = _state(llm=True)
        state.retriever.search_and_rerank.return_value = MagicMock(
            hits=[], retrieval_score="exact_citation", reranked=False
        )

        with (
            client_for(state) as client,
            client.stream(
                "POST",
                "/chat/stream",
                json={"question": "Điều 999 999/2099/NĐ-CP"},
            ) as response,
        ):
            events = _sse_events(response.read().decode())

        assert [name for name, _ in events][-2:] == ["error", "done"]
        assert not {"delta", "final", "replace"} & {name for name, _ in events}
        statuses = [data for name, data in events if name == "status"]
        assert [status["phase"] for status in statuses] == ["request", "terminal"]
        assert "Không tìm thấy trích dẫn chính xác" in statuses[-1]["message"]

    def test_rejected_request_emits_terminal_status_without_retrieval_claim(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "reject"

        with (
            client_for(state) as client,
            client.stream(
                "POST", "/chat/stream", json={"question": "nấu phở thế nào?"}
            ) as response,
        ):
            events = _sse_events(response.read().decode())

        statuses = [data for name, data in events if name == "status"]
        assert [status["phase"] for status in statuses] == ["request", "terminal"]
        assert "Không truy xuất" in statuses[-1]["message"]
        state.retriever.search_and_rerank.assert_not_called()

    def test_direct_answer_emits_terminal_status_without_retrieval_claim(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "direct_answer"

        with (
            client_for(state) as client,
            client.stream("POST", "/chat/stream", json={"question": "bạn là ai?"}) as response,
        ):
            events = _sse_events(response.read().decode())

        statuses = [data for name, data in events if name == "status"]
        assert [status["phase"] for status in statuses] == ["request", "terminal"]
        assert "Không truy xuất" in statuses[-1]["message"]
        state.retriever.search_and_rerank.assert_not_called()

    def test_safe_graph_template_streams_status_delta_final_and_done(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "reject"
        state.graph_queries = MagicMock()
        answer = "Nghị định 168/2024/NĐ-CP có 89 điều."
        state.graph_queries.answer.return_value = MagicMock(
            operation="article_count", answer=answer, sources=()
        )

        with (
            client_for(state) as client,
            client.stream(
                "POST",
                "/chat/stream",
                json={"question": "Nghị định 168/2024/NĐ-CP có bao nhiêu điều?"},
            ) as response,
        ):
            events = _sse_events(response.read().decode())

        names = [name for name, _ in events]
        statuses = [data["phase"] for name, data in events if name == "status"]
        final = next(data for name, data in events if name == "final")
        deltas = "".join(data["text"] for name, data in events if name == "delta")
        assert statuses == ["request", "graph_query", "citation_validation"]
        assert names[-2:] == ["final", "done"]
        assert "error" not in names
        assert deltas == final["answer"] == answer
        assert final["strategy"] == "graph_template"
        assert final["graph_operation"] == "article_count"
        assert final["citation_status"] == "not_applicable"
        state.router.route.assert_not_called()
        state.retriever.search_and_rerank.assert_not_called()

    def test_pipeline_error_emits_no_final_answer(self):
        state = _state(llm=True)
        state.retriever.search_and_rerank.side_effect = RuntimeError("provider draft")

        with (
            client_for(state, raise_server_exceptions=False) as client,
            client.stream("POST", "/chat/stream", json={"question": "q"}) as response,
        ):
            events = _sse_events(response.read().decode())

        assert [name for name, _ in events][-2:] == ["error", "done"]
        assert not {"delta", "final", "replace"} & {name for name, _ in events}

    def test_streaming_route_keeps_api_key_authentication(self):
        state = _state(llm=False, api_key="secret")
        with client_for(state) as client:
            response = client.post("/chat/stream", json={"question": "q"})

        assert response.status_code == 401
        state.retriever.search_and_rerank.assert_not_called()

    def test_runs_a_supported_graph_template_without_retrieval(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "reject"
        state.rewriter = MagicMock()
        state.rewriter.rewrite.return_value = "một câu hỏi khác"
        state.graph_queries = MagicMock()
        state.graph_queries.answer.return_value = MagicMock(
            operation="article_count",
            answer="Nghị định 168/2024/NĐ-CP có 89 điều.",
            sources=(),
        )
        question = "Nghị định 168/2024/NĐ-CP có bao nhiêu điều?"
        history = [{"role": "user", "content": "Hãy dùng graph."}]

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": question, "history": history}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "cypher_query"
        assert data["strategy"] == "graph_template"
        assert data["graph_operation"] == "article_count"
        assert data["citation_status"] == "not_applicable"
        assert data["resolved_question"] is None
        state.router.route.assert_not_called()
        state.rewriter.rewrite.assert_not_called()
        state.graph_queries.answer.assert_called_once_with(question, as_of=None)
        state.retriever.search_and_rerank.assert_not_called()

    def test_amendment_graph_answer_exposes_matching_verified_sources(self):
        question = "Văn bản nào sửa đổi Điều 6 168/2024/NĐ-CP?"
        answer = "Điều 2 100/2019/NĐ-CP sửa đổi Điều 6 168/2024/NĐ-CP."
        source = Hit(
            uid="100/2019/NĐ-CP::article::2",
            score=1.0,
            doc_identity="100/2019/NĐ-CP",
            label="Article",
            content="Sửa đổi Điều 6 Nghị định 168/2024/NĐ-CP.",
        )
        target = Hit(
            uid="168/2024/NĐ-CP::article::6",
            score=1.0,
            doc_identity="168/2024/NĐ-CP",
            label="Article",
            content="Quy định được sửa đổi.",
        )
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "retrieve"
        state.graph_queries = MagicMock()
        state.graph_queries.answer.return_value = MagicMock(
            operation="amendment_history", answer=answer, sources=(source, target)
        )

        with client_for(state) as client:
            response = client.post("/chat", json={"question": question})

        data = response.json()
        assert data["answer"] == answer
        assert data["citation_status"] == "verified"
        assert data["unsupported_citations"] == []
        assert {item["uid"] for item in data["retrieved_candidates"]} == {
            source.uid,
            target.uid,
        }
        assert {item["uid"] for item in data["cited_sources"]} == {
            source.uid,
            target.uid,
        }
        state.retriever.search_and_rerank.assert_not_called()

    def test_graph_template_citation_without_evidence_returns_safe_fallback(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "cypher_query"
        state.graph_queries = MagicMock()
        state.graph_queries.answer.return_value = MagicMock(
            operation="amendment_history",
            answer="Theo Điều 6 Nghị định 168/2024/NĐ-CP, có thay đổi.",
            sources=(),
        )

        with client_for(state) as client:
            response = client.post(
                "/chat",
                json={"question": "Văn bản nào sửa đổi Điều 6 168/2024/NĐ-CP?"},
            )

        data = response.json()
        assert "Theo Điều 6 Nghị định 168/2024/NĐ-CP" not in data["answer"]
        assert data["retrieved_candidates"] == []
        assert data["cited_sources"] == []
        assert data["citation_status"] == "unsupported"

        with (
            client_for(state) as client,
            client.stream(
                "POST",
                "/chat/stream",
                json={"question": "Văn bản nào sửa đổi Điều 6 168/2024/NĐ-CP?"},
            ) as stream,
        ):
            events = _sse_events(stream.read().decode())

        assert [name for name, _ in events][-2:] == ["error", "done"]
        assert not {"delta", "final"} & {name for name, _ in events}

    def test_missing_graph_document_is_terminal_and_does_not_generic_fallback(self):
        state = _state(llm=True)
        state.settings.intent_router_enabled = True
        state.router = MagicMock()
        state.router.route.return_value = "cypher_query"
        state.graph_queries = MagicMock()
        state.graph_queries.answer.return_value = MagicMock(
            operation="article_count",
            answer=(
                "Không thể thực hiện article_count: không có văn bản "
                "999/2099/NĐ-CP trong corpus."
            ),
        )

        with client_for(state) as client:
            response = client.post(
                "/chat", json={"question": "999/2099/NĐ-CP có bao nhiêu điều?"}
            )

        data = response.json()
        assert data["graph_operation"] == "article_count"
        assert "không có văn bản 999/2099/NĐ-CP trong corpus" in data["answer"]
        assert data["retrieved_candidates"] == []
        assert data["cited_sources"] == []
        state.retriever.search_and_rerank.assert_not_called()
