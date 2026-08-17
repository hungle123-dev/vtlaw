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
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from vtlaw.retrieve.search import Hit

HIT = Hit(
    uid="168/2024/NĐ-CP::article::6::clause::3::point::a",
    score=0.87,
    doc_identity="168/2024/NĐ-CP",
    label="Point",
    content="Điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h",
    title=None,
)


def _state(*, cached_hits=None, cached_answer=None, llm=False):
    state = MagicMock()
    state.settings.rerank_top = 15
    state.settings.context_k = 8
    state.settings.embed_model = "test-embed"
    state.settings.llm_model = "test-llm"
    state.llm_configured = llm
    state.graph = MagicMock()

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
    with patch("vtlaw.api.app.get_state", return_value=state):
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


class TestNoCache:
    def test_works_without_cache(self):
        state = _state(llm=True)
        state.cache = None
        with client_for(state) as client:
            response = client.post("/chat", json={"question": "q"})

        assert response.status_code == 200
