"""Tests for the generate and API stages — no network, no LLM calls.

Tests cover:
- build_user_prompt: correct question + context formatting
- build_context: empty hits, multiple hits, temporal date
- format_provision: citation, doc_identity, content formatting
- ChatRequest/ChatResponse Pydantic models: validation, source formatting
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vtlaw.generate.context_builder import build_context, format_provision
from vtlaw.generate.prompts import SYSTEM_PROMPT, build_user_prompt
from vtlaw.retrieve.search import Hit


def make_hit(
    uid: str = "168/2024/NĐ-CP::article::6::clause::3",
    score: float = 0.8,
    doc_identity: str = "168/2024/NĐ-CP",
    label: str = "Clause",
    content: str = "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng.",
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


# ---------------------------------------------------------------------------
# format_provision
# ---------------------------------------------------------------------------


class TestFormatProvision:
    def test_includes_citation_and_content(self):
        hit = make_hit(
            uid="168/2024/NĐ-CP::article::6::clause::3",
            content="Phạt tiền từ 800.000 đồng.",
        )
        text = format_provision(hit)

        assert "Khoản 3 Điều 6 168/2024/NĐ-CP" in text
        assert "168/2024/NĐ-CP" in text
        assert "Phạt tiền" in text

    def test_includes_title_for_articles(self):
        hit = make_hit(
            uid="168/2024/NĐ-CP::article::6",
            label="Article",
            title="Xử phạt người điều khiển xe ô tô",
            content="",
        )
        text = format_provision(hit)

        assert "Điều 6 168/2024/NĐ-CP" in text
        assert "Xử phạt người điều khiển xe ô tô" in text


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_empty_hits_returns_no_context_message(self):
        result = build_context([])
        assert "Không có ngữ cảnh" in result

    def test_includes_date(self):
        hits = [make_hit()]
        result = build_context(hits, as_of=date(2026, 1, 15))

        assert "2026-01-15" in result

    def test_includes_provision_count(self):
        hits = [make_hit(uid="a"), make_hit(uid="b"), make_hit(uid="c")]
        result = build_context(hits)

        assert "3 điều" in result

    def test_includes_all_provisions(self):
        hits = [
            make_hit(uid="168/2024/NĐ-CP::article::6", content="Nội dung 1"),
            make_hit(uid="168/2024/NĐ-CP::article::7", content="Nội dung 2"),
        ]
        result = build_context(hits)

        assert "Nội dung 1" in result
        assert "Nội dung 2" in result
        # context uses citation format (Điều 6) not raw UID
        assert "Điều 6" in result
        assert "Điều 7" in result

    def test_provisions_are_numbered(self):
        hits = [make_hit(uid="a"), make_hit(uid="b")]
        result = build_context(hits)

        assert "1." in result
        assert "2." in result


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    def test_includes_question_and_context(self):
        prompt = build_user_prompt("Phạt bao nhiêu?", "Nội dung ngữ cảnh")

        assert "[Ngữ cảnh]" in prompt
        assert "Nội dung ngữ cảnh" in prompt
        assert "[Câu hỏi của người dùng]" in prompt
        assert "Phạt bao nhiêu?" in prompt

    def test_ends_with_instruction(self):
        prompt = build_user_prompt("question", "context")

        assert "trích dẫn đúng" in prompt


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_prompt_is_vietnamese(self):
        assert "pháp luật" in SYSTEM_PROMPT
        assert "giao thông" in SYSTEM_PROMPT

    def test_prompt_has_rules(self):
        assert "TRUNG THÀNH" in SYSTEM_PROMPT
        assert "TRÍCH DẪN" in SYSTEM_PROMPT

    def test_prompt_has_citation_instructions(self):
        assert "Điểm" in SYSTEM_PROMPT
        assert "Khoản" in SYSTEM_PROMPT
        assert "Điều" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# API app (with mocked state)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_app():
    """TestClient with a mocked AppState that does not need Neo4j or LLM."""
    with patch("vtlaw.api.app.get_state") as mock_get_state:
        mock_state = MagicMock()
        mock_state.settings.llm_api_key = ""
        mock_state.settings.embed_model = "test-model"
        mock_state.settings.llm_model = "test-llm"
        mock_state.settings.rerank_top = 30
        mock_state.settings.context_k = 8
        # Explicit: a bare MagicMock attribute is truthy, which would switch on
        # API-key auth with a key no test could supply.
        mock_state.settings.api_key = ""
        mock_state.settings.rate_limit_per_minute = 1000
        mock_state.settings.intent_router_enabled = False
        mock_state.llm_configured = False
        mock_state.graph = MagicMock()

        mock_cache = MagicMock()
        mock_cache.get_retrieval.return_value = None
        mock_cache.get_answer.return_value = None
        mock_state.cache = mock_cache

        mock_retriever = MagicMock()
        mock_retriever.search_and_rerank.return_value = MagicMock(
            hits=[
                make_hit(
                    uid="168/2024/NĐ-CP::article::6::clause::3::point::a",
                    score=0.8,
                    content="Điều khiển xe chạy quá tốc độ",
                    label="Point",
                )
            ],
            retrieval_score="hybrid",
            reranked=False,
        )
        mock_state.retriever = mock_retriever
        mock_state.generator = None
        mock_state.router = None
        mock_state.rewriter = None
        mock_state.decomposer = None
        mock_state.graph_queries = None

        mock_get_state.return_value = mock_state

        from vtlaw.api.app import app

        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    def test_returns_status_ok(self, mock_app):
        response = mock_app.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["neo4j"] == "connected"
        assert data["llm_configured"] is False


class TestReadinessEndpoint:
    @staticmethod
    def _state(*, ready: bool):
        state = MagicMock()
        state.graph.readiness.return_value = {
            "ready": ready,
            "indexes": [],
            "embedding_coverage": [],
        }
        return state

    @pytest.mark.parametrize(
        ("ready", "expected_status"),
        [(True, 200), (False, 503)],
    )
    def test_returns_graph_readiness_status(self, ready, expected_status):
        from vtlaw.api.app import app

        state = self._state(ready=ready)
        with patch("vtlaw.api.app.get_state", return_value=state):
            response = TestClient(app).get("/readyz")

        assert response.status_code == expected_status
        assert response.json()["ready"] is ready
        state.graph.readiness.assert_called_once_with()

    def test_graph_readiness_requires_indexes_and_complete_embeddings(self):
        from vtlaw.config import Settings
        from vtlaw.graph.client import GraphClient

        indexes = [
            {"name": f"{label}_{kind}", "type": index_type, "state": "ONLINE"}
            for label in ("article", "clause", "point")
            for kind, index_type in (("embedding", "VECTOR"), ("fulltext", "FULLTEXT"))
        ]
        coverage = [
            {"label": label, "total": 2, "embedded": 2}
            for label in ("Article", "Clause", "Point")
        ]
        graph = GraphClient(Settings(neo4j_password="test"))

        with patch.object(graph, "index_states", return_value=indexes), patch(
            "vtlaw.embed.embedding_coverage", return_value=coverage
        ):
            assert graph.readiness()["ready"] is True

            indexes[0]["state"] = "POPULATING"
            assert graph.readiness()["ready"] is False

            indexes[0]["state"] = "ONLINE"
            indexes[0]["type"] = "FULLTEXT"
            assert graph.readiness()["ready"] is False

            indexes[0]["type"] = "VECTOR"
            coverage[0]["embedded"] = 1
            assert graph.readiness()["ready"] is False


class TestChatEndpoint:
    def test_returns_answer_and_sources(self, mock_app):
        response = mock_app.post(
            "/chat",
            json={"question": "không đội mũ bảo hiểm phạt bao nhiêu"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "không đội mũ bảo hiểm phạt bao nhiêu"
        assert len(data["retrieved_candidates"]) == 1
        assert (
            data["retrieved_candidates"][0]["uid"]
            == "168/2024/NĐ-CP::article::6::clause::3::point::a"
        )
        assert data["strategy"] == "hybrid"

    def test_rejects_empty_question(self, mock_app):
        response = mock_app.post("/chat", json={"question": ""})

        assert response.status_code == 422  # Pydantic validation error

    def test_rejects_question_over_2000_chars(self, mock_app):
        response = mock_app.post(
            "/chat", json={"question": "x" * 2001}
        )

        assert response.status_code == 422

    def test_accepts_strategy_parameter(self, mock_app):
        response = mock_app.post(
            "/chat",
            json={
                "question": "test",
                "strategy": "vector",
            },
        )

        assert response.status_code == 200

    def test_accepts_as_of_date(self, mock_app):
        response = mock_app.post(
            "/chat",
            json={
                "question": "test",
                "as_of": "2026-01-15",
            },
        )

        assert response.status_code == 200

    def test_source_includes_citation(self, mock_app):
        response = mock_app.post("/chat", json={"question": "test"})

        data = response.json()
        source = data["retrieved_candidates"][0]
        assert "citation" in source
        assert "Điều" in source["citation"]

    def test_cache_operations_run_in_threadpool(self, mock_app):
        async def invoke(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch("vtlaw.api.app.run_in_threadpool", side_effect=invoke) as run:
            response = mock_app.post("/chat", json={"question": "test"})

        assert response.status_code == 200
        dispatched = {getattr(call.args[0], "_mock_name", None) for call in run.call_args_list}
        assert {
            "get_retrieval",
            "set_retrieval",
            "get_answer",
            "set_answer",
        } <= dispatched


class TestCors:
    def test_uses_origins_loaded_by_settings_from_dotenv(self, tmp_path):
        from vtlaw.api.app import app
        from vtlaw.config import Settings

        env_file = tmp_path / ".env"
        env_file.write_text(
            "NEO4J_PASSWORD=test\nCORS_ORIGINS=https://law.example, https://admin.example\n",
            encoding="utf-8",
        )
        settings = Settings(_env_file=env_file)

        with (
            patch("vtlaw.api.app.get_settings", return_value=settings),
            patch("vtlaw.api.app.get_state", return_value=MagicMock()),
            patch("vtlaw.api.app._state", None),
            TestClient(app) as client,
        ):
            response = client.options(
                "/chat",
                headers={
                    "Origin": "https://admin.example",
                    "Access-Control-Request-Method": "POST",
                },
            )

        assert settings.cors_origin_list() == [
            "https://law.example",
            "https://admin.example",
        ]
        assert response.headers["access-control-allow-origin"] == "https://admin.example"


class TestStreamCompletionMetrics:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("response", "outcome"),
        [
            (
                {
                    "question": "q",
                    "answer": "verified",
                    "retrieved_candidates": [],
                    "cited_sources": [],
                    "strategy": "hybrid",
                    "reranked": False,
                    "profile": "baseline",
                    "citation_status": "verified",
                },
                "verified",
            ),
            (RuntimeError("pipeline failed"), "error"),
        ],
    )
    async def test_closing_after_done_keeps_terminal_outcome(self, response, outcome):
        from vtlaw.api.app import ChatRequest, ChatResponse, _stream_chat

        run_result = (
            AsyncMock(side_effect=response)
            if isinstance(response, Exception)
            else AsyncMock(return_value=ChatResponse(**response))
        )
        with patch("vtlaw.api.app._run_chat", run_result), patch(
            "vtlaw.api.app.observe_stream_completion"
        ) as observe:
            stream = _stream_chat(ChatRequest(question="q"), MagicMock())
            while "event: done" not in await anext(stream):
                pass
            await stream.aclose()

        observe.assert_called_once_with(outcome)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("response", "outcome"),
        [
            (
                {
                    "question": "q",
                    "answer": "verified",
                    "retrieved_candidates": [],
                    "cited_sources": [],
                    "strategy": "hybrid",
                    "reranked": False,
                    "profile": "baseline",
                    "citation_status": "verified",
                },
                "verified",
            ),
            (RuntimeError("pipeline failed"), "error"),
        ],
    )
    async def test_records_terminal_outcome(self, response, outcome):
        from vtlaw.api.app import ChatRequest, ChatResponse, _stream_chat

        run_result = (
            AsyncMock(side_effect=response)
            if isinstance(response, Exception)
            else AsyncMock(return_value=ChatResponse(**response))
        )
        with patch("vtlaw.api.app._run_chat", run_result), patch(
            "vtlaw.api.app.observe_stream_completion"
        ) as observe:
            _ = [event async for event in _stream_chat(ChatRequest(question="q"), MagicMock())]

        observe.assert_called_once_with(outcome)

    @pytest.mark.asyncio
    async def test_records_cancelled_when_consumer_closes_early(self):
        from vtlaw.api.app import ChatRequest, _stream_chat

        with patch("vtlaw.api.app.observe_stream_completion") as observe:
            stream = _stream_chat(ChatRequest(question="q"), MagicMock())
            await anext(stream)
            await stream.aclose()

        observe.assert_called_once_with("cancelled")


# ---------------------------------------------------------------------------
# Metrics endpoint (Prometheus)
# ---------------------------------------------------------------------------


@pytest.fixture
def client_without_state():
    """TestClient without AppState initialization (avoids Redis/Neo4j)."""
    from fastapi.testclient import TestClient


    class MockSettings:
        llm_api_key = ""
        embed_model = "test"
        llm_model = "test"
        rerank_top = 30
        context_k = 8
        api_key = ""
        intent_router_enabled = False

    from unittest.mock import MagicMock, patch

    with patch("vtlaw.api.app.get_state") as mock_get_state, \
         patch("vtlaw.cache.Cache"):
        mock_state = MagicMock()
        mock_state.settings = MockSettings()
        mock_state.graph = MagicMock()
        mock_state.embedder = MagicMock()
        mock_state.retriever = MagicMock()
        mock_state.generator = None
        mock_state.cache = MagicMock()
        mock_state.llm_configured = False
        mock_get_state.return_value = mock_state

        from vtlaw.api.app import app
        # yield, not return: returning exits the `with` block, so get_state is
        # unpatched by the time the test runs and /health hits real Neo4j.
        yield TestClient(app)


class TestMetricsEndpoint:
    def test_returns_prometheus_format(self, client_without_state):
        response = client_without_state.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "# HELP" in content
        assert "vtlaw_requests_total" in content or "requests_total{" in content

    def test_tracks_request_count(self, client_without_state):
        """Each request increments requests_total counter."""
        # Make a request
        r = client_without_state.get("/health")
        assert r.status_code == 200

        # Check that metric was tracked
        r_metrics = client_without_state.get("/metrics")
        content = r_metrics.text
        assert 'vtlaw_requests_total' in content or 'requests_total{' in content

    def test_unmatched_paths_use_a_single_metrics_label(self):
        """Raw 404 paths must not become unbounded Prometheus labels."""
        from vtlaw.api.app import app

        with patch("vtlaw.api.app.increment_requests") as increment_requests, patch(
            "vtlaw.api.app.observe_request_duration"
        ):
            response = TestClient(app).get("/not-found/unique-request-path")

        assert response.status_code == 404
        assert increment_requests.call_args.args[1] == "unmatched"
