"""Tests for the generate and API stages — no network, no LLM calls.

Tests cover:
- build_user_prompt: correct question + context formatting
- build_context: empty hits, multiple hits, temporal date
- format_provision: citation, doc_identity, content formatting
- ChatRequest/ChatResponse Pydantic models: validation, source formatting
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

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
        mock_state.llm_configured = False
        mock_state.graph = MagicMock()

        mock_cache = MagicMock()
        mock_cache.get_retrieval.return_value = None
        mock_cache.get_answer.return_value = None
        mock_state.cache = mock_cache

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = MagicMock(
            hits=[
                make_hit(
                    uid="168/2024/NĐ-CP::article::6::clause::3::point::a",
                    score=0.8,
                    content="Điều khiển xe chạy quá tốc độ",
                    label="Point",
                )
            ],
            strategy="hybrid",
        )
        mock_state.retriever = mock_retriever
        mock_state.generator = None

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


class TestChatEndpoint:
    def test_returns_answer_and_sources(self, mock_app):
        response = mock_app.post(
            "/chat",
            json={"question": "không đội mũ bảo hiểm phạt bao nhiêu"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "không đội mũ bảo hiểm phạt bao nhiêu"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["uid"] == "168/2024/NĐ-CP::article::6::clause::3::point::a"
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
        source = data["sources"][0]
        assert "citation" in source
        assert "Điều" in source["citation"]


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
        return TestClient(app)


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
