"""Tests for the API's cross-cutting concerns: request id, auth, rate limiting.

These cover the difference between "the endpoint answers" and "the endpoint is
safe to expose". /chat spends LLM tokens and CPU per call, so an open, unlimited
endpoint is a way to drain someone's budget.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from vtlaw.api.middleware import (
    API_KEY_HEADER,
    REQUEST_ID_HEADER,
    SlidingWindowRateLimiter,
    require_api_key,
    warn_if_unprotected,
)
from vtlaw.config import Settings
from vtlaw.retrieve.search import Hit

HIT = Hit(
    uid="100/2019/NĐ-CP::article::6::clause::3::point::i",
    score=0.9,
    doc_identity="100/2019/NĐ-CP",
    label="Point",
    content="Không đội mũ bảo hiểm",
    title=None,
)


def _request(headers: dict[str, str] | None = None) -> Request:
    """A Request with just enough scope for the header checks."""
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "headers": raw, "method": "POST", "path": "/chat"})


def _settings(**overrides) -> Settings:
    return Settings(neo4j_password="x" * 8, **overrides)


def _state(*, api_key="", rate_limit=1000):
    state = MagicMock()
    state.settings.rerank_top = 15
    state.settings.context_k = 8
    state.settings.embed_model = "e"
    state.settings.llm_model = "m"
    state.settings.api_key = api_key
    state.settings.rate_limit_per_minute = rate_limit
    state.llm_configured = False
    state.graph = MagicMock()
    state.cache = None
    state.retriever.search.return_value = MagicMock(hits=[HIT], strategy="hybrid")
    state.generator = None
    return state


@pytest.fixture
def client_factory():
    """TestClient with `get_state` patched and a fresh rate limiter per test.

    The limiter is module-level state, so without replacing it one test's requests
    count against the next one's budget.
    """
    patchers = []

    def build(state, *, rate_limit=1000):
        # `vtlaw/api/__init__.py` does `from vtlaw.api.app import app`, which
        # rebinds the name `vtlaw.api.app` from the submodule to the FastAPI
        # instance. So `import vtlaw.api.app as m` yields the app, not the module;
        # go through sys.modules to reach the module itself.
        import sys

        import vtlaw.api.app  # noqa: F401 — ensure the submodule is imported

        app_module = sys.modules["vtlaw.api.app"]

        app_module._rate_limiter = SlidingWindowRateLimiter(rate_limit)
        patcher = patch("vtlaw.api.app.get_state", return_value=state)
        patcher.start()
        patchers.append(patcher)
        return TestClient(app_module.app, raise_server_exceptions=False)

    yield build

    for p in patchers:
        p.stop()


class TestApiKeyAuth:
    def test_open_when_no_key_configured(self):
        """An empty api_key leaves the endpoint open — the local-dev default."""
        require_api_key(_settings(api_key=""), _request())

    def test_rejects_missing_key(self):
        with pytest.raises(HTTPException) as exc:
            require_api_key(_settings(api_key="secret"), _request())

        assert exc.value.status_code == 401

    def test_rejects_wrong_key(self):
        with pytest.raises(HTTPException) as exc:
            require_api_key(
                _settings(api_key="secret"), _request({API_KEY_HEADER: "wrong"})
            )

        assert exc.value.status_code == 401

    def test_accepts_correct_key(self):
        require_api_key(
            _settings(api_key="secret"), _request({API_KEY_HEADER: "secret"})
        )

    def test_rejects_key_that_is_a_prefix(self):
        """A prefix must not pass, or a key can be guessed one character at a time."""
        with pytest.raises(HTTPException):
            require_api_key(
                _settings(api_key="secret"), _request({API_KEY_HEADER: "sec"})
            )


class TestUnprotectedWarning:
    def test_warns_on_loopback(self, caplog):
        warn_if_unprotected(_settings(api_key="", api_host="127.0.0.1"))

        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_errors_when_bound_publicly_without_a_key(self, caplog):
        """Open plus a public bind is the case that costs money; log louder."""
        warn_if_unprotected(_settings(api_key="", api_host="0.0.0.0"))

        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_silent_when_key_is_set(self, caplog):
        warn_if_unprotected(_settings(api_key="secret", api_host="0.0.0.0"))

        assert not caplog.records


class TestRateLimiter:
    def test_allows_up_to_the_limit(self):
        limiter = SlidingWindowRateLimiter(3)

        assert [limiter.check("a")[0] for _ in range(3)] == [True, True, True]

    def test_blocks_past_the_limit(self):
        limiter = SlidingWindowRateLimiter(2)
        for _ in range(2):
            limiter.check("a")

        allowed, retry_after = limiter.check("a")

        assert allowed is False
        assert retry_after > 0

    def test_clients_are_counted_separately(self):
        limiter = SlidingWindowRateLimiter(1)
        limiter.check("a")

        assert limiter.check("b")[0] is True

    def test_window_slides(self):
        """Old hits fall out, so a client is not blocked forever."""
        limiter = SlidingWindowRateLimiter(1)
        limiter.check("a", now=0.0)

        assert limiter.check("a", now=30.0)[0] is False
        assert limiter.check("a", now=61.0)[0] is True

    def test_no_burst_across_a_window_boundary(self):
        """A fixed window would allow 2x the limit either side of the boundary."""
        limiter = SlidingWindowRateLimiter(2)
        limiter.check("a", now=59.0)
        limiter.check("a", now=59.5)

        assert limiter.check("a", now=60.5)[0] is False


class TestRequestId:
    def test_response_carries_a_request_id(self, client_factory):
        client = client_factory(_state())

        response = client.post("/chat", json={"question": "q"})

        assert response.headers[REQUEST_ID_HEADER]

    def test_inbound_request_id_is_preserved(self, client_factory):
        """The id must survive a proxy hop so logs can be correlated end to end."""
        client = client_factory(_state())

        response = client.post(
            "/chat", json={"question": "q"}, headers={REQUEST_ID_HEADER: "abc123"}
        )

        assert response.headers[REQUEST_ID_HEADER] == "abc123"


class TestChatEnforcement:
    def test_chat_rejects_without_key(self, client_factory):
        client = client_factory(_state(api_key="secret"))

        assert client.post("/chat", json={"question": "q"}).status_code == 401

    def test_chat_accepts_with_key(self, client_factory):
        client = client_factory(_state(api_key="secret"))

        response = client.post(
            "/chat", json={"question": "q"}, headers={API_KEY_HEADER: "secret"}
        )

        assert response.status_code == 200

    def test_chat_rate_limited(self, client_factory):
        client = client_factory(_state(), rate_limit=2)

        codes = [
            client.post("/chat", json={"question": f"q{i}"}).status_code
            for i in range(3)
        ]

        assert codes == [200, 200, 429]

    def test_rate_limit_response_has_retry_after(self, client_factory):
        client = client_factory(_state(), rate_limit=1)
        client.post("/chat", json={"question": "q"})

        blocked = client.post("/chat", json={"question": "q"})

        assert blocked.status_code == 429
        assert int(blocked.headers["Retry-After"]) > 0
