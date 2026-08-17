"""Tests for LLMClient's rate-limit retry.

A 429 from Gemini is a "come back in N seconds", not a failure — the free tier
allows 10 requests/minute and reports the wait it wants in the error body. Before
this retry existed, a loop over the 94-question dataset lost 77 of its calls to
one burst, and a live /chat request 500'd on a transient limit.
"""

from __future__ import annotations

import httpx
import pytest
from openai import RateLimitError

from vtlaw.generate.llm_client import MAX_SERVER_DELAY_S, LLMClient, _suggested_delay


def _rate_limit_error(message: str = "Rate limited") -> RateLimitError:
    """A RateLimitError shaped like the real one, without touching the network."""
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return RateLimitError(message, response=response, body=None)


def _client(monkeypatch, *, max_retries=3, base=0.01):
    from vtlaw.config import Settings

    settings = Settings(
        neo4j_password="x" * 8,
        llm_api_key="test",
        llm_max_retries=max_retries,
        llm_retry_base_s=base,
    )
    client = LLMClient(settings)
    slept: list[float] = []
    monkeypatch.setattr("vtlaw.generate.llm_client.time.sleep", slept.append)
    return client, slept


def _responses(client, monkeypatch, sequence):
    """Drive chat.completions.create through `sequence`, raising or returning."""
    calls = {"n": 0}

    def fake_create(**_kwargs):
        item = sequence[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)
    return calls


def _ok(text="đáp án"):
    class _Msg:
        content = text

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    return _Resp()


class TestSuggestedDelay:
    def test_reads_the_providers_own_hint(self):
        """Gemini says "Please retry in 52.793060508s" — use that, don't guess."""
        err = _rate_limit_error("Quota exceeded. Please retry in 52.793060508s.")

        assert _suggested_delay(err) == pytest.approx(52.793, abs=0.01)

    def test_caps_an_absurd_hint(self):
        """A multi-minute wait means the daily quota is gone; don't block on it."""
        err = _rate_limit_error("Please retry in 3600s.")

        assert _suggested_delay(err) == MAX_SERVER_DELAY_S

    def test_returns_none_without_a_hint(self):
        assert _suggested_delay(_rate_limit_error("Too many requests")) is None


class TestRetry:
    def test_succeeds_after_a_rate_limit(self, monkeypatch):
        client, _ = _client(monkeypatch)
        _responses(client, monkeypatch, [_rate_limit_error(), _ok("phạt 400.000 đồng")])

        assert client.complete([{"role": "user", "content": "q"}]) == "phạt 400.000 đồng"

    def test_waits_the_hinted_duration(self, monkeypatch):
        client, slept = _client(monkeypatch)
        _responses(
            client,
            monkeypatch,
            [_rate_limit_error("Please retry in 12.5s."), _ok()],
        )

        client.complete([{"role": "user", "content": "q"}])

        assert 12.5 <= slept[0] <= 13.0  # hint plus a little jitter

    def test_backs_off_exponentially_without_a_hint(self, monkeypatch):
        client, slept = _client(monkeypatch, max_retries=3, base=1.0)
        _responses(
            client,
            monkeypatch,
            [_rate_limit_error(), _rate_limit_error(), _rate_limit_error(), _ok()],
        )

        client.complete([{"role": "user", "content": "q"}])

        assert len(slept) == 3
        assert slept[0] < slept[1] < slept[2]

    def test_raises_after_exhausting_retries(self, monkeypatch):
        client, slept = _client(monkeypatch, max_retries=2)
        _responses(client, monkeypatch, [_rate_limit_error()] * 3)

        with pytest.raises(RateLimitError):
            client.complete([{"role": "user", "content": "q"}])

        assert len(slept) == 2  # slept between attempts, not after the last

    def test_no_retry_when_disabled(self, monkeypatch):
        client, slept = _client(monkeypatch, max_retries=0)
        _responses(client, monkeypatch, [_rate_limit_error()])

        with pytest.raises(RateLimitError):
            client.complete([{"role": "user", "content": "q"}])

        assert slept == []

    def test_does_not_sleep_on_success(self, monkeypatch):
        client, slept = _client(monkeypatch)
        _responses(client, monkeypatch, [_ok()])

        client.complete([{"role": "user", "content": "q"}])

        assert slept == []

    def test_other_errors_are_not_retried(self, monkeypatch):
        """Only 429 is transient. A bad request retried is a bad request twice."""
        client, slept = _client(monkeypatch)
        _responses(client, monkeypatch, [ValueError("bad request")])

        with pytest.raises(ValueError):
            client.complete([{"role": "user", "content": "q"}])

        assert slept == []
