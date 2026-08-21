"""Cross-cutting HTTP concerns: request identity, auth, and rate limiting.

Kept out of app.py so the endpoint bodies stay about traffic law rather than
plumbing, and so each piece is testable without a TestClient.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from vtlaw.config import Settings

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach an id to every request, echo it back, and log the outcome.

    Without this a 500 in the logs cannot be tied to the request that caused it,
    which is the difference between "an error happened" and a debuggable trace.
    An inbound X-Request-ID is honoured so the id survives a proxy hop.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log with the id before the exception handler turns it into a 500,
            # so the traceback and the client's id land in the same place.
            log.exception(
                "request failed request_id=%s method=%s path=%s",
                request_id,
                request.method,
                request.url.path,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class SlidingWindowRateLimiter:
    """Per-client request budget over a 60-second sliding window.

    In-process and per-worker: two uvicorn workers each allow the configured rate,
    so the effective limit is `workers × limit`. That is the honest ceiling of
    this approach and the reason to move the counter to Redis before running more
    than one worker.

    A sliding window rather than a fixed one because a fixed window lets a client
    spend its whole budget in the last second of one window and again in the first
    second of the next — a burst of 2× the limit across a window boundary.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str, *, now: float | None = None) -> tuple[bool, int]:
        """Record a request. Returns ``(allowed, seconds_until_retry)``."""
        now = time.monotonic() if now is None else now
        window = self._hits[client]

        cutoff = now - 60.0
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) >= self._limit:
            retry_after = max(1, int(window[0] + 60.0 - now) + 1)
            return False, retry_after

        window.append(now)
        return True, 0

    def reset(self) -> None:
        self._hits.clear()


def require_api_key(settings: Settings, request: Request) -> None:
    """Reject a request without a valid API key, unless local access is open.

    An empty ``api_key`` leaves the endpoint open for local development.
    :func:`warn_if_unprotected` refuses a non-loopback startup without one.
    """
    if not settings.api_key:
        return

    supplied = request.headers.get(API_KEY_HEADER, "")
    # Length-first comparison then constant-time compare, so a wrong key cannot
    # be discovered one character at a time from response timing.
    import hmac

    if not hmac.compare_digest(supplied, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing or invalid {API_KEY_HEADER}",
        )


def warn_if_unprotected(settings: Settings) -> None:
    """Warn for local unauthenticated access and refuse public access.

    /chat spends LLM tokens and CPU per call. An unauthenticated endpoint on a
    non-loopback address can drain both, so it must not start.
    """
    if settings.api_key:
        return
    if settings.api_host in ("127.0.0.1", "localhost", "::1"):
        log.warning(
            "API_KEY is not set. /chat is unauthenticated — acceptable on %s, "
            "but set API_KEY before binding to a public address.",
            settings.api_host,
        )
        return

    raise RuntimeError(
        "API_KEY must be set when the server is bound to a non-loopback address "
        f"({settings.api_host})."
    )
