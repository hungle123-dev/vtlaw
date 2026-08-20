"""Prometheus metrics for the vtlaw RAG API.

Exported at ``/metrics``:
  - vtlaw_requests_total{method, route, status_code} — total HTTP requests
  - vtlaw_request_duration_seconds{route} — histogram of request durations

Cache counters live in :mod:`vtlaw.cache`, next to the code that increments
them, so there is exactly one definition per metric.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Histogram

log = logging.getLogger(__name__)

# Prefixed to namespace these against the default process/GC collectors that
# prometheus_client registers, and to match the names quoted in the docs.
REQUESTS_TOTAL = Counter(
    "vtlaw_requests_total",
    "Total HTTP requests by method, route and status code",
    ["method", "route", "status_code"],
)

REQUEST_DURATION = Histogram(
    "vtlaw_request_duration_seconds",
    "Request duration in seconds",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def increment_requests(method: str, route: str, status_code: int) -> None:
    REQUESTS_TOTAL.labels(method=method, route=route, status_code=str(status_code)).inc()


def observe_request_duration(route: str, duration: float) -> None:
    REQUEST_DURATION.labels(route=route).observe(duration)
