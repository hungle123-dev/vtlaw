"""Prometheus metrics for the vtlaw RAG API.

Exports standard Prometheus format at /metrics endpoint.

Metrics exported:
  - requests_total{method, route, status_code} — total HTTP requests
  - request_duration_seconds{route} — histogram of request durations
  - cache_hits_total{type} — answer/retrieval cache hits
  - cache_misses_total{type} — cache misses
  - retrieval_latency_seconds — retrieval latency histogram
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Histogram, generate_latest

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP Metrics
# ---------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "requests_total",
    "Total HTTP requests by method, route and status code",
    ["method", "route", "status_code"],
)

REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["route"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


# ---------------------------------------------------------------------------
# Cache Metrics
# ---------------------------------------------------------------------------

CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hits by type",
    ["type"],  # answer | retrieval
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Cache misses by type",
    ["type"],  # answer | retrieval
)


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "Retrieval latency in seconds",
    ["strategy"],  # hybrid | vector | bm25
)


def get_metrics_endpoint():
    """Return a function that returns Prometheus-formatted text."""
    def _endpoint():
        return generate_latest()

    return _endpoint


# ---------------------------------------------------------------------------
# Convenience functions used during application runtime
# ---------------------------------------------------------------------------

def increment_requests(method: str, route: str, status_code: int) -> None:
    REQUESTS_TOTAL.labels(method=method, route=route, status_code=str(status_code)).inc()


def observe_request_duration(route: str, duration: float) -> None:
    REQUEST_DURATION.labels(route=route).observe(duration)


def record_cache_hit(cache_type: str) -> None:
    CACHE_HITS.labels(type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    CACHE_MISSES.labels(type=cache_type).inc()


def observe_retrieval_latency(strategy: str, latency: float) -> None:
    RETRIEVAL_LATENCY.labels(strategy=strategy).observe(latency)
