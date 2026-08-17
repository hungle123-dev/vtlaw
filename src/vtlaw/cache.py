"""Redis-backed caching layer for retrieval results and generated answers.

Two caches, each keyed by a hash of the query plus the settings that affect
the result:

  answer_cache      — full LLM-generated answer (TTL: answer_cache_ttl)
  retrieval_cache   — retrieval results (UIDs + scores) (TTL: retrieval_cache_ttl)

Keys include the query hash plus all settings that affect the output, so a
cache miss is forced when any of those change.

Prometheus metrics are recorded on every get/set via prometheus_client.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis
from prometheus_client import Counter

from vtlaw.config import Settings, get_settings

log = logging.getLogger(__name__)

# Prometheus metrics
CACHE_HITS = Counter(
    "vtlaw_cache_hits_total",
    "Cache hits by type",
    ["type"],  # answer | retrieval
)

CACHE_MISSES = Counter(
    "vtlaw_cache_misses_total",
    "Cache misses by type",
    ["type"],  # answer | retrieval
)


def _hash_key(parts: list[str]) -> str:
    """Hash a list of strings into a cache key."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
    return h.hexdigest()[:16]


class Cache:
    """Redis-backed cache for retrieval and answer caching."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._redis: redis.Redis | None = None

    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self._settings.redis_url, decode_responses=True)
        return self._redis

    # -- retrieval cache -----------------------------------------------------

    def get_retrieval(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
    ) -> list[dict[str, Any]] | None:
        """Get cached retrieval results, or None if cache miss."""
        key = "ret:" + _hash_key([
            query,
            strategy,
            str(rerank_top or 0),
            str(context_k),
        ])
        cached = self.redis.get(key)
        if cached:
            CACHE_HITS.labels(type="retrieval").inc()
            log.debug("retrieval cache HIT")
            return json.loads(cached)
        CACHE_MISSES.labels(type="retrieval").inc()
        return None

    def set_retrieval(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        results: list[dict[str, Any]],
    ) -> None:
        """Cache retrieval results with TTL."""
        key = "ret:" + _hash_key([
            query,
            strategy,
            str(rerank_top or 0),
            str(context_k),
        ])
        self.redis.setex(key, self._settings.retrieval_cache_ttl, json.dumps(results))

    # -- answer cache --------------------------------------------------------

    def get_answer(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
    ) -> str | None:
        """Get cached answer, or None if cache miss."""
        key = "ans:" + _hash_key([
            query,
            strategy,
            str(rerank_top or 0),
            str(context_k),
        ])
        cached = self.redis.get(key)
        if cached:
            CACHE_HITS.labels(type="answer").inc()
            log.debug("answer cache HIT")
            return cached
        CACHE_MISSES.labels(type="answer").inc()
        return None

    def set_answer(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        answer: str,
    ) -> None:
        """Cache generated answer with TTL."""
        key = "ans:" + _hash_key([
            query,
            strategy,
            str(rerank_top or 0),
            str(context_k),
        ])
        self.redis.setex(key, self._settings.answer_cache_ttl, answer)
        log.info("answer cached: %s", key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None
