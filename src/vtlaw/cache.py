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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import redis
from prometheus_client import Counter

from vtlaw.config import Settings, get_settings

log = logging.getLogger(__name__)

# v6 preserves retrieval metadata so cached responses report their execution
# honestly; v5 entries cannot make that claim.
CACHE_KEY_VERSION = "v6"

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


def _snapshot_fingerprint() -> str:
    """Fingerprint the committed corpus manifest when it is available."""
    try:
        return hashlib.sha256(
            Path("data/snapshot/manifest.json").read_bytes()
        ).hexdigest()[:16]
    except OSError:
        return "no-snapshot"


@dataclass(frozen=True)
class CachedRetrieval:
    """Hits plus the execution details a cached API response must retain."""

    hits: list[dict[str, Any]]
    reranked: bool = False
    sub_queries: list[str] | None = None


class Cache:
    """Redis-backed cache for retrieval and answer caching."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._redis: redis.Redis | None = None
        self._corpus_fingerprint = _snapshot_fingerprint()

    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self._settings.redis_url, decode_responses=True)
        return self._redis

    # -- retrieval cache -----------------------------------------------------

    def _result_key_parts(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        profile: str,
        as_of: date | None,
    ) -> list[str]:
        """Inputs that can change the retrieved provisions or their order."""
        settings = self._settings
        effective_as_of = as_of or date.today()
        return [
            CACHE_KEY_VERSION,
            self._corpus_fingerprint,
            query,
            strategy,
            profile,
            str(rerank_top or 0),
            str(context_k),
            effective_as_of.isoformat(),
            settings.embed_model,
            settings.embed_model_revision,
            str(settings.fetch_k),
            str(settings.overfetch_factor),
            str(settings.rrf_k),
            str(settings.rrf_vector_weight),
            str(settings.rrf_bm25_weight),
            str(settings.rerank_enabled),
            settings.rerank_model,
            settings.rerank_model_revision,
            str(settings.decompose_queries),
        ]

    def get_retrieval(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        *,
        profile: str = "baseline",
        as_of: date | None = None,
    ) -> CachedRetrieval | None:
        """Get cached retrieval results, or None if cache miss."""
        key = "ret:" + _hash_key(
            self._result_key_parts(query, strategy, rerank_top, context_k, profile, as_of)
        )
        cached = self.redis.get(key)
        if cached:
            CACHE_HITS.labels(type="retrieval").inc()
            log.debug("retrieval cache HIT")
            payload = json.loads(cached)
            return CachedRetrieval(
                hits=payload["hits"],
                reranked=payload.get("reranked", False),
                sub_queries=payload.get("sub_queries"),
            )
        CACHE_MISSES.labels(type="retrieval").inc()
        return None

    def set_retrieval(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        results: list[dict[str, Any]],
        *,
        profile: str = "baseline",
        as_of: date | None = None,
        reranked: bool = False,
        sub_queries: list[str] | None = None,
    ) -> None:
        """Cache retrieval results with TTL."""
        key = "ret:" + _hash_key(
            self._result_key_parts(query, strategy, rerank_top, context_k, profile, as_of)
        )
        self.redis.setex(
            key,
            self._settings.retrieval_cache_ttl,
            json.dumps(
                {
                    "hits": results,
                    "reranked": reranked,
                    "sub_queries": sub_queries or [],
                }
            ),
        )

    # -- answer cache --------------------------------------------------------

    def get_answer(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        *,
        profile: str = "baseline",
        as_of: date | None = None,
    ) -> str | None:
        """Get cached answer, or None if cache miss."""
        key = "ans:" + _hash_key(
            self._result_key_parts(query, strategy, rerank_top, context_k, profile, as_of)
            + [self._settings.llm_base_url, self._settings.llm_model]
        )
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
        *,
        profile: str = "baseline",
        as_of: date | None = None,
    ) -> None:
        """Cache generated answer with TTL."""
        key = "ans:" + _hash_key(
            self._result_key_parts(query, strategy, rerank_top, context_k, profile, as_of)
            + [self._settings.llm_base_url, self._settings.llm_model]
        )
        self.redis.setex(key, self._settings.answer_cache_ttl, answer)
        log.info("answer cached: %s", key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None
