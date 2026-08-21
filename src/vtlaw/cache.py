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
from redis.exceptions import RedisError

from vtlaw.config import Settings, get_settings

log = logging.getLogger(__name__)

# v7 uses boundary-safe key encoding and preserves retrieval metadata so cached
# responses report their execution honestly; v6 entries are not reused.
CACHE_KEY_VERSION = "v7"
# Only answer entries need a new format: retrieval entries remain warm.
ANSWER_CACHE_VERSION = "evidence-v1"

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
    """Hash boundary-safe cache-key parts."""
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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


@dataclass(frozen=True)
class CachedAnswer:
    """Generated text plus the exact evidence rendered into its prompt."""

    text: str
    evidence_uids: frozenset[str]


class Cache:
    """Redis-backed cache for retrieval and answer caching."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._redis: redis.Redis | None = None
        self._corpus_fingerprint = _snapshot_fingerprint()

    @property
    def redis(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                self._settings.redis_url,
                decode_responses=True,
                socket_timeout=self._settings.redis_socket_timeout_s,
                socket_connect_timeout=self._settings.redis_connect_timeout_s,
            )
        return self._redis

    def _read(self, key: str) -> str | None:
        try:
            return self.redis.get(key)
        except (RedisError, UnicodeError):
            log.warning("cache read failed; treating as miss", exc_info=True)
            return None

    def _write(self, key: str, ttl: int, value: str) -> None:
        try:
            self.redis.setex(key, ttl, value)
        except RedisError:
            log.warning("cache write failed; continuing without cache", exc_info=True)

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
        cached = self._read(key)
        if cached:
            try:
                payload = json.loads(cached)
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                hits = payload.get("hits")
                reranked = payload.get("reranked", False)
                sub_queries = payload.get("sub_queries")
                if (
                    isinstance(hits, list)
                    and all(
                        isinstance(hit, dict)
                        and isinstance(hit.get("uid"), str)
                        and isinstance(hit.get("score"), (int, float))
                        and not isinstance(hit.get("score"), bool)
                        and isinstance(hit.get("doc_identity"), str)
                        and hit.get("label") in ("Article", "Clause", "Point")
                        and isinstance(hit.get("content"), str)
                        and (
                            hit.get("title") is None
                            or isinstance(hit.get("title"), str)
                        )
                        for hit in hits
                    )
                    and isinstance(reranked, bool)
                    and (
                        sub_queries is None
                        or (
                            isinstance(sub_queries, list)
                            and all(isinstance(query, str) for query in sub_queries)
                        )
                    )
                ):
                    CACHE_HITS.labels(type="retrieval").inc()
                    log.debug("retrieval cache HIT")
                    return CachedRetrieval(
                        hits=hits,
                        reranked=reranked,
                        sub_queries=sub_queries,
                    )
            log.debug("invalid retrieval cache entry; treating as miss")
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
        self._write(
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

    def _answer_key(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        profile: str,
        as_of: date | None,
    ) -> str:
        return "ans:" + _hash_key(
            self._result_key_parts(
                query, strategy, rerank_top, context_k, profile, as_of
            )
            + [
                ANSWER_CACHE_VERSION,
                self._settings.llm_base_url,
                self._settings.llm_model,
            ]
        )

    def get_answer(
        self,
        query: str,
        strategy: str,
        rerank_top: int | None,
        context_k: int,
        *,
        profile: str = "baseline",
        as_of: date | None = None,
    ) -> CachedAnswer | None:
        """Get cached answer, or None if cache miss."""
        key = self._answer_key(query, strategy, rerank_top, context_k, profile, as_of)
        cached = self._read(key)
        if cached:
            try:
                payload = json.loads(cached)
            except (TypeError, ValueError):
                payload = None
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("text"), str)
                and isinstance(payload.get("evidence_uids"), list)
                and all(isinstance(uid, str) for uid in payload["evidence_uids"])
            ):
                CACHE_HITS.labels(type="answer").inc()
                log.debug("answer cache HIT")
                return CachedAnswer(
                    text=payload["text"], evidence_uids=frozenset(payload["evidence_uids"])
                )
            log.debug("answer cache entry lacks an evidence sidecar; treating as miss")
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
        evidence_uids: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Cache generated answer with TTL."""
        key = self._answer_key(query, strategy, rerank_top, context_k, profile, as_of)
        self._write(
            key,
            self._settings.answer_cache_ttl,
            json.dumps(
                {
                    "text": answer,
                    "evidence_uids": sorted(evidence_uids or ()),
                },
                ensure_ascii=False,
            ),
        )

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None
