"""Tests for the Redis caching layer.

Tests cover:
- Retrieval cache: set/get, cache miss
- Answer cache: set/get, cache miss
- Cache key hashing: deterministic, different inputs → different keys
- Cache TTL: respects configured TTL

Uses unittest.mock to mock Redis, so no Redis connection needed.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from vtlaw.cache import Cache, _hash_key


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return MagicMock()


@pytest.fixture
def cache(mock_redis):
    """Cache with mocked Redis."""
    with patch("vtlaw.cache.redis") as mock_redis_module:
        mock_redis_module.from_url.return_value = mock_redis
        mock_redis_module.Redis = MagicMock(return_value=mock_redis)
        from vtlaw.config import Settings
        settings = Settings(
            neo4j_password="test",
            redis_url="redis://localhost:16379/0",
            answer_cache_ttl=3600,
            retrieval_cache_ttl=1800,
        )
        c = Cache(settings)
        c._redis = mock_redis
        return c


# ---------------------------------------------------------------------------
# Cache key hashing
# ---------------------------------------------------------------------------


class TestHashKey:
    def test_deterministic(self):
        """Same inputs produce same hash."""
        key1 = _hash_key(["query", "hybrid", "30", "8"])
        key2 = _hash_key(["query", "hybrid", "30", "8"])
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Different inputs produce different hashes."""
        key1 = _hash_key(["query1", "hybrid"])
        key2 = _hash_key(["query2", "hybrid"])
        assert key1 != key2

    def test_key_length(self):
        """Hash should be 16 chars (truncated SHA256)."""
        key = _hash_key(["test"])
        assert len(key) == 16

    def test_empty_input(self):
        """Empty string is still a valid input."""
        key = _hash_key([""])
        assert len(key) == 16


# ---------------------------------------------------------------------------
# Retrieval cache
# ---------------------------------------------------------------------------


class TestRetrievalCache:
    def test_set_and_get_retrieval(self, cache, mock_redis):
        """Should store and retrieve retrieval results."""
        results = [
            {"uid": "test::1", "score": 0.9},
            {"uid": "test::2", "score": 0.8},
        ]
        mock_redis.get.return_value = None  # first call: cache miss

        # Set
        cache.set_retrieval(
            "query", "hybrid", 30, 8, results, reranked=True, sub_queries=["query"]
        )

        # Verify setex was called with correct TTL
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[1] == 1800  # retrieval_cache_ttl
        assert json.loads(args[2]) == {
            "hits": results,
            "reranked": True,
            "sub_queries": ["query"],
        }

        # Simulate Redis returning the cached value
        mock_redis.get.return_value = args[2]

        # Get
        cached = cache.get_retrieval("query", "hybrid", 30, 8)
        assert cached is not None
        assert len(cached.hits) == 2
        assert cached.hits[0]["uid"] == "test::1"
        assert cached.reranked is True
        assert cached.sub_queries == ["query"]

    def test_cache_miss_returns_none(self, cache, mock_redis):
        """Should return None on cache miss."""
        mock_redis.get.return_value = None

        result = cache.get_retrieval("query", "hybrid", 30, 8)
        assert result is None

    def test_different_strategy_different_key(self, cache, mock_redis):
        """Different strategy should use different cache key."""
        mock_redis.get.return_value = None

        cache.set_retrieval("query", "hybrid", 30, 8, [])
        cache.set_retrieval("query", "vector", 30, 8, [])

        # Two different setex calls with different keys
        assert mock_redis.setex.call_count == 2
        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 != key2

    def test_different_k_different_key(self, cache, mock_redis):
        """Different context_k should use different cache key."""
        mock_redis.get.return_value = None

        cache.set_retrieval("query", "hybrid", 30, 8, [])
        cache.set_retrieval("query", "hybrid", 30, 10, [])

        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 != key2

    def test_different_as_of_dates_use_different_keys(self, cache, mock_redis):
        cache.set_retrieval(
            "query", "hybrid", 30, 8, [], as_of=date(2025, 1, 1)
        )
        cache.set_retrieval(
            "query", "hybrid", 30, 8, [], as_of=date(2026, 1, 1)
        )

        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 != key2

    def test_default_as_of_is_keyed_as_today(self, cache, mock_redis):
        cache.set_retrieval("query", "hybrid", 30, 8, [])
        cache.set_retrieval("query", "hybrid", 30, 8, [], as_of=date.today())

        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 == key2

    def test_rerank_setting_changes_the_retrieval_key(self, cache, mock_redis):
        cache.set_retrieval("query", "hybrid", 30, 8, [])
        cache._settings.rerank_enabled = True
        cache.set_retrieval("query", "hybrid", 30, 8, [])

        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 != key2

    def test_retrieval_profiles_use_distinct_cache_keys(self, cache, mock_redis):
        cache.set_retrieval("query", "hybrid", 30, 8, [], profile="baseline")
        baseline_key = mock_redis.setex.call_args.args[0]

        cache.set_retrieval("query", "hybrid", 30, 8, [], profile="quality")
        quality_key = mock_redis.setex.call_args.args[0]

        assert baseline_key != quality_key

    def test_overfetch_factor_changes_the_retrieval_key(self, cache, mock_redis):
        """It changes which candidates survive the date filter, so it changes results.

        Left out of the key, raising the factor keeps serving entries produced by
        the narrower one until the TTL expires.
        """
        cache.set_retrieval("query", "hybrid", 30, 8, [])
        cache._settings.overfetch_factor = 8
        cache.set_retrieval("query", "hybrid", 30, 8, [])

        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 != key2


# ---------------------------------------------------------------------------
# Answer cache
# ---------------------------------------------------------------------------


class TestAnswerCache:
    def test_answer_set_and_get(self, cache, mock_redis):
        """Should store and retrieve answer text."""
        answer = "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng."

        cache.set_answer("query", "hybrid", 30, 8, answer)

        # Verify setex was called with correct TTL
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[1] == 3600  # answer_cache_ttl

        # Simulate cache hit
        mock_redis.get.return_value = answer

        cached = cache.get_answer("query", "hybrid", 30, 8)
        assert cached == answer

    def test_cache_miss_returns_none(self, cache, mock_redis):
        """Should return None on cache miss."""
        mock_redis.get.return_value = None

        result = cache.get_answer("query", "hybrid", 30, 8)
        assert result is None

    def test_different_query_different_key(self, cache, mock_redis):
        """Different query should use different cache key."""
        cache.set_answer("query1", "hybrid", 30, 8, "answer1")
        cache.set_answer("query2", "hybrid", 30, 8, "answer2")

        key1 = mock_redis.setex.call_args_list[0][0][0]
        key2 = mock_redis.setex.call_args_list[1][0][0]
        assert key1 != key2

    def test_answer_cache_key_prefix(self, cache, mock_redis):
        """Answer cache keys should start with 'ans:'."""
        cache.set_answer("query", "hybrid", 30, 8, "answer")
        key = mock_redis.setex.call_args[0][0]
        assert key.startswith("ans:")

    def test_retrieval_cache_key_prefix(self, cache, mock_redis):
        """Retrieval cache keys should start with 'ret:'."""
        cache.set_retrieval("query", "hybrid", 30, 8, [])
        key = mock_redis.setex.call_args[0][0]
        assert key.startswith("ret:")
