"""Runtime configuration. Every value comes from the environment or `.env`.

Nothing here is hardcoded to a machine, and nothing secret has a default.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model revisions are pinned so a silently-updated upstream checkpoint cannot
# change embeddings underneath an index that was built with the old weights.
EMBED_MODEL_REVISION = "84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0"
RERANK_MODEL_REVISION = "f536976248403314225d7fdfdbc87f0e9516a54e"

EMBED_DIM = 768
# The real limit from `sentence_bert_config.json`. The tokenizer config reports
# `model_max_length` of ~1e30, a sentinel; trusting it lets oversized text pass a
# length check and get silently truncated inside the encoder.
EMBED_MAX_TOKENS = 256


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    neo4j_uri: str = "bolt://localhost:17687"
    neo4j_user: str = "neo4j"
    neo4j_password: str
    neo4j_database: str = "neo4j"

    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "google/gemini-2.5-flash"
    llm_timeout_s: float = 60.0

    # Rate-limit retry. Gemini's free tier allows 10 requests/minute, and a burst
    # past it answers 429 with the wait in `retryDelay` — so a 429 is a "come back
    # shortly", not a failure. Without a retry, a loop over a dataset gives up on
    # the first burst and a live /chat request 500s on a transient limit.
    llm_max_retries: int = Field(default=5, ge=0, le=10)
    llm_retry_base_s: float = Field(default=2.0, gt=0.0, le=60.0)

    embed_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    embed_model_revision: str = EMBED_MODEL_REVISION
    rerank_model: str = "AITeamVN/Vietnamese_Reranker"
    rerank_model_revision: str = RERANK_MODEL_REVISION
    rerank_enabled: bool = False
    rerank_min_available_memory_mb: int = Field(default=2048, ge=256, le=65_536)
    embed_device: Literal["auto", "cpu", "cuda"] = "auto"
    embed_batch_size: int = 32

    api_host: str = "127.0.0.1"
    api_port: int = 18080

    # API surface controls. The default host is loopback, so an unauthenticated
    # deployment stays local until someone deliberately binds it wider.
    #
    # api_key gates /chat. Empty means open — correct for local development, and
    # logged as a warning at startup so an open deployment is never silent. /chat
    # spends money (LLM tokens) and CPU (embedding + graph traversal) per request,
    # so an open endpoint on a public address is an invitation to drain both.
    api_key: str = ""
    # Per-client request budget for /chat, enforced in-process.
    rate_limit_per_minute: int = Field(default=20, ge=1, le=10_000)
    # Browser origins allowed to call the API. Empty means no CORS headers at
    # all, which blocks browser clients — deliberate, since "*" plus an API key
    # header is a combination browsers refuse anyway.
    cors_origins: str = ""

    # Redis cache
    redis_url: str = "redis://localhost:16379/0"
    redis_socket_timeout_s: float = Field(default=1.0, gt=0.0, le=60.0)
    redis_connect_timeout_s: float = Field(default=1.0, gt=0.0, le=60.0)
    answer_cache_ttl: int = 3600      # seconds — full answers
    retrieval_cache_ttl: int = 1800   # seconds — retrieval results

    # Retrieval budget. Server-side only — never overridable per request, so a
    # caller cannot ask for an unbounded scan.
    fetch_k: int = Field(default=30, ge=1, le=200)
    rerank_top: int = Field(default=30, ge=1, le=100)
    context_k: int = Field(default=8, ge=1, le=20)

    # Neo4j's vector index has no pre-filter: it returns the globally nearest k
    # and any predicate is applied afterwards. Overfetching by this factor before
    # filtering is what stops an ineligible hit from evicting an eligible one.
    overfetch_factor: int = Field(default=4, ge=1, le=20)

    # RRF fusion. A small damping constant makes rank differences meaningful in
    # this deliberately small candidate pool. Treat these as a baseline, not a
    # universal legal-retrieval optimum; evaluation records every value used.
    rrf_k: int = Field(default=10, ge=1, le=100)

    # Vector is given the stronger prior; benchmark a proposed change against
    # the recorded evaluation tracks before making it the default.
    rrf_vector_weight: float = Field(default=3.0, gt=0.0, le=10.0)
    rrf_bm25_weight: float = Field(default=1.0, gt=0.0, le=10.0)

    @model_validator(mode="after")
    def candidate_budgets_match(self) -> Settings:
        """Keep the served fetch and rerank pools on one candidate budget."""
        if self.fetch_k != self.rerank_top:
            raise ValueError("fetch_k and rerank_top must match")
        return self

    # Query decomposition adds an LLM request and makes a run non-deterministic
    # unless the provider/model/prompt are controlled. Keep it off for the
    # reproducible retrieval baseline; benchmark it as a separate experiment.
    decompose_queries: bool = False

    # Classifying the original turn prevents a greeting or off-topic question
    # from being rewritten into a legal query. It is only used when an LLM key
    # is configured; disabling it keeps the deterministic retrieval baseline.
    intent_router_enabled: bool = True

    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated browser origins from Settings/.env."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def resolved_device(self) -> str:
        if self.embed_device != "auto":
            return self.embed_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
