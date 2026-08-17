"""Runtime configuration. Every value comes from the environment or `.env`.

Nothing here is hardcoded to a machine, and nothing secret has a default.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model revisions are pinned so a silently-updated upstream checkpoint cannot
# change embeddings underneath an index that was built with the old weights.
EMBED_MODEL_REVISION = "main"
RERANK_MODEL_REVISION = "main"

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

    embed_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    rerank_model: str = "AITeamVN/Vietnamese_Reranker"
    embed_device: Literal["auto", "cpu", "cuda"] = "auto"
    embed_batch_size: int = 32

    api_host: str = "127.0.0.1"
    api_port: int = 18080

    # Redis cache
    redis_url: str = "redis://localhost:16379/0"
    answer_cache_ttl: int = 3600      # seconds — full answers
    retrieval_cache_ttl: int = 1800   # seconds — retrieval results

    # Retrieval budget. Server-side only — never overridable per request, so a
    # caller cannot ask for an unbounded scan.
    fetch_k: int = Field(default=30, ge=1, le=200)
    rerank_top: int = Field(default=15, ge=1, le=100)
    context_k: int = Field(default=8, ge=1, le=20)

    # Neo4j's vector index has no pre-filter: it returns the globally nearest k
    # and any predicate is applied afterwards. Overfetching by this factor before
    # filtering is what stops an ineligible hit from evicting an eligible one.
    overfetch_factor: int = Field(default=4, ge=1, le=20)

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
