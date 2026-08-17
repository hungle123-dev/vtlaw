"""FastAPI app for the vtlaw pipeline.

Exposes:
    POST /chat    — answer a question with citations
    GET  /health  — service health check
    GET  /metrics — Prometheus metrics (requests, cache stats)

The app loads heavy components (Neo4j driver, embedding model, LLM client)
once at startup and shares them across requests. Components are optional:
if the LLM API key is empty, /chat returns the retrieved provisions without
generation, so the API works for retrieval-only demos.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from vtlaw.cache import Cache
from vtlaw.config import Settings, get_settings
from vtlaw.embed import Embedder
from vtlaw.generate.answer import AnswerGenerator
from vtlaw.generate.context_builder import build_context
from vtlaw.generate.llm_client import LLMClient
from vtlaw.graph.client import GraphClient
from vtlaw.metrics import increment_requests, observe_request_duration
from vtlaw.parse.models import uid_to_citation
from vtlaw.retrieve.search import Hit, HybridRetriever

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState(get_settings())
        _state.init()
    return _state


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.graph: GraphClient | None = None
        self.embedder: Embedder | None = None
        self.retriever: HybridRetriever | None = None
        self.generator: AnswerGenerator | None = None
        self.cache: Cache | None = None
        self.llm_configured = bool(settings.llm_api_key)

    def init(self) -> None:
        self.graph = GraphClient(self.settings)
        self.graph.verify()

        self.embedder = Embedder(self.settings)
        self.retriever = HybridRetriever(self.graph, self.embedder, self.settings)
        self.cache = Cache(self.settings)

        if self.llm_configured:
            llm = LLMClient(self.settings)
            self.generator = AnswerGenerator(
                self.graph, self.embedder, llm, self.settings
            )

    def close(self) -> None:
        if self.graph:
            self.graph.close()
        if self.cache:
            self.cache.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("initialising app state...")
    get_state()
    yield
    if _state:
        _state.close()
    log.info("shutdown complete")


app = FastAPI(
    title="vtlaw",
    description="Vietnamese traffic-law graph RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: callable) -> Response:
    route = request.url.path
    method = request.method
    start = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        duration = time.time() - start
        increment_requests(method, route, status_code)
        observe_request_duration(route, duration)

    return response


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: Literal["hybrid", "vector", "bm25"] = "hybrid"
    as_of: date | None = None


class SourceItem(BaseModel):
    uid: str
    citation: str
    label: str
    doc_identity: str
    score: float
    snippet: str = Field(default="", description="First 200 chars of content")


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    strategy: str
    reranked: bool


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    embed_model: str
    llm_model: str
    llm_configured: bool


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    state = get_state()
    return HealthResponse(
        status="ok" if state.graph else "initializing",
        neo4j="connected" if state.graph else "not connected",
        embed_model=state.settings.embed_model,
        llm_model=state.settings.llm_model,
        llm_configured=state.llm_configured,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    state = get_state()
    cache = state.cache
    ck = (req.question, req.strategy, state.settings.rerank_top, state.settings.context_k)

    # --- Retrieval, cached ---------------------------------------------------
    cached_hits = cache.get_retrieval(*ck) if cache else None
    if cached_hits is not None:
        # get_retrieval already returns decoded dicts, one per Hit field.
        hits = [Hit(**row) for row in cached_hits]
        reranked = False
    else:
        result = state.retriever.search_and_rerank(
            req.question,
            k=state.settings.context_k,
            strategy=req.strategy,
            rerank_top=state.settings.rerank_top,
            heuristic_rerank=True,
            as_of=req.as_of,
        ) if state.llm_configured else state.retriever.search(
            req.question, k=state.settings.context_k, strategy=req.strategy
        )
        hits = list(result.hits)
        reranked = getattr(result, "reranked", False)
        if cache and hits:
            cache.set_retrieval(*ck, [asdict(h) for h in hits])

    # --- Generation, cached --------------------------------------------------
    text = cache.get_answer(*ck) if cache else None
    if text is None:
        if state.generator:
            text = await run_in_threadpool(
                state.generator.generate_from_hits,
                req.question,
                hits,
                as_of=req.as_of,
            )
        else:
            text = "(LLM not configured) Retrieved provisions:\n\n" + build_context(
                hits, as_of=req.as_of
            )
        if cache and text:
            cache.set_answer(*ck, text)

    return ChatResponse(
        question=req.question,
        answer=text,
        sources=[
            SourceItem(
                uid=h.uid,
                citation=uid_to_citation(h.uid),
                label=h.label,
                doc_identity=h.doc_identity,
                score=h.score,
                snippet=h.content[:200],
            )
            for h in hits
        ],
        strategy=req.strategy,
        reranked=reranked,
    )
