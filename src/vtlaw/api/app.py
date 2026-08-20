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
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from vtlaw.api.middleware import (
    API_KEY_HEADER,
    REQUEST_ID_HEADER,
    RequestIdMiddleware,
    SlidingWindowRateLimiter,
    require_api_key,
    warn_if_unprotected,
)
from vtlaw.cache import Cache
from vtlaw.config import Settings, get_settings
from vtlaw.embed import Embedder
from vtlaw.generate.answer import AnswerGenerator
from vtlaw.generate.citations import assess_citations
from vtlaw.generate.context_builder import build_context
from vtlaw.generate.conversation import ConversationRewriter
from vtlaw.generate.llm_client import LLMClient
from vtlaw.graph.client import GraphClient
from vtlaw.graph.query_templates import StructuredGraphQueries
from vtlaw.metrics import increment_requests, observe_request_duration
from vtlaw.parse.models import uid_to_citation
from vtlaw.retrieve.query_parser import QueryDecomposer
from vtlaw.retrieve.router import Intent, QueryRouter
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
        self.decomposer: QueryDecomposer | None = None
        self.rewriter: ConversationRewriter | None = None
        self.router: QueryRouter | None = None
        self.graph_queries: StructuredGraphQueries | None = None
        self.cache: Cache | None = None
        self.llm_configured = bool(settings.llm_api_key)

    def init(self) -> None:
        self.graph = GraphClient(self.settings)
        self.graph.verify()
        self.graph_queries = StructuredGraphQueries(self.graph)

        self.embedder = Embedder(self.settings)
        self.retriever = HybridRetriever(self.graph, self.embedder, self.settings)
        self.cache = Cache(self.settings)

        if self.llm_configured:
            llm = LLMClient(self.settings)
            self.generator = AnswerGenerator(
                self.graph, self.embedder, llm, self.settings
            )
            self.decomposer = QueryDecomposer(llm)
            self.rewriter = ConversationRewriter(llm)
            self.router = QueryRouter(llm)

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
    state = get_state()
    warn_if_unprotected(state.settings)
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

app.add_middleware(RequestIdMiddleware)
_UI_PATH = Path(__file__).with_name("static") / "index.html"

# Read straight from the environment rather than through Settings: constructing
# Settings validates every field, including the required neo4j_password, which
# would make `import vtlaw.api.app` fail without a full .env. Importing a module
# should not require production config — only running it should.
_cors_origins = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", API_KEY_HEADER],
    )

_rate_limiter: SlidingWindowRateLimiter | None = None


def get_rate_limiter() -> SlidingWindowRateLimiter:
    """Built on first use, for the same reason as the CORS origins above."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = SlidingWindowRateLimiter(
            get_settings().rate_limit_per_minute
        )
    return _rate_limiter


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON error without leaking internals.

    FastAPI's default for an unhandled exception is an HTML traceback page, which
    both breaks JSON clients and hands an attacker the file layout. The traceback
    is logged against the request id by RequestIdMiddleware instead.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
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


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    strategy: Literal["hybrid", "vector", "bm25"] = "hybrid"
    profile: Literal["baseline", "decomposition", "rerank", "quality"] = "baseline"
    as_of: date | None = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)


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
    profile: str
    sub_queries: list[str] = Field(default_factory=list)
    resolved_question: str | None = None
    intent: Intent = "retrieve"
    timings: dict[str, float] = Field(default_factory=dict)
    graph_operation: str | None = None
    citation_status: Literal["verified", "missing", "unsupported", "not_applicable"] = (
        "not_applicable"
    )
    unsupported_citations: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    redis: str
    embed_model: str
    llm_model: str
    llm_configured: bool
    api_key_required: bool


@app.get("/", include_in_schema=False)
async def portfolio_ui() -> FileResponse:
    """Serve the dependency-free portfolio UI without initialising the RAG stack."""
    return FileResponse(_UI_PATH)


@app.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    """Report the state of each dependency, not just that the process is up.

    A health check that only answers "the server started" is worse than none: an
    orchestrator keeps routing traffic to a pod whose database went away. Each
    dependency is probed, and the response is 503 when any of them is down so a
    load balancer can act on it.
    """
    state = get_state()

    neo4j_status = "not connected"
    if state.graph:
        try:
            await run_in_threadpool(state.graph.verify)
            neo4j_status = "connected"
        except Exception as exc:  # noqa: BLE001 — the reason belongs in the body
            neo4j_status = f"error: {type(exc).__name__}"

    # The cache is optional by design: retrieval and generation both work without
    # it, only slower. So a dead Redis is reported but does not fail the check.
    redis_status = "not configured"
    if state.cache:
        try:
            await run_in_threadpool(state.cache.redis.ping)
            redis_status = "connected"
        except Exception as exc:  # noqa: BLE001
            redis_status = f"error: {type(exc).__name__}"

    healthy = neo4j_status == "connected"
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if healthy else "degraded",
        neo4j=neo4j_status,
        redis=redis_status,
        embed_model=state.settings.embed_model,
        llm_model=state.settings.llm_model,
        llm_configured=state.llm_configured,
        api_key_required=bool(state.settings.api_key),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    state = get_state()
    require_api_key(state.settings, request)

    # Rate limit per API key when one is configured, otherwise per source address.
    # Keying on the address alone would let one key behind a NAT starve the rest.
    client_id = request.headers.get(API_KEY_HEADER) or (
        request.client.host if request.client else "unknown"
    )
    allowed, retry_after = get_rate_limiter().check(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit exceeded ({state.settings.rate_limit_per_minute}/min)",
            headers={"Retry-After": str(retry_after)},
        )

    timings: dict[str, float] = {}
    intent: Intent = "retrieve"
    if state.settings.intent_router_enabled and state.router:
        started = time.perf_counter()
        intent = await run_in_threadpool(state.router.route, req.question)
        timings["route"] = round(time.perf_counter() - started, 3)

    if intent == "reject":
        return ChatResponse(
            question=req.question,
            answer=(
                "Tôi chỉ hỗ trợ tra cứu pháp luật giao thông đường bộ Việt Nam "
                "trong bộ dữ liệu hiện có."
            ),
            sources=[],
            strategy="not_applicable",
            reranked=False,
            profile=req.profile,
            intent=intent,
            timings=timings,
        )
    if intent == "direct_answer":
        return ChatResponse(
            question=req.question,
            answer=(
                "Tôi là trợ lý tra cứu pháp luật giao thông Việt Nam. "
                "Bạn có thể hỏi về quy định, mức phạt hoặc viện dẫn một điều luật."
            ),
            sources=[],
            strategy="not_applicable",
            reranked=False,
            profile=req.profile,
            intent=intent,
            timings=timings,
        )

    history = [turn.model_dump() for turn in req.history]
    resolved_question = req.question
    if history and state.rewriter:
        started = time.perf_counter()
        resolved_question = await run_in_threadpool(
            state.rewriter.rewrite, history, req.question
        )
        timings["rewrite"] = round(time.perf_counter() - started, 3)

    if intent == "cypher_query" and state.graph_queries:
        started = time.perf_counter()
        graph_answer = await run_in_threadpool(
            state.graph_queries.answer, resolved_question, as_of=req.as_of
        )
        timings["graph_query"] = round(time.perf_counter() - started, 3)
        if graph_answer:
            return ChatResponse(
                question=req.question,
                answer=graph_answer.answer,
                sources=[],
                strategy="graph_template",
                reranked=False,
                profile=req.profile,
                resolved_question=(
                    resolved_question if resolved_question != req.question else None
                ),
                intent=intent,
                timings=timings,
                graph_operation=graph_answer.operation,
            )
        # The LLM may over-classify a graph question. Preserve availability by
        # falling through to ordinary retrieval when no safe template fits.
        intent = "retrieve"

    cache = state.cache
    cache_question = "\n".join(
        [*(f"{turn['role']}:{turn['content']}" for turn in history), req.question]
    )
    ck = (
        cache_question,
        req.strategy,
        state.settings.rerank_top,
        state.settings.context_k,
    )

    # --- Retrieval, cached ---------------------------------------------------
    cached_hits = (
        cache.get_retrieval(*ck, profile=req.profile, as_of=req.as_of) if cache else None
    )
    if cached_hits is not None:
        # Cache persists these details so repeated requests do not misreport
        # the profile that produced their evidence. List support is only for
        # pre-v6 test doubles; v6 Redis entries use CachedRetrieval.
        if isinstance(cached_hits, list):
            hits = [Hit(**row) for row in cached_hits]
            reranked = False
            sub_queries = []
        else:
            hits = [Hit(**row) for row in cached_hits.hits]
            reranked = cached_hits.reranked
            sub_queries = cached_hits.sub_queries or []
    else:
        sub_queries = []
        if req.profile in ("decomposition", "quality") and state.decomposer:
            started = time.perf_counter()
            generated = await run_in_threadpool(state.decomposer.decompose, resolved_question)
            generated_queries = [sub["query"] for sub in generated]
            sub_queries = list(dict.fromkeys([resolved_question, *generated_queries]))
            timings["decompose"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        result = await run_in_threadpool(
            state.retriever.search_and_rerank,
            resolved_question,
            k=state.settings.context_k,
            strategy=req.strategy,
            rerank_top=state.settings.rerank_top,
            rerank_enabled=req.profile in ("rerank", "quality"),
            heuristic_rerank=True,
            as_of=req.as_of,
            sub_queries=sub_queries or None,
        )
        timings["retrieve"] = round(time.perf_counter() - started, 3)
        hits = list(result.hits)
        reranked = getattr(result, "reranked", False)
        if cache and hits:
            cache.set_retrieval(
                *ck,
                [asdict(h) for h in hits],
                profile=req.profile,
                as_of=req.as_of,
                reranked=reranked,
                sub_queries=sub_queries,
            )

    # --- Generation, cached --------------------------------------------------
    text = cache.get_answer(*ck, profile=req.profile, as_of=req.as_of) if cache else None
    if text is None:
        if state.generator:
            started = time.perf_counter()
            text = await run_in_threadpool(
                state.generator.generate_from_hits,
                # The resolved question, not the raw turn. Retrieval already ran
                # on the rewrite; handing the generator "còn ô tô thì sao?" makes
                # it answer a question the evidence was not gathered for.
                resolved_question,
                hits,
                as_of=req.as_of,
            )
            timings["generate"] = round(time.perf_counter() - started, 3)
        else:
            text = "(LLM not configured) Retrieved provisions:\n\n" + build_context(
                hits, as_of=req.as_of
            )
        if cache and text:
            cache.set_answer(*ck, text, profile=req.profile, as_of=req.as_of)

    citation_check = assess_citations(text, hits)

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
        profile=req.profile,
        sub_queries=sub_queries,
        resolved_question=(
            resolved_question if resolved_question != req.question else None
        ),
        intent=intent,
        timings=timings,
        citation_status=citation_check.status,
        unsupported_citations=citation_check.unsupported,
    )
