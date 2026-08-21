"""FastAPI app for the vtlaw pipeline.

Exposes:
    POST /chat        — answer a question with citations
    POST /chat/stream — stream only the citation-checked answer
    GET  /health      — service health check
    GET  /metrics     — Prometheus metrics (requests, cache stats)

The app loads heavy components (Neo4j driver, embedding model, LLM client)
once at startup and shares them across requests. Components are optional:
if the LLM API key is empty, /chat returns the retrieved provisions without
generation, so the API works for retrieval-only demos.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
from vtlaw.generate.citations import CitationCheck, assess_citations
from vtlaw.generate.context_builder import build_context
from vtlaw.generate.conversation import ConversationRewriter
from vtlaw.generate.llm_client import LLMClient
from vtlaw.graph.client import GraphClient
from vtlaw.graph.query_templates import StructuredGraphQueries, detect_operation
from vtlaw.metrics import (
    increment_requests,
    observe_request_duration,
    observe_stream_completion,
)
from vtlaw.parse.models import uid_to_citation
from vtlaw.retrieve.query_parser import QueryDecomposer
from vtlaw.retrieve.router import Intent, QueryRouter
from vtlaw.retrieve.search import Hit, HybridRetriever, resolve_citation

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
    settings = get_settings()
    _cors_origins[:] = settings.cors_origin_list()
    warn_if_unprotected(settings)
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

app.add_middleware(RequestIdMiddleware)
_UI_PATH = Path(__file__).with_name("static") / "index.html"

# The list is populated from Settings in lifespan, after middleware construction.
# Keeping the object stable lets CORSMiddleware see .env-backed values without
# requiring production secrets merely to import this module.
_cors_origins: list[str] = []
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


def _authorize_chat(state: AppState, request: Request) -> None:
    """Apply the same API-key and rate-limit policy to every chat transport."""
    require_api_key(state.settings, request)

    # An authenticated key is stable; without API_KEY, ignore the caller-controlled
    # header and rate-limit by source address.
    client_id = (
        request.headers[API_KEY_HEADER]
        if state.settings.api_key
        else request.client.host if request.client else "unknown"
    )
    allowed, retry_after = get_rate_limiter().check(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit exceeded ({state.settings.rate_limit_per_minute}/min)",
            headers={"Retry-After": str(retry_after)},
        )


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
    route = "unmatched"
    method = request.method
    start = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        # Routing populates this during dispatch. Never use the raw path: it
        # makes 404s and future path parameters unbounded Prometheus labels.
        route = getattr(request.scope.get("route"), "path", route)
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
    retrieved_candidates: list[SourceItem]
    cited_sources: list[SourceItem]
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


def _source_item(hit: Hit) -> SourceItem:
    return SourceItem(
        uid=hit.uid,
        citation=uid_to_citation(hit.uid),
        label=hit.label,
        doc_identity=hit.doc_identity,
        score=hit.score,
        snippet=hit.content[:200],
    )


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


@app.get("/readyz")
async def readiness(response: Response) -> dict[str, object]:
    """Return 503 until retrieval indexes and provision embeddings are complete."""
    state = get_state()
    if not state.graph:
        payload: dict[str, object] = {
            "ready": False,
            "indexes": [],
            "embedding_coverage": [],
            "error": "graph not configured",
        }
    else:
        try:
            payload = await run_in_threadpool(state.graph.readiness)
        except Exception as exc:  # noqa: BLE001 — expose only the exception type
            log.warning("graph readiness check failed", exc_info=True)
            payload = {
                "ready": False,
                "indexes": [],
                "embedding_coverage": [],
                "error": type(exc).__name__,
            }
    if not payload["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


async def _run_chat(
    req: ChatRequest,
    state: AppState,
) -> ChatResponse:
    """Run the legal-RAG pipeline for either the JSON or SSE transport."""

    timings: dict[str, float] = {}
    full_citation = resolve_citation(req.question)
    graph_operation = detect_operation(req.question) if state.graph_queries else None
    intent: Intent = "retrieve"
    if graph_operation is not None:
        intent = "cypher_query"
    elif full_citation is None and state.settings.intent_router_enabled and state.router:
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
            retrieved_candidates=[],
            cited_sources=[],
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
            retrieved_candidates=[],
            cited_sources=[],
            strategy="not_applicable",
            reranked=False,
            profile=req.profile,
            intent=intent,
            timings=timings,
        )

    history = [turn.model_dump() for turn in req.history]
    resolved_question = req.question
    if history and state.rewriter and full_citation is None and graph_operation is None:
        started = time.perf_counter()
        resolved_question = await run_in_threadpool(
            state.rewriter.rewrite, history, req.question
        )
        timings["rewrite"] = round(time.perf_counter() - started, 3)
    retrieval_question = req.question if full_citation is not None else resolved_question

    if intent == "cypher_query" and state.graph_queries:
        started = time.perf_counter()
        graph_answer = await run_in_threadpool(
            state.graph_queries.answer,
            req.question if graph_operation is not None else resolved_question,
            as_of=req.as_of,
        )
        timings["graph_query"] = round(time.perf_counter() - started, 3)
        if graph_answer:
            answer = graph_answer.answer
            graph_sources = [
                source
                for source in getattr(graph_answer, "sources", ())
                if isinstance(source, Hit) and source.content.strip()
            ]
            citation_check = assess_citations(answer, graph_sources)
            if citation_check.cited and citation_check.status != "verified":
                answer = (
                    "Tôi không thể trả về kết quả truy vấn đồ thị có viện dẫn "
                    "quy định khi chưa có bằng chứng nguồn đầy đủ."
                )
                graph_sources = []
            cited_sources = [
                source
                for source in graph_sources
                if any(
                    source.uid == uid or source.uid.startswith(uid + "::")
                    for uid in citation_check.cited
                )
            ]
            return ChatResponse(
                question=req.question,
                answer=answer,
                retrieved_candidates=[_source_item(source) for source in graph_sources],
                cited_sources=[_source_item(source) for source in cited_sources],
                strategy="graph_template",
                reranked=False,
                profile=req.profile,
                resolved_question=(
                    resolved_question if resolved_question != req.question else None
                ),
                intent=intent,
                timings=timings,
                graph_operation=graph_answer.operation,
                citation_status=(
                    citation_check.status
                    if citation_check.cited
                    else "not_applicable"
                ),
                unsupported_citations=citation_check.unsupported,
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
    retrieval_strategy = req.strategy
    has_full_citation = (
        full_citation is not None or resolve_citation(resolved_question) is not None
    )
    cached_hits = None
    if cache and not has_full_citation:
        cached_hits = await run_in_threadpool(
            cache.get_retrieval, *ck, profile=req.profile, as_of=req.as_of
        )
    if cached_hits is not None:
        # Cache persists these details so repeated requests do not misreport
        # the profile that produced their evidence. List support is only for
        # legacy test doubles; Redis entries use CachedRetrieval.
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
            retrieval_question,
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
        retrieval_strategy = result.retrieval_score
        if not hits and result.retrieval_score == "exact_citation":
            return ChatResponse(
                question=req.question,
                answer=(
                    f"Không tìm thấy trích dẫn chính xác “{retrieval_question}” "
                    "trong corpus hiện có."
                ),
                retrieved_candidates=[],
                cited_sources=[],
                strategy="exact_citation",
                reranked=False,
                profile=req.profile,
                sub_queries=sub_queries,
                resolved_question=(
                    resolved_question if resolved_question != req.question else None
                ),
                intent=intent,
                timings=timings,
            )
        if cache and hits:
            await run_in_threadpool(
                cache.set_retrieval,
                *ck,
                [asdict(h) for h in hits],
                profile=req.profile,
                as_of=req.as_of,
                reranked=reranked,
                sub_queries=sub_queries,
            )

    # --- Generation, cached --------------------------------------------------
    rendered_evidence: dict[str, Hit] = {}
    cached_answer = None
    if cache:
        cached_answer = await run_in_threadpool(
            cache.get_answer, *ck, profile=req.profile, as_of=req.as_of
        )
    if cached_answer is not None:
        cached_text = cached_answer if isinstance(cached_answer, str) else cached_answer.text
        if assess_citations(cached_text, hits).cited:
            cached_answer = None
    if cached_answer is not None:
        # A string here is only compatible with legacy test doubles. Real
        # Cache entries without the sidecar are rejected as misses.
        if isinstance(cached_answer, str):
            text = cached_answer
            evidence_uids = {hit.uid for hit in hits}
        else:
            text = cached_answer.text
            evidence_uids = set(cached_answer.evidence_uids)
    else:
        evidence_uids: set[str] = set()
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
                evidence_uids=evidence_uids,
                rendered_evidence=rendered_evidence,
            )
            timings["generate"] = round(time.perf_counter() - started, 3)
        else:
            evidence_uids.update(hit.uid for hit in hits)
            text = "(LLM not configured) Retrieved provisions:\n\n" + build_context(
                hits, as_of=req.as_of
            )
            rendered_evidence.update((hit.uid, hit) for hit in hits)
        if cache and text:
            await run_in_threadpool(
                cache.set_answer,
                *ck,
                text,
                profile=req.profile,
                as_of=req.as_of,
                evidence_uids=evidence_uids,
            )

    citation_check = assess_citations(
        text,
        hits,
        evidence_uids=set(rendered_evidence),
    )
    missing_records = [
        uid for uid in citation_check.cited if uid not in rendered_evidence
    ]
    if missing_records:
        citation_check = CitationCheck(
            status="unsupported",
            cited=citation_check.cited,
            unsupported=list(
                dict.fromkeys([*citation_check.unsupported, *missing_records])
            ),
        )

    return ChatResponse(
        question=req.question,
        answer=text,
        retrieved_candidates=[_source_item(hit) for hit in hits],
        cited_sources=[
            _source_item(rendered_evidence[uid])
            for uid in citation_check.cited
            if uid in rendered_evidence
        ],
        strategy=retrieval_strategy,
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


def _sse(event: str, data: dict[str, object]) -> str:
    """Encode one Server-Sent Event without trusting user text as wire format."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_chat(req: ChatRequest, state: AppState) -> AsyncIterator[str]:
    """Type only the verified final answer after the citation guard completes."""
    outcome: Literal["verified", "error", "cancelled"] = "cancelled"
    try:
        yield _sse(
            "status",
            {"phase": "request", "message": "Đang xử lý yêu cầu..."},
        )
        try:
            chat_response = await _run_chat(req, state)
        except Exception:  # noqa: BLE001 - SSE cannot change status mid-stream
            log.exception("streaming chat failed")
            outcome = "error"
            yield _sse(
                "error", {"detail": "Không thể hoàn tất câu trả lời. Hãy thử lại."}
            )
            yield _sse("done", {})
            return

        if chat_response.intent == "reject":
            terminal_message = (
                "Yêu cầu ngoài phạm vi. Không truy xuất quy định hoặc kiểm tra trích dẫn."
            )
        elif chat_response.intent == "direct_answer":
            terminal_message = (
                "Đây là phản hồi hướng dẫn. Không truy xuất quy định hoặc kiểm tra trích dẫn."
            )
        elif (
            chat_response.strategy == "exact_citation"
            and not chat_response.retrieved_candidates
        ):
            terminal_message = "Không tìm thấy trích dẫn chính xác. Không kiểm tra trích dẫn."
        else:
            terminal_message = None
        if terminal_message:
            outcome = "error"
            yield _sse("status", {"phase": "terminal", "message": terminal_message})
            yield _sse(
                "error",
                {"detail": "Không thể xác thực câu trả lời từ các nguồn hiện có."},
            )
            yield _sse("done", {})
            return

        is_graph_answer = (
            chat_response.intent == "cypher_query"
            and chat_response.strategy == "graph_template"
        )
        yield _sse(
            "status",
            {
                "phase": "graph_query" if is_graph_answer else "retrieval",
                "message": (
                    "Đã hoàn tất truy vấn đồ thị."
                    if is_graph_answer
                    else "Đã hoàn tất truy xuất nguồn liên quan."
                ),
            },
        )
        yield _sse(
            "status",
            {
                "phase": "citation_validation",
                "message": "Đã hoàn tất kiểm tra trích dẫn.",
            },
        )
        if chat_response.citation_status != "verified" and not (
            is_graph_answer and chat_response.citation_status == "not_applicable"
        ):
            outcome = "error"
            yield _sse(
                "error",
                {"detail": "Không thể xác thực câu trả lời từ các nguồn hiện có."},
            )
            yield _sse("done", {})
            return

        for start in range(0, len(chat_response.answer), 24):
            yield _sse("delta", {"text": chat_response.answer[start : start + 24]})
        yield _sse("final", chat_response.model_dump(mode="json"))
        outcome = "verified"
        yield _sse("done", {})
    finally:
        observe_stream_completion(outcome)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    state = get_state()
    _authorize_chat(state, request)
    return await _run_chat(req, state)


@app.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events: status, verified delta, final, done.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        }
    },
)
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream statuses, then the verified answer and normal ChatResponse."""
    state = get_state()
    _authorize_chat(state, request)
    return StreamingResponse(
        _stream_chat(req, state),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
