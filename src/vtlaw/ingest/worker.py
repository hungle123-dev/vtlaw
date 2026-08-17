"""Async ingestion worker (arq, Redis-backed).

Runs the pipeline as background jobs instead of blocking a CLI invocation:

    scrape   : source API -> snapshot -> parse -> import
    parse     : re-parse an existing snapshot and import
    reindex   : re-embed provisions, e.g. after an embedding-model change

Each job is idempotent. `parse` and `import` are safe to repeat because the
importer MERGEs on uid; `reindex` skips nodes that already carry a vector.

Usage
-----
    arq vtlaw.ingest.worker.WorkerSettings

Enqueue from Python:

    from arq import create_pool
    from arq.connections import RedisSettings
    pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    await pool.enqueue_job("handle_worker_job", "reindex", {})
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from vtlaw.config import Settings, get_settings

log = logging.getLogger(__name__)


class ScrapeJob(BaseModel):
    type: str = "scrape"
    keywords: str = "giao thông đường bộ"
    max_documents: int | None = None
    throttle_seconds: float = 1.2
    output_dir: str = "data/snapshot"


class ParseJob(BaseModel):
    type: str = "parse"
    output_dir: str = "data/snapshot"


class ReindexJob(BaseModel):
    type: str = "reindex"
    output_dir: str = "data/snapshot"


Jobs = ScrapeJob | ParseJob | ReindexJob


def _graph_client(settings: Settings | None = None):
    """Lazy import: the worker module is imported by arq before Neo4j is needed."""
    from vtlaw.graph.client import GraphClient

    return GraphClient(settings or get_settings())


def _scrape_to_snapshot(
    snapshot, keywords: str, *, max_documents: int | None, throttle_s: float
):
    """Fetch documents into the snapshot. Returns the scrape report.

    Runs in a worker thread — the scrape client is synchronous httpx.
    """
    from vtlaw.scrape.client import LegalDocumentClient
    from vtlaw.scrape.scraper import Scraper

    with LegalDocumentClient(throttle_s=throttle_s) as client:
        scraper = Scraper(client, snapshot)
        return scraper.run(keywords, max_documents=max_documents)


def _failure(stats: dict[str, Any], exc: Exception, start: float) -> dict[str, Any]:
    """Record a job failure. The job returns rather than raising so arq stores a
    result a caller can inspect; arq's own retry handles transient faults."""
    stats.update({
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "duration_s": round(time.time() - start, 2),
    })
    return stats


async def handle_scrape(job: ScrapeJob) -> dict[str, Any]:
    """Scrape from the source API, parse, and import into Neo4j."""
    from vtlaw.graph.importer import import_documents
    from vtlaw.parse.corpus import parse_corpus
    from vtlaw.scrape.snapshot import Snapshot

    log.info(
        "scrape: keywords=%r max_documents=%s dir=%s",
        job.keywords, job.max_documents, job.output_dir,
    )
    start = time.time()
    stats: dict[str, Any] = {"type": "scrape", "status": "running"}

    try:
        snapshot = Snapshot(job.output_dir)
        report = await asyncio.to_thread(
            _scrape_to_snapshot,
            snapshot,
            job.keywords,
            max_documents=job.max_documents,
            throttle_s=job.throttle_seconds,
        )

        # A refusal is not a fault: stop and leave the committed snapshot usable.
        if report.blocked:
            stats.update({
                "status": "blocked",
                "detail": "source refused the request (403); not retrying",
                "duration_s": round(time.time() - start, 2),
            })
            return stats

        parsed, parse_stats = await asyncio.to_thread(parse_corpus, snapshot)
        client = _graph_client()
        try:
            imported = await asyncio.to_thread(import_documents, client, parsed)
        finally:
            client.close()

        stats.update({
            "status": "done",
            "documents_written": report.written,
            "provisions_parsed": parse_stats.provisions,
            "provisions_imported": imported.provisions,
            "duration_s": round(time.time() - start, 2),
        })
        log.info(
            "scrape done: %d documents, %d provisions in %.1fs",
            report.written, imported.provisions, stats["duration_s"],
        )
        return stats
    except Exception as exc:  # noqa: BLE001 — recorded in the job result
        log.exception("scrape job failed")
        return _failure(stats, exc, start)


async def handle_parse(job: ParseJob) -> dict[str, Any]:
    """Re-parse an existing snapshot and import it."""
    from vtlaw.graph.importer import import_documents
    from vtlaw.parse.corpus import parse_corpus
    from vtlaw.scrape.snapshot import Snapshot

    log.info("parse: dir=%s", job.output_dir)
    start = time.time()
    stats: dict[str, Any] = {"type": "parse", "status": "running"}

    try:
        snapshot = Snapshot(job.output_dir)
        parsed, parse_stats = await asyncio.to_thread(parse_corpus, snapshot)
        client = _graph_client()
        try:
            imported = await asyncio.to_thread(import_documents, client, parsed)
        finally:
            client.close()

        stats.update({
            "status": "done",
            "documents_parsed": parse_stats.documents,
            "provisions_parsed": parse_stats.provisions,
            "provisions_imported": imported.provisions,
            "parse_failures": len(parse_stats.failed),
            "duration_s": round(time.time() - start, 2),
        })
        log.info(
            "parse done: %d documents, %d provisions in %.1fs",
            parse_stats.documents, imported.provisions, stats["duration_s"],
        )
        return stats
    except Exception as exc:  # noqa: BLE001
        log.exception("parse job failed")
        return _failure(stats, exc, start)


async def handle_reindex(job: ReindexJob) -> dict[str, Any]:
    """Embed provisions that are missing a vector."""
    from vtlaw.embed import Embedder, embed_corpus

    log.info("reindex: dir=%s", job.output_dir)
    start = time.time()
    stats: dict[str, Any] = {"type": "reindex", "status": "running"}

    try:
        settings = get_settings()
        embedder = Embedder(settings)
        client = _graph_client(settings)
        try:
            result = await asyncio.to_thread(
                embed_corpus, client, embedder, batch_size=settings.embed_batch_size
            )
        finally:
            client.close()

        stats.update({
            "status": "done",
            "embedded": sum(result.embedded.values()),
            "already_had_vectors": sum(result.skipped.values()),
            "failed": len(result.failed),
            "duration_s": round(time.time() - start, 2),
        })
        log.info(
            "reindex done: %d embedded, %d skipped, %d failed in %.1fs",
            stats["embedded"], stats["already_had_vectors"], stats["failed"],
            stats["duration_s"],
        )
        return stats
    except Exception as exc:  # noqa: BLE001
        log.exception("reindex job failed")
        return _failure(stats, exc, start)


_HANDLERS = {
    "scrape": (ScrapeJob, handle_scrape),
    "parse": (ParseJob, handle_parse),
    "reindex": (ReindexJob, handle_reindex),
}


async def handle_worker_job(
    ctx: dict[str, Any], job_type: str, job_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """arq entry point.

    arq passes its own context as the first argument and the enqueued arguments
    after it, so the job type arrives as a parameter — reading it out of `ctx`
    (as this module previously did) always found nothing.
    """
    entry = _HANDLERS.get(job_type)
    if entry is None:
        log.error("unknown job type %r", job_type)
        return {"status": "error", "error": f"unknown job type: {job_type!r}"}

    model, handler = entry
    log.info("job %s data=%s", job_type, job_data)
    try:
        job = model(**(job_data or {}))
    except ValidationError as exc:
        # Bad job payload is the caller's error, not a transient fault: return it
        # so arq records the reason instead of retrying four times over.
        log.error("invalid %s payload: %s", job_type, exc)
        return {"status": "error", "error": f"ValidationError: {exc}"}
    return await handler(job)


def _redis_settings():
    from arq.connections import RedisSettings

    return RedisSettings.from_dsn(get_settings().redis_url)


class WorkerSettings:
    """arq worker configuration.

    `redis_settings` is required — without it arq connects to localhost:6379 and
    silently misses the queue this project uses on port 16379.
    """

    functions = [handle_worker_job]
    redis_settings = _redis_settings()
    retry_jobs = True
    max_tries = 4
    keep_result = 3600 * 24 * 7
    # Ingestion runs for minutes: embedding 7,381 provisions on CPU takes longer
    # than any default job timeout would allow.
    job_timeout = 3600
