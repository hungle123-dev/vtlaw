"""Async ingestion worker powered by arq (Redis-based async job queue).

This is the production-grade background processor that replaces synchronous
CLI ingestion. The worker runs as a separate process, pulls jobs from a Redis
queue, and executes the full pipeline: scrape → parse → embed → import.

The worker supports three job types:
    - scrape_and_import     : full pipeline (scrape → parse → embed → import)
    - parse                 : parse existing snapshots into provisions
    - reindex               : re-embed provisions after model change

Each job is idempotent: re-running it never duplicates data. Jobs are
serialized through Redis so concurrent workers don't step on each other.

Usage
-----
    # Start the worker (background process)
    arq vtlaw.ingest.worker.WorkerSettings

Health check
------------
    curl http://localhost:8080/metrics          # Prometheus metrics included
    curl http://localhost:8080/jobs             # queued/pending/resolved counts
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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


# ---------------------------------------------------------------------------
# Helper functions for worker handlers
# ---------------------------------------------------------------------------


def _make_scraper(snapshot: Snapshot, keywords: str, *, max_documents: int | None = None):
    """Build scraper instance and discover documents."""
    from ..scraper.client import LegalDocumentClient
    from ..scraper.scraper import Scraper

    client = LegalDocumentClient(
        base_url="https://phapluat.gov.vn/api/legal-documents",
        timeout=30.0,
    )
    scraper = Scraper(client=client, snapshot=snapshot)
    docs = scraper.discover(keywords, max_documents=max_documents)
    return scraper, docs


def _get_graph_client():
    """Lazy-import GraphClient."""
    from ..config import get_settings
    from ..graph.client import GraphClient
    settings = get_settings()
    return GraphClient(settings=settings)


async def handle_scrape(ctx: dict[str, Any]) -> dict[str, Any]:
    """Scrape documents and import into Neo4j.

    Returns status dict with stats.
    """
    job_data = ctx.get("job_data", {})
    job = ScrapeJob(**job_data) if job_data else ScrapeJob()
    log.info("Starting scrape: keywords=%s max_doc=%s dir=%s",
             job.keywords, job.max_documents, job.output_dir)
    start = time.time()
    stats = {"type": "scrape", "status": "running"}

    try:
        from ..graph.importer import import_documents
        from ..parse.corpus import parse_corpus
        from ..scrape.snapshot import Snapshot

        graph_client = _get_graph_client()
        snapshot = Snapshot(root=Path(job.output_dir))

        scraper, scraped_docs = await asyncio.to_thread(
            _make_scraper, snapshot, job.keywords,
            max_documents=job.max_documents,
        )

        parsed_docs, corp_stats = await asyncio.to_thread(parse_corpus, snapshot)
        result = await asyncio.to_thread(import_documents, graph_client, parsed_docs)

        stats.update({
            "status": "done",
            "documents_scraped": len(scraped_docs),
            "provisions_parsed": sum(corp_stats.values()),
            "clauses_imported": result.clauses,
            "points_imported": result.points,
            "duration_s": round(time.time() - start, 2),
        })
        log.info("Scrape+import done: %d docs, %d provisions in %.2fs",
                 len(scraped_docs), sum(corp_stats.values()), stats["duration_s"])
        return stats

    except Exception as e:
        stats.update({"status": "error", "error": str(e), "duration_s": round(time.time() - start, 2)})
        log.exception("Scrape job failed")
        return stats


async def handle_parse(ctx: dict[str, Any]) -> dict[str, Any]:
    """Parse all snapshots in output_dir into provisions."""
    job_data = ctx.get("job_data", {})
    job = ParseJob(**job_data) if job_data else ParseJob()
    log.info("Starting parse: dir=%s", job.output_dir)
    start = time.time()
    stats = {"type": "parse", "status": "running"}

    try:
        from ..graph.importer import import_documents
        from ..parse.corpus import parse_corpus
        from ..scrape.snapshot import Snapshot

        snapshot = Snapshot(root=Path(job.output_dir))
        graph_client = _get_graph_client()

        parsed_docs, corp_stats = await asyncio.to_thread(parse_corpus, snapshot)
        result = await asyncio.to_thread(import_documents, graph_client, parsed_docs)

        stats.update({
            "status": "done",
            "snapshots_found": len(list(Path(job.output_dir).glob("*.json"))),
            "documents_parsed": sum(corp_stats.values()),
            "clauses_imported": result.clauses,
            "points_imported": result.points,
            "duration_s": round(time.time() - start, 2),
        })
        log.info("Parse done: %d docs, %d provisions in %.2fs",
                 sum(corp_stats.values()), sum(result.clauses + result.points), stats["duration_s"])
        return stats

    except Exception as e:
        stats.update({"status": "error", "error": str(e), "duration_s": round(time.time() - start, 2)})
        log.exception("Parse job error")
        return stats


async def handle_reindex(ctx: dict[str, Any]) -> dict[str, Any]:
    """Re-embed all provisions after model change."""
    job_data = ctx.get("job_data", {})
    job = ReindexJob(**job_data) if job_data else ReindexJob()
    log.info("Starting reindex: dir=%s", job.output_dir)
    start = time.time()
    stats = {"type": "reindex", "status": "running"}

    try:
        from ..config import get_settings
        from ..embed.embedder import Embedder, embed_corpus

        settings = get_settings()
        graph_client = _get_graph_client()
        embedder = Embedder(settings=settings)

        result = await asyncio.to_thread(embed_corpus, graph_client, embedder, batch_size=256)
        total_embedded = sum(result.embedded.values())
        total_skipped = sum(result.skipped.values())
        total_failed = len(result.failed)

        stats.update({
            "status": "done",
            "total_embedded": total_embedded,
            "skipped": total_skipped,
            "failed": total_failed,
            "duration_s": round(time.time() - start, 2),
        })
        log.info("Reindexed: %d embedded, %d skipped, %d failed in %.2fs",
                 total_embedded, total_skipped, total_failed, stats["duration_s"])
        return stats

    except Exception as e:
        stats.update({"status": "error", "error": str(e), "duration_s": round(time.time() - start, 2)})
        log.exception("Reindex job error")
        return stats


async def handle_worker_job(ctx: dict[str, Any]) -> dict[str, Any]:
    """Main handler called by arq worker."""
    job_type = ctx.get("job_type", "")
    job_data = ctx.get("job_data", {})

    log.info("Handling job: %s data=%s", job_type, job_data)

    if job_type == "scrape":
        return await handle_scrape(ctx)
    elif job_type == "parse":
        return await handle_parse(ctx)
    elif job_type == "reindex":
        return await handle_reindex(ctx)
    else:
        return {"status": "error", "error": f"Unknown job type: {job_type}"}


# ---------------------------------------------------------------------------
# Arq Worker Settings
# ---------------------------------------------------------------------------


class WorkerSettings:
    functions = [handle_worker_job]
    cron_jobs = []  # Configurable for scheduled ingest
    retry_jobs = True
    keep_result = 3600 * 24 * 7  # Keep results for 7 days
    enable_time_limit = False   # Disable time limit for long operations
    max_tries = 4
