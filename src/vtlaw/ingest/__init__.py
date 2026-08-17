"""Stage 9 — async ingestion worker.

Runs scrape / parse / reindex as background jobs on an arq queue, so a long
ingestion does not occupy a CLI process or an HTTP request.
"""

from vtlaw.ingest.worker import (
    ParseJob,
    ReindexJob,
    ScrapeJob,
    WorkerSettings,
    handle_worker_job,
)

__all__ = [
    "ParseJob",
    "ReindexJob",
    "ScrapeJob",
    "WorkerSettings",
    "handle_worker_job",
]
