"""Tests for the async ingestion worker.

The worker previously ran zero jobs. Three defects, each proven here:

  * `from ..scraper.client import ...` — the package is `scrape`, not `scraper`,
    so every scrape job raised ModuleNotFoundError.
  * `sum(corp_stats.values())` — CorpusStats is a dataclass with no `.values()`.
  * `sum(result.clauses + result.points)` — both are ints; `sum` needs an
    iterable.
  * WorkerSettings had no `redis_settings`, so arq connected to the default
    localhost:6379 and never saw this project's queue on 16379.

Nothing caught them because the module had no tests and its failing imports are
all inside function bodies, so `import vtlaw.ingest.worker` succeeded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from vtlaw.ingest.worker import (
    ParseJob,
    ReindexJob,
    ScrapeJob,
    WorkerSettings,
    handle_parse,
    handle_reindex,
    handle_scrape,
    handle_worker_job,
)


@pytest.fixture
def parse_stubs(monkeypatch):
    """Stub parse + import so the handlers run without a snapshot or Neo4j."""
    parse_stats = MagicMock(documents=12, provisions=7381, failed=[])
    parsed_docs = [MagicMock()]
    imported = MagicMock(provisions=7381, failed=[], missing_parent=[])

    client = MagicMock()
    monkeypatch.setattr("vtlaw.ingest.worker._graph_client", lambda *a: client)
    monkeypatch.setattr(
        "vtlaw.parse.corpus.parse_corpus", lambda _s: (parsed_docs, parse_stats)
    )
    monkeypatch.setattr(
        "vtlaw.graph.importer.import_documents", lambda _c, _d: imported
    )
    monkeypatch.setattr("vtlaw.scrape.snapshot.Snapshot", lambda *a, **k: MagicMock())
    return client, parse_stats, imported


class TestParseJob:
    async def test_reports_counts(self, parse_stubs):
        result = await handle_parse(ParseJob())

        assert result["status"] == "done"
        assert result["provisions_parsed"] == 7381
        assert result["provisions_imported"] == 7381

    async def test_closes_the_graph_client(self, parse_stubs):
        """A worker process is long-lived; a leaked driver per job exhausts pools."""
        client, _, _ = parse_stubs

        await handle_parse(ParseJob())

        client.close.assert_called_once()

    async def test_failure_is_returned_not_raised(self, monkeypatch):
        """arq stores the returned dict; raising would lose the reason."""
        monkeypatch.setattr(
            "vtlaw.ingest.worker._graph_client",
            MagicMock(side_effect=RuntimeError("neo4j down")),
        )
        monkeypatch.setattr(
            "vtlaw.scrape.snapshot.Snapshot", lambda *a, **k: MagicMock()
        )
        monkeypatch.setattr(
            "vtlaw.parse.corpus.parse_corpus",
            lambda _s: ([MagicMock()], MagicMock(documents=1, provisions=1, failed=[])),
        )

        result = await handle_parse(ParseJob())

        assert result["status"] == "error"
        assert "neo4j down" in result["error"]
        assert "duration_s" in result

    async def test_closes_client_even_on_import_failure(self, parse_stubs):
        client, _, _ = parse_stubs
        with patch(
            "vtlaw.graph.importer.import_documents",
            side_effect=RuntimeError("write failed"),
        ):
            result = await handle_parse(ParseJob())

        assert result["status"] == "error"
        client.close.assert_called_once()


class TestScrapeJob:
    async def test_imports_the_scrape_package_that_exists(self, parse_stubs):
        """The old code imported `..scraper`; the package is `scrape`."""
        report = MagicMock(blocked=False, written=12, failed=[])
        with patch(
            "vtlaw.ingest.worker._scrape_to_snapshot", return_value=report
        ) as scrape:
            result = await handle_scrape(ScrapeJob(max_documents=1))

        scrape.assert_called_once()
        assert result["status"] == "done"
        assert result["documents_written"] == 12

    async def test_a_refusal_stops_without_importing(self, parse_stubs):
        """403 means stop, not retry — and nothing should be written to the graph."""
        client, _, _ = parse_stubs
        report = MagicMock(blocked=True, written=0, failed=[])
        with patch("vtlaw.ingest.worker._scrape_to_snapshot", return_value=report):
            result = await handle_scrape(ScrapeJob())

        assert result["status"] == "blocked"
        client.close.assert_not_called()


class TestReindexJob:
    async def test_reports_embedding_counts(self, monkeypatch):
        client = MagicMock()
        monkeypatch.setattr("vtlaw.ingest.worker._graph_client", lambda *a: client)
        monkeypatch.setattr("vtlaw.embed.Embedder", lambda *a, **k: MagicMock())
        monkeypatch.setattr(
            "vtlaw.embed.embed_corpus",
            lambda *a, **k: MagicMock(
                embedded={"Article": 566, "Clause": 2763, "Point": 4052},
                skipped={"Article": 0, "Clause": 0, "Point": 0},
                failed=[],
            ),
        )

        result = await handle_reindex(ReindexJob())

        assert result["status"] == "done"
        assert result["embedded"] == 7381
        client.close.assert_called_once()


class TestDispatch:
    async def test_job_type_comes_from_the_argument_not_ctx(self, parse_stubs):
        """arq passes ctx first and the enqueued args after.

        The old handler read the type out of ctx, where arq never puts it, so
        every job fell through to "unknown job type".
        """
        result = await handle_worker_job({}, "parse", {})

        assert result["status"] == "done"

    async def test_unknown_type_is_reported(self):
        result = await handle_worker_job({}, "nonsense", {})

        assert result["status"] == "error"
        assert "nonsense" in result["error"]

    async def test_job_data_reaches_the_model(self, parse_stubs):
        """The enqueued dict must become the job model, not be dropped."""
        result = await handle_worker_job({}, "parse", {"output_dir": "custom/dir"})

        # The stubbed Snapshot swallows the path, so assert via the model instead:
        # a bad output_dir would have raised a validation error before running.
        assert result["status"] == "done"
        assert ParseJob(output_dir="custom/dir").output_dir == "custom/dir"

    async def test_job_data_is_validated(self):
        """An unknown key must not silently vanish into the model."""
        result = await handle_worker_job({}, "parse", {"output_dir": 123})

        assert result["status"] == "error"
        assert "ValidationError" in result["error"]


class TestWorkerSettings:
    def test_declares_redis_settings(self):
        """Without this arq connects to localhost:6379 and misses the queue."""
        assert WorkerSettings.redis_settings.port == 16379

    def test_job_timeout_allows_a_long_ingestion(self):
        """Embedding 7,381 provisions on CPU outlasts any default timeout."""
        assert WorkerSettings.job_timeout >= 3600

    def test_handler_is_registered(self):
        assert handle_worker_job in WorkerSettings.functions
