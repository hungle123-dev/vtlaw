"""Neo4j connection handling.

The driver is created lazily so that importing this module never opens a socket —
unit tests must not need a database.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import Neo4jError

from vtlaw.config import Settings, get_settings
from vtlaw.graph.schema import ALL_SCHEMA_STATEMENTS, PROVISION_LABELS

log = logging.getLogger(__name__)


class GraphClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._driver: Driver | None = None

    @property
    def database(self) -> str:
        return self._settings.neo4j_database

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
                # Suppress the "already exists" notice that every idempotent
                # schema statement produces on re-run.
                notifications_disabled_classifications=["SCHEMA"],
            )
        return self._driver

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.driver.session(database=self.database) as session:
            yield session

    def verify(self) -> None:
        """Fail fast with a clear error rather than at the first query."""
        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        """Apply constraints and indexes. Idempotent, safe on every startup."""
        with self.session() as session:
            for statement in ALL_SCHEMA_STATEMENTS:
                try:
                    session.run(statement)
                except Neo4jError as exc:
                    # `IF NOT EXISTS` covers re-runs, but an equivalent index
                    # created under a different name still conflicts. Report it
                    # instead of aborting startup.
                    log.warning("schema statement skipped: %s (%s)", statement, exc)

    def await_indexes(self, timeout_s: int = 300) -> None:
        """Block until indexes are online.

        Vector and full-text indexes populate asynchronously; querying before they
        are ONLINE returns silently incomplete results.
        """
        with self.session() as session:
            session.run("CALL db.awaitIndexes($timeout)", timeout=timeout_s * 1000)

    def index_states(self) -> list[dict]:
        with self.session() as session:
            return session.run(
                "SHOW INDEXES YIELD name, type, state, populationPercent "
                'WHERE type <> "LOOKUP" '
                "RETURN name, type, state, populationPercent ORDER BY name"
            ).data()

    def readiness(self) -> dict[str, object]:
        """Report whether all retrieval indexes and provision vectors are ready."""
        from vtlaw.embed import embedding_coverage

        indexes = self.index_states()
        coverage = embedding_coverage(self)
        required_indexes = {
            (f"{label.lower()}_{kind}", index_type)
            for label in PROVISION_LABELS
            for kind, index_type in (("embedding", "VECTOR"), ("fulltext", "FULLTEXT"))
        }
        online_indexes = {
            (row.get("name"), row.get("type"))
            for row in indexes
            if isinstance(row, dict) and row.get("state") == "ONLINE"
        }
        coverage_by_label = {
            row.get("label"): row
            for row in coverage
            if isinstance(row, dict)
        }
        embeddings_ready = set(coverage_by_label) == set(PROVISION_LABELS) and all(
            isinstance(coverage_by_label[label].get("total"), int)
            and coverage_by_label[label].get("embedded")
            == coverage_by_label[label].get("total")
            for label in PROVISION_LABELS
        )
        return {
            "ready": required_indexes <= online_indexes and embeddings_ready,
            "indexes": indexes,
            "embedding_coverage": coverage,
        }

    def wipe(self) -> None:
        """Delete all data, keeping the schema. For tests and re-imports."""
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> GraphClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
