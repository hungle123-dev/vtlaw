"""Safety rails shared by destructive integration tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_finish(session: pytest.Session) -> None:
    """Require an explicit opt-in before tests can wipe Neo4j."""
    has_destructive_test = any("integration" in item.keywords for item in session.items)
    if has_destructive_test and os.environ.get("VTLAW_TEST_WIPE") != "1":
        raise pytest.UsageError(
            "integration tests wipe Neo4j; set VTLAW_TEST_WIPE=1 only for an "
            "isolated database"
        )
