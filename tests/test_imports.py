"""Import-time guards.

Two failures the rest of the suite cannot see, because pytest imports test
modules directly and never exercises `import vtlaw.retrieve` as a package:

  * A circular import. `retrieve/__init__` imports query_parser, which imports
    LLMClient from `vtlaw.generate`, whose `__init__` imports answer.py. If
    answer.py imports query_parser at module scope the cycle closes and
    `import vtlaw.retrieve` raises ImportError — while every test still passed.
  * A Cypher query naming relationship types the importer never creates.
"""

from __future__ import annotations

import importlib

import pytest

# Every package that participates in the retrieve/generate import cycle.
PACKAGES = [
    "vtlaw.retrieve",
    "vtlaw.generate",
    "vtlaw.retrieve.query_parser",
    "vtlaw.retrieve.search",
    "vtlaw.generate.answer",
    "vtlaw.api.app",
    "vtlaw.cli",
]


@pytest.mark.parametrize("module", PACKAGES)
def test_module_imports_standalone(module):
    """Each module must import on its own, in any order.

    Parametrised rather than looped so a failure names the module that broke.
    """
    importlib.import_module(module)


def test_retrieve_package_imports_first():
    """`import vtlaw.retrieve` before anything else — the order that failed.

    Importing vtlaw.generate first happens to prime the cycle and hide the bug,
    which is why the whole test suite stayed green while the package was broken.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import vtlaw.retrieve"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_hierarchy_query_only_uses_relationships_the_importer_creates():
    """The graph-walk query must not name relationship types nothing writes.

    The query was carried over from a schema with Part/Chapter/Section levels.
    This project's importer creates only HAS_ARTICLE/HAS_CLAUSE/HAS_POINT, so
    Neo4j answered every graph-walk with three "relationship type does not
    exist" warnings — harmless to results, but it means the query and the
    schema disagree about what the graph contains.
    """
    import re

    from vtlaw.graph.schema import CHILD_RELATIONSHIP
    from vtlaw.retrieve.context_builder import _HIERARCHY_QUERY

    named = set(re.findall(r"HAS_[A-Z]+", _HIERARCHY_QUERY))
    created = set(CHILD_RELATIONSHIP.values())

    assert named <= created, (
        f"query walks {sorted(named - created)}, which the importer never creates"
    )
