"""Stage 3 tests — schema shape and merge semantics without a database.

Integration tests that need a live Neo4j live in `test_import.py`.
"""

from __future__ import annotations

import pytest

from vtlaw.config import EMBED_DIM
from vtlaw.graph.importer import _iso, _row
from vtlaw.graph.schema import (
    ALL_SCHEMA_STATEMENTS,
    CHILD_RELATIONSHIP,
    CONSTRAINTS,
    FULLTEXT_INDEXES,
    PROPERTY_INDEXES,
    PROVISION_LABELS,
    VECTOR_INDEXES,
)
from vtlaw.parse import Provision


def test_provision_labels_are_ordered_outermost_first():
    """The importer relies on this order: a clause MATCHes its article, so
    articles must be written first or the edge is silently lost."""
    assert PROVISION_LABELS == ("Article", "Clause", "Point")


def test_every_provision_label_has_a_child_relationship():
    assert set(CHILD_RELATIONSHIP) == set(PROVISION_LABELS)
    assert CHILD_RELATIONSHIP["Article"] == "HAS_ARTICLE"
    assert CHILD_RELATIONSHIP["Clause"] == "HAS_CLAUSE"
    assert CHILD_RELATIONSHIP["Point"] == "HAS_POINT"


def test_all_schema_statements_are_idempotent():
    """`ensure_schema` runs on every startup, so re-application must be a no-op."""
    for statement in ALL_SCHEMA_STATEMENTS:
        assert "IF NOT EXISTS" in statement, statement


def test_uid_uniqueness_is_enforced_for_every_provision_label():
    """`uid` is the merge key and the citation identity. Uniqueness is the
    database's job, not something to trust from the parser."""
    for label in PROVISION_LABELS:
        assert any(
            f"FOR (n:{label}) REQUIRE n.uid IS UNIQUE" in c for c in CONSTRAINTS
        ), f"no uid constraint for {label}"


def test_document_identity_is_unique():
    assert any("d.doc_identity IS UNIQUE" in c for c in CONSTRAINTS)


def test_vector_indexes_declare_the_configured_dimension():
    """A mismatch between the index dimension and the model's output makes every
    write fail at query time, not at write time."""
    assert len(VECTOR_INDEXES) == len(PROVISION_LABELS)
    for statement in VECTOR_INDEXES:
        assert f"`vector.dimensions`: {EMBED_DIM}" in statement
        assert "'cosine'" in statement


def test_article_fulltext_index_covers_title_and_content():
    """Most articles have no body of their own; the title carries the topical
    signal, so omitting it would make them unfindable by keyword."""
    article_index = next(i for i in FULLTEXT_INDEXES if "Article" in i)
    assert "n.title" in article_index
    assert "n.content" in article_index


def test_temporal_properties_are_indexed():
    """`effect_date` and `expire_date` are filtered on every query."""
    joined = " ".join(PROPERTY_INDEXES)
    assert "d.effect_date" in joined
    assert "d.expire_date" in joined


def test_doc_identity_is_indexed_on_every_provision_label():
    joined = " ".join(PROPERTY_INDEXES)
    for label in PROVISION_LABELS:
        assert f"FOR (n:{label}) ON (n.doc_identity)" in joined


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------


def test_row_preserves_the_parser_supplied_parent_uid():
    """Rebuilding the parent UID from (doc, article, clause) here would point the
    second `khoản 4` at the first one's parent. The parser already resolved it
    with the `#n` suffix; the importer must pass it through untouched."""
    provision = Provision(
        uid="118/2025/QH15::article::5::clause::4#2",
        level="clause",
        doc_identity="118/2025/QH15",
        number="4",
        content="nội dung khác",
        parent_uid="118/2025/QH15::article::5",
        parent_article="5",
        occurrence=2,
    )

    row = _row(provision)

    assert row["uid"].endswith("#2")
    assert row["parent_uid"] == "118/2025/QH15::article::5"
    assert row["number"] == "4", "the human-facing number stays clean"
    assert row["occurrence"] == 2


def test_row_carries_ordinal_for_document_order():
    provision = Provision(
        uid="x::article::1", level="article", doc_identity="x",
        number="1", content="", title="T", ordinal=7,
    )
    assert _row(provision)["ordinal"] == 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None)],
)
def test_iso_passes_none_through(value, expected):
    """A null `expire_date` means open-ended, not epoch."""
    assert _iso(value) == expected


def test_iso_formats_dates():
    from datetime import date

    assert _iso(date(2025, 1, 1)) == "2025-01-01"
