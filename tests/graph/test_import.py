"""Stage 3 integration tests — require a live Neo4j.

Run with a database up:

    docker compose up -d
    pytest tests/graph -m integration

These tests use their own isolated database-wide wipe, so they must not run
against an instance holding data you care about. They skip cleanly when no Neo4j
is reachable.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from vtlaw.graph import GraphClient, count_graph, import_amends_directory, import_documents
from vtlaw.parse import Document, ParsedDocument, parse_corpus, parse_text
from vtlaw.scrape import Snapshot

SNAPSHOT_ROOT = Path("data/snapshot")

# Measured on the committed snapshot.
EXPECTED_DOCUMENTS = 12
EXPECTED_ARTICLES = 566
EXPECTED_CLAUSES = 2_763
EXPECTED_POINTS = 4_052
EXPECTED_PROVISIONS = 7_381
EXPECTED_AMEND_EDGES = 440
EXPECTED_UNRESOLVED_AMENDS = 30

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    graph = GraphClient()
    try:
        graph.verify()
    except Exception as exc:  # noqa: BLE001 — no database available
        pytest.skip(f"Neo4j not reachable: {exc}")
    graph.ensure_schema()
    yield graph
    graph.close()


@pytest.fixture
def clean(client: GraphClient):
    """Empty database before each test; schema is kept."""
    client.wipe()
    yield client
    client.wipe()


def make_doc(identity: str, **kw) -> Document:
    return Document(
        doc_guid=f"guid-{identity}",
        doc_identity=identity,
        doc_name=f"Doc {identity}",
        effect_date=kw.pop("effect_date", date(2025, 1, 1)),
        **kw,
    )


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


def test_import_is_idempotent(clean: GraphClient):
    """Re-importing the same corpus must converge, not duplicate: MERGE on `uid`
    is what makes a re-run safe after a partial failure."""
    text = "Điều 6. Tiêu đề\n3. Phạt tiền:\na) Hành vi;\n"
    parsed = parse_text(text, make_doc("1/2025/NĐ-CP"))

    first = import_documents(clean, [parsed])
    counts_first = count_graph(clean)

    second = import_documents(clean, [parsed])
    counts_second = count_graph(clean)

    assert first.provisions == second.provisions == 3
    assert counts_first.provisions == counts_second.provisions == 3
    assert counts_first.relationships == counts_second.relationships


def test_duplicate_numbered_clauses_both_survive(clean: GraphClient):
    """The defect this stage exists to avoid.

    118/2025/QH15 Điều 5 numbers two distinct clauses `4`. A UID of
    (doc, article, clause) merges them, so MERGE keeps one and the other's legal
    text disappears. Measured against the reference parse of that document: 123
    clause rows collapse to 121 distinct UIDs without the `#n` suffix — two
    provisions lost. With the suffix, all 123 land.
    """
    text = (
        "Điều 5. Sửa đổi\n"
        "4. Sửa đổi khoản 4 Điều 25 như sau: nội dung A.\n"
        "5. Sửa đổi Điều 27 như sau: nội dung B.\n"
        "4. Sửa đổi điểm a khoản 1 Điều 29 như sau: nội dung C.\n"
        "5. Sửa đổi điểm a khoản 4 Điều 30 như sau: nội dung D.\n"
    )
    parsed = parse_text(text, make_doc("118/2025/QH15"))
    assert len(parsed.by_level("clause")) == 4

    import_documents(clean, [parsed])

    with clean.session() as session:
        rows = session.run(
            "MATCH (a:Article {uid: '118/2025/QH15::article::5'})"
            "-[:HAS_CLAUSE]->(c:Clause) "
            "RETURN c.uid AS uid, c.number AS number, c.content AS content "
            "ORDER BY c.uid"
        ).data()

    assert len(rows) == 4, "no clause may be lost to a UID collision"
    assert {r["uid"] for r in rows} == {
        "118/2025/QH15::article::5::clause::4",
        "118/2025/QH15::article::5::clause::4#2",
        "118/2025/QH15::article::5::clause::5",
        "118/2025/QH15::article::5::clause::5#2",
    }
    # The whole point: the two same-numbered clauses hold different text.
    assert len({r["content"] for r in rows}) == 4
    # The number stays clean for citation rendering.
    assert sorted(r["number"] for r in rows) == ["4", "4", "5", "5"]


def test_uid_constraint_rejects_a_genuine_duplicate(clean: GraphClient):
    """The database enforces uid uniqueness; the parser is not the only guard."""
    with clean.session() as session:
        session.run(
            "CREATE (:Article {uid: 'x::article::1', doc_identity: 'x'})"
        )
        with pytest.raises(Exception, match="already exists|ConstraintValidation"):
            session.run(
                "CREATE (:Article {uid: 'x::article::1', doc_identity: 'x'})"
            )


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


def test_hierarchy_edges_are_created_in_the_right_direction(clean: GraphClient):
    text = "Điều 6. Tiêu đề\n3. Phạt tiền:\na) Hành vi;\n"
    import_documents(clean, [parse_text(text, make_doc("1/2025/NĐ-CP"))])

    with clean.session() as session:
        row = session.run(
            "MATCH (d:Document)-[:HAS_ARTICLE]->(a:Article)"
            "-[:HAS_CLAUSE]->(c:Clause)-[:HAS_POINT]->(p:Point) "
            "RETURN d.doc_identity AS doc, a.number AS article, "
            "c.number AS clause, p.letter AS point"
        ).single()

    assert row is not None, "the full Document->Point path must exist"
    assert (row["doc"], row["article"], row["clause"], row["point"]) == (
        "1/2025/NĐ-CP", "6", "3", "a",
    )


def test_provision_without_a_parent_is_reported_not_written(clean: GraphClient):
    """A node whose edge cannot be created reads as a missing penalty later, so
    it is counted rather than written silently."""
    from vtlaw.parse import Provision

    orphan = ParsedDocument(
        document=make_doc("2/2025/NĐ-CP"),
        provisions=(
            # A clause whose parent_uid is None: the parser could not resolve it.
            Provision(
                uid="2/2025/NĐ-CP::article::9::clause::1",
                level="clause",
                doc_identity="2/2025/NĐ-CP",
                number="1",
                content="không có điều mẹ",
                parent_uid=None,
                parent_article="9",
            ),
        ),
    )

    stats = import_documents(clean, [orphan])

    assert stats.missing_parent == ["2/2025/NĐ-CP::article::9::clause::1"]
    assert stats.clauses == 0
    assert count_graph(clean).clauses == 0


def test_no_orphans_after_a_normal_import(clean: GraphClient):
    text = "Điều 6. T\n1. A:\na) B;\nb) C;\n2. D:\n"
    import_documents(clean, [parse_text(text, make_doc("3/2025/NĐ-CP"))])

    counts = count_graph(clean)

    assert counts.orphan_clauses == 0
    assert counts.orphan_points == 0
    assert counts.relationships == counts.provisions, (
        "each provision has exactly one parent edge"
    )


# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------


def test_document_dates_are_stored_as_neo4j_dates(clean: GraphClient):
    """Temporal filtering compares dates in Cypher, so they must be stored as
    `date`, not strings — a string comparison would be lexicographic."""
    document = make_doc(
        "4/2025/NĐ-CP",
        issue_date=date(2024, 12, 26),
        effect_date=date(2025, 1, 1),
        expire_date=date(2026, 12, 31),
    )
    import_documents(clean, [parse_text("Điều 1. X\n1. Y.\n", document)])

    with clean.session() as session:
        row = session.run(
            "MATCH (d:Document {doc_identity: '4/2025/NĐ-CP'}) "
            "RETURN d.effect_date AS effect, d.expire_date AS expire, "
            "d.effect_date < date('2025-06-01') AS in_force"
        ).single()

    assert str(row["effect"]) == "2025-01-01"
    assert str(row["expire"]) == "2026-12-31"
    assert row["in_force"] is True


def test_null_expire_date_stays_null(clean: GraphClient):
    """A null `expire_date` means open-ended. Coercing it to a date would make an
    in-force document look expired."""
    document = make_doc("5/2025/NĐ-CP", expire_date=None)
    import_documents(clean, [parse_text("Điều 1. X\n1. Y.\n", document)])

    with clean.session() as session:
        value = session.run(
            "MATCH (d:Document {doc_identity: '5/2025/NĐ-CP'}) "
            "RETURN d.expire_date AS expire"
        ).single()["expire"]

    assert value is None


def test_document_list_properties_round_trip(clean: GraphClient):
    document = Document(
        doc_guid="g",
        doc_identity="6/2025/NĐ-CP",
        doc_name="N",
        effect_date=date(2025, 1, 1),
        fields=("Giao thông", "Hành chính"),
        organizations=("Chính phủ",),
        signers=("Trần Hồng Hà",),
    )
    import_documents(clean, [parse_text("Điều 1. X\n1. Y.\n", document)])

    with clean.session() as session:
        row = session.run(
            "MATCH (d:Document {doc_identity: '6/2025/NĐ-CP'}) "
            "RETURN d.fields AS fields, d.signers AS signers"
        ).single()

    assert row["fields"] == ["Giao thông", "Hành chính"]
    assert row["signers"] == ["Trần Hồng Hà"]


def test_one_bad_document_does_not_abort_the_batch(
    clean: GraphClient, monkeypatch: pytest.MonkeyPatch
):
    """One transaction per document: a failure rolls back alone.

    The failure is injected rather than crafted from bad data, because the domain
    models validate on construction — there is no way to build an invalid
    Provision, which is itself the desired property.
    """
    from vtlaw.graph import importer

    first = parse_text("Điều 1. X\n1. Y.\n", make_doc("7/2025/NĐ-CP"))
    second = parse_text("Điều 1. X\n1. Y.\n", make_doc("8/2025/NĐ-CP"))
    third = parse_text("Điều 1. X\n1. Y.\n", make_doc("9/2025/NĐ-CP"))

    original = importer._write_document

    def flaky(session, parsed):
        if parsed.document.doc_identity == "8/2025/NĐ-CP":
            raise RuntimeError("simulated write failure")
        return original(session, parsed)

    monkeypatch.setattr(importer, "_write_document", flaky)

    stats = importer.import_documents(clean, [first, second, third])

    assert stats.documents == 2, "the surviving documents still import"
    assert len(stats.failed) == 1
    assert "8/2025/NĐ-CP" in stats.failed[0]

    counts = count_graph(clean)
    assert counts.documents == 2
    with clean.session() as session:
        identities = [
            r["id"] for r in session.run(
                "MATCH (d:Document) RETURN d.doc_identity AS id ORDER BY id"
            ).data()
        ]
    assert identities == ["7/2025/NĐ-CP", "9/2025/NĐ-CP"], (
        "the failed document left nothing behind"
    )


# ---------------------------------------------------------------------------
# Full corpus
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (SNAPSHOT_ROOT / "manifest.json").exists(), reason="snapshot not present"
)
def test_full_corpus_lands_exactly(clean: GraphClient):
    """End-to-end on the real dataset: parse count, import count, and the count
    read back from the database must all agree. Reading back is the point — it
    catches a write that reported success without landing."""
    documents, parse_stats = parse_corpus(Snapshot(SNAPSHOT_ROOT))
    assert parse_stats.provisions == EXPECTED_PROVISIONS

    import_stats = import_documents(clean, documents)
    counts = count_graph(clean)

    assert import_stats.failed == []
    assert import_stats.missing_parent == []

    assert counts.documents == EXPECTED_DOCUMENTS
    assert counts.articles == EXPECTED_ARTICLES
    assert counts.clauses == EXPECTED_CLAUSES
    assert counts.points == EXPECTED_POINTS
    assert counts.provisions == EXPECTED_PROVISIONS

    assert counts.orphan_clauses == 0
    assert counts.orphan_points == 0
    assert counts.relationships == EXPECTED_PROVISIONS


@pytest.mark.skipif(
    not (SNAPSHOT_ROOT / "manifest.json").exists(), reason="snapshot not present"
)
def test_known_citation_path_is_traversable(clean: GraphClient):
    """A Point states the offence; the amount lives in its parent Clause. This is
    why retrieval must walk up the hierarchy rather than return the Point alone.
    """
    documents, _ = parse_corpus(Snapshot(SNAPSHOT_ROOT))
    import_documents(clean, documents)

    with clean.session() as session:
        row = session.run(
            "MATCH (d:Document {doc_identity: '168/2024/NĐ-CP'})"
            "-[:HAS_ARTICLE]->(a:Article {number: '6'})"
            "-[:HAS_CLAUSE]->(c:Clause {number: '3'})"
            "-[:HAS_POINT]->(p:Point {letter: 'a'}) "
            "RETURN a.title AS article_title, c.content AS clause_text, "
            "p.content AS point_text, d.effect_date AS effect_date"
        ).single()

    assert row is not None
    assert "tốc độ" in row["point_text"]
    assert "Phạt tiền" in row["clause_text"], "the penalty lives in the parent clause"
    assert "ô tô" in row["article_title"]
    assert str(row["effect_date"]) == "2025-01-01"


@pytest.mark.skipif(
    not (SNAPSHOT_ROOT / "manifest.json").exists(), reason="snapshot not present"
)
def test_amendment_annotations_resolve_to_the_snapshot_graph(clean: GraphClient):
    """The graph must preserve every annotation whose endpoints exist.

    Some labels deliberately point to provisions inserted only inside quoted
    amendment text; those are not graph nodes and are reported as unresolved,
    never fabricated.
    """
    documents, _ = parse_corpus(Snapshot(SNAPSHOT_ROOT))
    import_documents(clean, documents)

    stats = import_amends_directory(clean, "data/amends")
    with clean.session() as session:
        edge_count = session.run(
            "MATCH ()-[r:AMENDS]->() RETURN count(r) AS n"
        ).single()["n"]

    assert stats.imported == EXPECTED_AMEND_EDGES
    assert stats.skipped == EXPECTED_UNRESOLVED_AMENDS
    assert edge_count == EXPECTED_AMEND_EDGES
