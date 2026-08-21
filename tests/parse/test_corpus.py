"""Corpus-level parser tests: run against the real 12-document snapshot.

These are the tests that catch a regression a synthetic fixture cannot: they hold
the parser to the actual dataset, and they pin the *measured* facts rather than
assumptions about them.

"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from vtlaw.parse import parse_corpus, parse_document
from vtlaw.parse.patterns import closes_quote, opens_quote
from vtlaw.scrape import Snapshot

SNAPSHOT_ROOT = Path("data/snapshot")

# Measured on this snapshot. If a change moves any of these, the parser changed
# behaviour on real data and the diff needs justifying.
EXPECTED_DOCUMENTS = 12
EXPECTED_ARTICLES = 566
EXPECTED_CLAUSES = 2_763
EXPECTED_POINTS = 4_052
EXPECTED_PROVISIONS = 7_381
EXPECTED_DUPLICATE_NUMBERED = 2  # 118/2025/QH15 Điều 5: a second khoản 4 and 5

pytestmark = pytest.mark.skipif(
    not (SNAPSHOT_ROOT / "manifest.json").exists(),
    reason="snapshot not present",
)


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    return Snapshot(SNAPSHOT_ROOT)


@pytest.fixture(scope="module")
def corpus(snapshot: Snapshot):
    documents, stats = parse_corpus(snapshot)
    return documents, stats


# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------


def test_every_snapshot_document_parses(corpus):
    documents, stats = corpus

    assert stats.failed == []
    assert stats.documents == EXPECTED_DOCUMENTS
    assert len(documents) == EXPECTED_DOCUMENTS


def test_provision_totals_match_measured_baseline(corpus):
    _, stats = corpus

    assert stats.articles == EXPECTED_ARTICLES
    assert stats.clauses == EXPECTED_CLAUSES
    assert stats.points == EXPECTED_POINTS
    assert stats.provisions == EXPECTED_PROVISIONS


def test_no_provision_has_empty_uid_or_missing_document(corpus):
    documents, _ = corpus

    for document in documents:
        for provision in document.provisions:
            assert provision.uid
            assert provision.doc_identity == document.document.doc_identity
            assert provision.uid.startswith(f"{provision.doc_identity}::")


def test_all_uids_are_unique_across_the_whole_corpus(corpus):
    """The single most important invariant: a UID is a citation key and the graph
    node key. One collision silently merges two different provisions."""
    documents, _ = corpus
    seen: dict[str, str] = {}

    for document in documents:
        for provision in document.provisions:
            previous = seen.get(provision.uid)
            assert previous is None, (
                f"UID collision {provision.uid}: "
                f"{previous} vs {document.document.doc_identity}"
            )
            seen[provision.uid] = document.document.doc_identity

    assert len(seen) == EXPECTED_PROVISIONS


# ---------------------------------------------------------------------------
# Measured facts about this dataset
# ---------------------------------------------------------------------------


def test_letter_suffixed_headings_only_ever_appear_inside_quotes():
    """Corrects an earlier assumption of mine.

    I claimed a `^(\\d+)\\.` clause pattern loses 17 provisions to `2a.`/`18a.`
    style numbering. Measuring with quote tracking shows all 18 such lines sit
    inside quoted amendment text — they are provisions being inserted into *other*
    laws, not headings of the document being parsed. Skipping them is correct, and
    this test locks that in so a future "fix" cannot fabricate provisions.
    """
    inside = outside = 0
    clause_pattern = re.compile(r"^(\d{1,2}[a-z])\.\s+")
    article_pattern = re.compile(r"^Điều\s+(\d+[a-z])\.?\s")

    for path in sorted((SNAPSHOT_ROOT / "documents").glob("*.txt")):
        text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
        in_quote = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not in_quote and opens_quote(line):
                in_quote = True
            is_suffixed = bool(clause_pattern.match(line) or article_pattern.match(line))
            if in_quote:
                inside += is_suffixed
                if closes_quote(line):
                    in_quote = False
                continue
            outside += is_suffixed

    assert inside == 18, "measured: 18 letter-suffixed lines inside quoted text"
    assert outside == 0, "none appear as a document's own heading"


def test_duplicate_numbering_appears_exactly_where_measured(corpus):
    documents, stats = corpus

    assert stats.duplicate_numbering == EXPECTED_DUPLICATE_NUMBERED

    duplicates = [
        provision
        for document in documents
        for provision in document.provisions
        if provision.has_duplicate_numbering
    ]
    assert {p.doc_identity for p in duplicates} == {"118/2025/QH15"}
    assert {p.number for p in duplicates} == {"4", "5"}
    assert all(p.parent_article == "5" for p in duplicates)

    # The whole point: the two same-numbered clauses hold different text.
    contents = {p.content for p in duplicates}
    assert len(contents) == len(duplicates)


def test_duplicate_clauses_still_render_a_clean_citation(corpus):
    documents, _ = corpus

    for document in documents:
        for provision in document.provisions:
            if provision.has_duplicate_numbering:
                assert "#" not in provision.citation


# ---------------------------------------------------------------------------
# Per-document structure
# ---------------------------------------------------------------------------


def test_clauses_and_points_are_linked_to_a_real_parent(corpus):
    """A dangling parent_uid becomes an orphan node in the graph, which reads as
    a missing penalty rather than a broken import."""
    documents, _ = corpus

    for document in documents:
        uids = {p.uid for p in document.provisions}
        for provision in document.provisions:
            if provision.level == "article":
                continue
            assert provision.parent_uid is not None, f"{provision.uid} has no parent"
            assert provision.parent_uid in uids, (
                f"{provision.uid} points at a parent outside its document"
            )


def test_point_parents_are_clauses_and_clause_parents_are_articles(corpus):
    documents, _ = corpus

    for document in documents:
        level_of = {p.uid: p.level for p in document.provisions}
        for provision in document.provisions:
            if provision.level == "clause":
                assert level_of[provision.parent_uid] == "article"
            elif provision.level == "point":
                assert level_of[provision.parent_uid] == "clause"


def test_non_article_provisions_always_carry_content(corpus):
    """An empty article is normal — its text lives in its clauses. An empty clause
    or point means the parser dropped the body."""
    documents, _ = corpus

    for document in documents:
        for provision in document.provisions:
            if provision.level in ("clause", "point"):
                assert provision.content.strip(), f"{provision.uid} has no content"


def test_articles_always_have_a_title_or_content(corpus):
    documents, _ = corpus

    for document in documents:
        for provision in document.by_level("article"):
            assert provision.title or provision.content, (
                f"{provision.uid} has neither title nor content"
            )


def test_embedding_text_is_never_empty(corpus):
    """A provision with no embeddable text would be unreachable by vector search."""
    documents, _ = corpus

    for document in documents:
        for provision in document.provisions:
            assert provision.embedding_text().strip(), f"{provision.uid} embeds nothing"


def test_every_document_has_identity_and_effect_date(corpus):
    """Temporal filtering needs effect_date on every document."""
    documents, _ = corpus

    for document in documents:
        assert document.document.doc_identity
        assert document.document.doc_guid
        assert document.document.effect_date is not None, (
            f"{document.document.doc_identity} has no effect_date"
        )


def test_footer_split_discards_nothing_normative(corpus):
    """Measured: the footer is 0.5% of the corpus and holds no `Điều` heading."""
    documents, _ = corpus
    article_pattern = re.compile(r"^Điều\s+\d+[a-z]?\.?\s", re.MULTILINE)

    for document in documents:
        assert not article_pattern.search(document.footer), (
            f"{document.document.doc_identity} footer contains an article heading"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_parsing_is_deterministic(snapshot: Snapshot):
    """Re-parsing must produce identical UIDs and content: the graph stage uses
    the UID as its merge key, so instability would create duplicate nodes."""
    doc_guid = sorted(snapshot.load_manifest())[0]

    first = parse_document(snapshot, doc_guid)
    second = parse_document(snapshot, doc_guid)

    assert [p.uid for p in first.provisions] == [p.uid for p in second.provisions]
    assert [p.content for p in first.provisions] == [p.content for p in second.provisions]


def test_known_provision_is_parsed_with_its_penalty(snapshot: Snapshot):
    """A concrete end-to-end check on real text.

    NĐ 168/2024 Điều 6 Khoản 3 điểm a is a speeding offence. The point states the
    behaviour; the amount lives in the parent clause — which is exactly why
    retrieval has to return the clause, not the point alone.
    """
    manifest = snapshot.load_manifest()
    guid = next(
        g for g, record in manifest.items()
        if record["doc_identity"] == "168/2024/NĐ-CP"
    )
    parsed = parse_document(snapshot, guid)
    by_uid = {p.uid: p for p in parsed.provisions}

    point = by_uid["168/2024/NĐ-CP::article::6::clause::3::point::a"]
    clause = by_uid[point.parent_uid]
    article = by_uid[clause.parent_uid]

    assert "tốc độ" in point.content
    assert "Phạt tiền" in clause.content
    assert "đồng" in clause.content
    assert article.title is not None and "ô tô" in article.title
    assert point.citation == "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP"
