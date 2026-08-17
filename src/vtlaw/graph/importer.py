"""Write parsed documents into Neo4j.

Design choices worth stating, because each prevents a specific failure:

**MERGE on `uid`, one transaction per document.** Re-importing the same corpus
converges instead of duplicating, and a malformed document rolls back alone
rather than poisoning the batch.

**Parents before children.** A clause relationship MATCHes its article, so
articles are written first. Getting the order wrong loses the edge silently, not
loudly.

**`parent_uid` comes from the parser.** It already accounts for the `#n`
disambiguation suffix used when a document numbers two provisions identically.
Rebuilding the parent UID here from `(doc, article, clause)` would point the
second `khoản 4` at the first one's parent — the exact bug the suffix exists to
avoid.

**Orphans are counted, not assumed away.** If a provision's parent is missing
from the source, its edge cannot be created. That is reported, because an orphan
Point in the graph reads as a missing penalty later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from neo4j import Session

from vtlaw.graph.client import GraphClient
from vtlaw.graph.schema import PROVISION_LABELS
from vtlaw.parse import ParsedDocument, Provision

log = logging.getLogger(__name__)

BATCH_SIZE = 500

_MERGE_DOCUMENT = """
MERGE (d:Document {doc_identity: $doc_identity})
SET d.doc_guid      = $doc_guid,
    d.doc_name      = $doc_name,
    d.doc_type      = $doc_type,
    d.effect_status = $effect_status,
    d.issue_date    = CASE WHEN $issue_date  IS NULL THEN NULL ELSE date($issue_date)  END,
    d.effect_date   = CASE WHEN $effect_date IS NULL THEN NULL ELSE date($effect_date) END,
    d.expire_date   = CASE WHEN $expire_date IS NULL THEN NULL ELSE date($expire_date) END,
    d.fields        = $fields,
    d.organizations = $organizations,
    d.signers       = $signers
"""

# Articles hang off the Document; clauses and points hang off the exact parent UID
# the parser recorded.
_MERGE_ARTICLES = """
UNWIND $rows AS row
MERGE (a:Article {uid: row.uid})
SET a.doc_identity = row.doc_identity,
    a.number       = row.number,
    a.title        = row.title,
    a.content      = row.content,
    a.ordinal      = row.ordinal,
    a.occurrence   = row.occurrence
WITH a, row
MATCH (d:Document {doc_identity: row.doc_identity})
MERGE (d)-[:HAS_ARTICLE]->(a)
"""

_MERGE_CLAUSES = """
UNWIND $rows AS row
MERGE (c:Clause {uid: row.uid})
SET c.doc_identity   = row.doc_identity,
    c.number         = row.number,
    c.content        = row.content,
    c.parent_article = row.parent_article,
    c.ordinal        = row.ordinal,
    c.occurrence     = row.occurrence
WITH c, row
MATCH (a:Article {uid: row.parent_uid})
MERGE (a)-[:HAS_CLAUSE]->(c)
"""

_MERGE_POINTS = """
UNWIND $rows AS row
MERGE (p:Point {uid: row.uid})
SET p.doc_identity   = row.doc_identity,
    p.letter         = row.number,
    p.content        = row.content,
    p.parent_article = row.parent_article,
    p.parent_clause  = row.parent_clause,
    p.ordinal        = row.ordinal,
    p.occurrence     = row.occurrence
WITH p, row
MATCH (c:Clause {uid: row.parent_uid})
MERGE (c)-[:HAS_POINT]->(p)
"""

_MERGE_BY_LEVEL = {
    "article": _MERGE_ARTICLES,
    "clause": _MERGE_CLAUSES,
    "point": _MERGE_POINTS,
}


@dataclass
class ImportStats:
    documents: int = 0
    articles: int = 0
    clauses: int = 0
    points: int = 0
    missing_parent: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def provisions(self) -> int:
        return self.articles + self.clauses + self.points

    def merge(self, other: ImportStats) -> None:
        self.documents += other.documents
        self.articles += other.articles
        self.clauses += other.clauses
        self.points += other.points
        self.missing_parent.extend(other.missing_parent)
        self.failed.extend(other.failed)

    def summary(self) -> str:
        text = (
            f"{self.documents} documents -> {self.provisions:,} provisions "
            f"(articles={self.articles:,} clauses={self.clauses:,} "
            f"points={self.points:,})"
        )
        if self.missing_parent:
            text += f"; missing_parent={len(self.missing_parent)}"
        if self.failed:
            text += f"; failed={len(self.failed)}"
        return text


def _row(provision: Provision) -> dict:
    return {
        "uid": provision.uid,
        "doc_identity": provision.doc_identity,
        "number": provision.number,
        "title": provision.title,
        "content": provision.content,
        "parent_uid": provision.parent_uid,
        "parent_article": provision.parent_article,
        "parent_clause": provision.parent_clause,
        "ordinal": provision.ordinal,
        "occurrence": provision.occurrence,
    }


def _iso(value: object) -> str | None:
    return value.isoformat() if value is not None else None  # type: ignore[union-attr]


def _write_document(session: Session, parsed: ParsedDocument) -> ImportStats:
    """Write one document and its provisions inside a single transaction."""
    document = parsed.document
    stats = ImportStats(documents=1)

    session.run(
        _MERGE_DOCUMENT,
        doc_identity=document.doc_identity,
        doc_guid=document.doc_guid,
        doc_name=document.doc_name,
        doc_type=document.doc_type,
        effect_status=document.effect_status,
        issue_date=_iso(document.issue_date),
        effect_date=_iso(document.effect_date),
        expire_date=_iso(document.expire_date),
        fields=list(document.fields),
        organizations=list(document.organizations),
        signers=list(document.signers),
    )

    # Group by level so parents are always written before their children.
    by_level: dict[str, list[dict]] = {"article": [], "clause": [], "point": []}
    for provision in parsed.provisions:
        if provision.level != "article" and provision.parent_uid is None:
            # No parent to attach to. Record it rather than writing a node whose
            # missing edge would look like a missing penalty at query time.
            stats.missing_parent.append(provision.uid)
            continue
        by_level[provision.level].append(_row(provision))

    for level in ("article", "clause", "point"):
        rows = by_level[level]
        statement = _MERGE_BY_LEVEL[level]
        for start in range(0, len(rows), BATCH_SIZE):
            session.run(statement, rows=rows[start : start + BATCH_SIZE])
        setattr(stats, f"{level}s", len(rows))

    return stats


def import_documents(
    client: GraphClient, documents: list[ParsedDocument]
) -> ImportStats:
    """Import a whole corpus. One transaction per document."""
    client.ensure_schema()
    total = ImportStats()

    with client.session() as session:
        for parsed in documents:
            identity = parsed.document.doc_identity
            try:
                stats = session.execute_write(_write_document, parsed)
            except Exception as exc:  # noqa: BLE001 — one bad document must not stop the run
                total.failed.append(f"{identity}: {exc}")
                log.warning("failed to import %s: %s", identity, exc)
                continue

            total.merge(stats)
            log.info("imported %s: %s", identity, stats.summary())

    if total.missing_parent:
        log.warning(
            "%d provisions had no resolvable parent: %s",
            len(total.missing_parent), total.missing_parent[:5],
        )
    return total


# ---------------------------------------------------------------------------
# Verification — read back what was written
# ---------------------------------------------------------------------------


@dataclass
class GraphCounts:
    documents: int = 0
    articles: int = 0
    clauses: int = 0
    points: int = 0
    relationships: int = 0
    orphan_clauses: int = 0
    orphan_points: int = 0

    @property
    def provisions(self) -> int:
        return self.articles + self.clauses + self.points

    def summary(self) -> str:
        text = (
            f"{self.documents} documents, {self.provisions:,} provisions "
            f"(articles={self.articles:,} clauses={self.clauses:,} "
            f"points={self.points:,}), {self.relationships:,} relationships"
        )
        if self.orphan_clauses or self.orphan_points:
            text += (
                f"; orphans: clauses={self.orphan_clauses} "
                f"points={self.orphan_points}"
            )
        return text


def count_graph(client: GraphClient) -> GraphCounts:
    """Count what is actually in the database.

    Read back rather than trusting the import counters: the point of the check is
    to catch a write that reported success without landing.
    """
    with client.session() as session:
        counts = GraphCounts(
            documents=session.run(
                "MATCH (d:Document) RETURN count(d) AS n"
            ).single()["n"],
            relationships=session.run(
                "MATCH ()-[r]->() RETURN count(r) AS n"
            ).single()["n"],
        )
        for label in PROVISION_LABELS:
            value = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS n"
            ).single()["n"]
            setattr(counts, f"{label.lower()}s", value)

        counts.orphan_clauses = session.run(
            "MATCH (c:Clause) WHERE NOT (:Article)-[:HAS_CLAUSE]->(c) "
            "RETURN count(c) AS n"
        ).single()["n"]
        counts.orphan_points = session.run(
            "MATCH (p:Point) WHERE NOT (:Clause)-[:HAS_POINT]->(p) "
            "RETURN count(p) AS n"
        ).single()["n"]

    return counts
