"""Graph schema: labels, relationships, constraints, indexes.

Kept apart from the importer so the shape of the graph is readable in one place
and every DDL statement is idempotent (`IF NOT EXISTS`), letting `ensure_schema`
run on every startup.

    (:Document)-[:HAS_ARTICLE]->(:Article)-[:HAS_CLAUSE]->(:Clause)-[:HAS_POINT]->(:Point)

Why a graph rather than a table: a legal citation *is* a path through this tree,
and a Point on its own is not an answer. Take NĐ 168/2024 Điều 6 Khoản 3 điểm a —

    Point : "Điều khiển xe chạy quá tốc độ từ 05 km/h đến dưới 10 km/h"
    Clause: "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng ..."

the offence is in the Point, the amount is in its parent Clause. Retrieval that
returns the Point alone reads like an answer while omitting the penalty, so
walking up the hierarchy is a correctness requirement, not a convenience.
"""

from __future__ import annotations

from vtlaw.config import EMBED_DIM

# Retrievable levels. Ordered outermost-first: parents must be written before
# children so a relationship never targets a missing node.
PROVISION_LABELS: tuple[str, ...] = ("Article", "Clause", "Point")

DOCUMENT_LABEL = "Document"

CHILD_RELATIONSHIP = {
    "Article": "HAS_ARTICLE",   # Document -> Article
    "Clause": "HAS_CLAUSE",     # Article  -> Clause
    "Point": "HAS_POINT",       # Clause   -> Point
}

# `uid` is the merge key for every provision and the identity a citation renders
# from, so uniqueness is enforced by the database rather than trusted from the
# parser.
CONSTRAINTS: tuple[str, ...] = (
    f"CREATE CONSTRAINT document_identity_unique IF NOT EXISTS "
    f"FOR (d:{DOCUMENT_LABEL}) REQUIRE d.doc_identity IS UNIQUE",
    *(
        f"CREATE CONSTRAINT {label.lower()}_uid_unique IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.uid IS UNIQUE"
        for label in PROVISION_LABELS
    ),
)

# One vector index per label. Neo4j fixes the dimension per index, and searching
# a single label is cheaper than filtering a mixed one.
VECTOR_INDEXES: tuple[str, ...] = tuple(
    f"CREATE VECTOR INDEX {label.lower()}_embedding IF NOT EXISTS "
    f"FOR (n:{label}) ON n.embedding "
    "OPTIONS {indexConfig: {"
    f"`vector.dimensions`: {EMBED_DIM}, "
    "`vector.similarity_function`: 'cosine'}}"
    for label in PROVISION_LABELS
)

# Lucene full-text indexes for the BM25 half of hybrid retrieval. An Article's
# title carries most of its topical signal, so it is indexed alongside content.
FULLTEXT_INDEXES: tuple[str, ...] = (
    "CREATE FULLTEXT INDEX article_fulltext IF NOT EXISTS "
    "FOR (n:Article) ON EACH [n.title, n.content]",
    "CREATE FULLTEXT INDEX clause_fulltext IF NOT EXISTS "
    "FOR (n:Clause) ON EACH [n.content]",
    "CREATE FULLTEXT INDEX point_fulltext IF NOT EXISTS "
    "FOR (n:Point) ON EACH [n.content]",
)

# Range indexes for the properties that filter or join. `effect_date` and
# `expire_date` back temporal eligibility on every query; `doc_identity` joins a
# provision to its document.
PROPERTY_INDEXES: tuple[str, ...] = (
    f"CREATE INDEX document_effect_date IF NOT EXISTS "
    f"FOR (d:{DOCUMENT_LABEL}) ON (d.effect_date)",
    f"CREATE INDEX document_expire_date IF NOT EXISTS "
    f"FOR (d:{DOCUMENT_LABEL}) ON (d.expire_date)",
    *(
        f"CREATE INDEX {label.lower()}_doc_identity IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.doc_identity)"
        for label in PROVISION_LABELS
    ),
)

ALL_SCHEMA_STATEMENTS: tuple[str, ...] = (
    *CONSTRAINTS,
    *VECTOR_INDEXES,
    *FULLTEXT_INDEXES,
    *PROPERTY_INDEXES,
)
