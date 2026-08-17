"""Amends importer: creates AMENDS edges in the Neo4j graph.

Imports amendment relationships from ``data/amends/*.json`` files. Each amend
record creates an ``AMENDS`` edge from the amending provision to the target
provision, with a ``type`` property recording whether it is ``sửa đổi``,
``bổ sung``, ``bãi bỏ``, ``thay thế``, or ``sửa đổi, bổ sung``.

This is the foundation for the legal-safety heuristic: at retrieval time we
check whether a hit has been ``bãi bỏ`` (abolished) or ``thay thế`` (replaced)
by a newer document, and penalise it so we do not cite a provision that is no
longer in force.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neo4j import ManagedTransaction

from vtlaw.parse.models import article_uid, clause_uid, point_uid

if TYPE_CHECKING:
    from vtlaw.graph.client import GraphClient

log = logging.getLogger(__name__)


@dataclass
class AmendStats:
    imported: int = 0
    skipped: int = 0
    failed: list[str] = None

    def __post_init__(self) -> None:
        if self.failed is None:
            self.failed = []

    def summary(self) -> str:
        text = f"imported={self.imported}, skipped={self.skipped}"
        if self.failed:
            text += f", failed={len(self.failed)}"
        return text


def _resolve_uid(
    doc_identity: str,
    article: str | None,
    clause: str | None,
    point: str | None,
) -> tuple[str, str]:
    """Resolve an amend record's reference to a (label, uid) pair.

    Amending/target provisions are specified at varying granularity: some
    amends target an Article, some a Clause, some a Point. We pick the most
    specific one that has all its parent identifiers filled.
    """
    if point and article and clause:
        return "Point", point_uid(doc_identity, article, clause, point)
    if clause and article:
        return "Clause", clause_uid(doc_identity, article, clause)
    if article:
        return "Article", article_uid(doc_identity, article)
    return "Document", doc_identity


def _import_amends_tx(
    tx: ManagedTransaction,
    amends: list[dict[str, Any]],
) -> AmendStats:
    """Transaction function: import all amend records in one batch."""
    stats = AmendStats()

    for amend in amends:
        src_doc = amend.get("amending_doc_identity", "")
        src_art = amend.get("amending_article")
        src_cl = amend.get("amending_clause")
        src_pt = amend.get("amending_point")
        src_label, src_uid = _resolve_uid(src_doc, src_art, src_cl, src_pt)

        tgt_doc = amend.get("target_doc_identity", "")
        tgt_art = amend.get("target_article")
        tgt_cl = amend.get("target_clause")
        tgt_pt = amend.get("target_point")
        tgt_label, tgt_uid = _resolve_uid(tgt_doc, tgt_art, tgt_cl, tgt_pt)

        amend_type = amend.get("amend_type", "sửa đổi")

        result = tx.run(
            f"""
            MATCH (source:{src_label} {{uid: $src_uid}})
            MATCH (target:{tgt_label} {{uid: $tgt_uid}})
            MERGE (source)-[r:AMENDS]->(target)
            SET r.type = $amend_type
            RETURN count(r) AS n
            """,
            src_uid=src_uid,
            tgt_uid=tgt_uid,
            amend_type=amend_type,
        )

        record = result.single()
        if record and record["n"] > 0:
            stats.imported += 1
        else:
            stats.skipped += 1
            stats.failed.append(f"{src_uid} -> {tgt_uid}")
            log.debug(
                "amend skipped (source or target not in graph): %s -> %s",
                src_uid, tgt_uid,
            )

    return stats


def import_amends_file(
    client: GraphClient,
    path: Path,
) -> AmendStats:
    """Import amends from a single JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    amends = payload.get("amends", [])

    if not amends:
        return AmendStats()

    log.info("importing %d amends from %s", len(amends), path.name)

    with client.session() as session:
        return session.execute_write(_import_amends_tx, amends)


def import_amends_directory(
    client: GraphClient,
    directory: str | Path,
) -> AmendStats:
    """Import amends from all JSON files in a directory.

    Returns aggregated stats across all files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        log.warning("amends directory not found: %s", dir_path)
        return AmendStats()

    total = AmendStats()

    for path in sorted(dir_path.glob("*.json")):
        stats = import_amends_file(client, path)
        total.imported += stats.imported
        total.skipped += stats.skipped
        total.failed.extend(stats.failed)
        log.info("  %s: %s", path.name, stats.summary())

    log.info("amends total: %s", total.summary())
    return total
