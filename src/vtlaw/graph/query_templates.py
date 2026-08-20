"""Small, parameterized graph-query catalog for supported user questions.

This is deliberately not an LLM-to-Cypher executor. The router may select the
cypher_query intent, but the user text only fills parameters in fixed,
read-only templates that have tests and bounded result sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Literal

from vtlaw.parse.models import uid_to_citation
from vtlaw.parse.patterns import RE_DOC_IDENTITY

if TYPE_CHECKING:
    from vtlaw.graph.client import GraphClient

Operation = Literal["article_count", "signers", "amendment_history"]


@dataclass(frozen=True)
class GraphAnswer:
    operation: Operation
    answer: str


def _document_identity(question: str) -> str | None:
    match = RE_DOC_IDENTITY.search(question)
    return match.group("doc").upper() if match else None


def detect_operation(question: str) -> Operation | None:
    """Return an operation only when the required document reference is explicit."""
    if _document_identity(question) is None:
        return None
    text = question.casefold()
    if "bao nhiêu điều" in text or "mấy điều" in text:
        return "article_count"
    if "ai ký" in text or "người ký" in text:
        return "signers"
    if any(word in text for word in ("sửa đổi", "bổ sung", "bãi bỏ", "thay thế")):
        return "amendment_history"
    return None


class StructuredGraphQueries:
    """Execute the small, evidence-backed subset of graph questions."""

    def __init__(self, client: GraphClient) -> None:
        self._client = client

    def answer(self, question: str, *, as_of: date | None = None) -> GraphAnswer | None:
        operation = detect_operation(question)
        identity = _document_identity(question)
        if operation is None or identity is None:
            return None
        cutoff = as_of or date.today()
        if operation == "article_count":
            return self._article_count(identity, cutoff)
        if operation == "signers":
            return self._signers(identity, cutoff)
        return self._amendment_history(identity, cutoff)

    def _article_count(self, identity: str, as_of: date) -> GraphAnswer:
        with self._client.session() as session:
            rows = session.run(
                # Same effective/expiry predicate as retrieval. Without it a
                # document answers "has 55 articles" on a date it does not apply,
                # which is the one thing the date selector is there to prevent.
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                WHERE d.effect_date <= date($as_of)
                  AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
                OPTIONAL MATCH (d)-[:HAS_ARTICLE]->(a:Article)
                RETURN d.doc_identity AS doc_identity, d.doc_name AS doc_name, count(a) AS count
                """,
                doc_identity=identity,
                as_of=as_of.isoformat(),
            ).data()
        if not rows:
            return GraphAnswer(
                "article_count",
                f"Không có văn bản {identity} đang áp dụng trong corpus "
                f"tính đến {as_of.isoformat()}.",
            )
        row = rows[0]
        return GraphAnswer(
            "article_count",
            f"{row['doc_identity']} có {row['count']} điều trong corpus "
            f"(tính đến {as_of.isoformat()}).",
        )

    def _signers(self, identity: str, as_of: date) -> GraphAnswer:
        with self._client.session() as session:
            rows = session.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                WHERE d.effect_date <= date($as_of)
                  AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
                RETURN d.doc_identity AS doc_identity, d.signers AS signers
                """,
                doc_identity=identity,
                as_of=as_of.isoformat(),
            ).data()
        if not rows:
            return GraphAnswer(
                "signers",
                f"Không có văn bản {identity} đang áp dụng trong corpus "
                f"tính đến {as_of.isoformat()}.",
            )
        signers = rows[0].get("signers") or []
        text = ", ".join(signers) if signers else "không có dữ liệu người ký"
        return GraphAnswer("signers", f"Người ký {identity}: {text}.")

    def _amendment_history(self, identity: str, as_of: date) -> GraphAnswer:
        with self._client.session() as session:
            rows = session.run(
                """
                MATCH (source)-[r:AMENDS]->(target)
                WHERE source.doc_identity = $doc_identity OR target.doc_identity = $doc_identity
                MATCH (source_document:Document {doc_identity: source.doc_identity})
                WHERE source_document.effect_date IS NULL
                   OR source_document.effect_date <= date($as_of)
                RETURN coalesce(source.uid, source.doc_identity) AS source_uid,
                       source.doc_identity AS source_doc,
                       coalesce(target.uid, target.doc_identity) AS target_uid,
                       target.doc_identity AS target_doc,
                       r.type AS amend_type
                ORDER BY source_document.effect_date, source_uid
                LIMIT 20
                """,
                doc_identity=identity,
                as_of=as_of.isoformat(),
            ).data()
        if not rows:
            return GraphAnswer(
                "amendment_history",
                f"Không có cạnh sửa đổi/bổ sung/bãi bỏ được ghi nhận cho {identity} "
                f"tính đến {as_of.isoformat()}.",
            )
        lines = []
        for row in rows:
            source = _render_reference(row["source_uid"])
            target = _render_reference(row["target_uid"])
            lines.append(f"- {source} {row['amend_type']} {target}.")
        return GraphAnswer(
            "amendment_history",
            f"Các cạnh AMENDS của {identity} tính đến {as_of.isoformat()}:\n" + "\n".join(lines),
        )


def _render_reference(uid: str) -> str:
    return uid_to_citation(uid) if "::" in uid else uid
