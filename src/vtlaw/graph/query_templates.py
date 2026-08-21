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
    from vtlaw.retrieve.search import Hit

Operation = Literal["article_count", "signers", "amendment_history"]


@dataclass(frozen=True)
class GraphAnswer:
    operation: Operation
    answer: str
    sources: tuple[Hit, ...] = ()


def _missing_document(operation: Operation, identity: str) -> GraphAnswer:
    return GraphAnswer(
        operation,
        f"Không thể thực hiện {operation}: không có văn bản {identity} trong corpus.",
    )


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
                OPTIONAL MATCH (d)-[:HAS_ARTICLE]->(a:Article)
                RETURN d.doc_identity AS doc_identity, d.doc_name AS doc_name,
                       d.effect_date <= date($as_of)
                         AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
                         AS applicable,
                       count(a) AS count
                """,
                doc_identity=identity,
                as_of=as_of.isoformat(),
            ).data()
        if not rows:
            return _missing_document("article_count", identity)
        row = rows[0]
        if not row.get("applicable", True):
            return GraphAnswer(
                "article_count",
                f"Văn bản {identity} có trong corpus nhưng không áp dụng "
                f"tính đến {as_of.isoformat()}.",
            )
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
                RETURN d.doc_identity AS doc_identity, d.signers AS signers,
                       d.effect_date <= date($as_of)
                         AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
                         AS applicable
                """,
                doc_identity=identity,
                as_of=as_of.isoformat(),
            ).data()
        if not rows:
            return _missing_document("signers", identity)
        if not rows[0].get("applicable", True):
            return GraphAnswer(
                "signers",
                f"Văn bản {identity} có trong corpus nhưng không áp dụng "
                f"tính đến {as_of.isoformat()}.",
            )
        signers = rows[0].get("signers") or []
        text = ", ".join(signers) if signers else "không có dữ liệu người ký"
        return GraphAnswer("signers", f"Người ký {identity}: {text}.")

    def _amendment_history(self, identity: str, as_of: date) -> GraphAnswer:
        # ``Hit`` belongs to the retrieval package, whose search module imports
        # the embedder.  Delay this concrete conversion until an amendment
        # template actually needs evidence to keep graph package imports acyclic.
        from vtlaw.retrieve.search import Hit

        with self._client.session() as session:
            rows = session.run(
                """
                MATCH (d:Document {doc_identity: $doc_identity})
                OPTIONAL MATCH (source)-[r:AMENDS]->(target)
                WHERE source.doc_identity = d.doc_identity OR target.doc_identity = d.doc_identity
                OPTIONAL MATCH (source_document:Document {doc_identity: source.doc_identity})
                WITH d, source, target, r, source_document,
                     d.effect_date <= date($as_of)
                       AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
                       AS applicable
                ORDER BY source_document.effect_date, source.uid
                WITH d, applicable,
                     collect(
                       CASE
                         WHEN r IS NOT NULL
                          AND (source_document.effect_date IS NULL
                               OR source_document.effect_date <= date($as_of))
                         THEN {
                           source_uid: coalesce(source.uid, source.doc_identity),
                           source_doc: source.doc_identity,
                           source_label: head(labels(source)),
                           source_content: source.content,
                           target_uid: coalesce(target.uid, target.doc_identity),
                           target_doc: target.doc_identity,
                           target_label: head(labels(target)),
                           target_content: target.content,
                           amend_type: r.type
                         }
                       END
                     ) AS amendments
                RETURN d.doc_identity AS requested_doc,
                       applicable, amendments[..20] AS amendments
                """,
                doc_identity=identity,
                as_of=as_of.isoformat(),
            ).data()
        if not rows:
            return _missing_document("amendment_history", identity)
        row = rows[0]
        if not row.get("applicable", True):
            return GraphAnswer(
                "amendment_history",
                f"Văn bản {identity} có trong corpus nhưng không áp dụng "
                f"tính đến {as_of.isoformat()}.",
            )
        amendments = row.get("amendments") or []
        if not amendments:
            return GraphAnswer(
                "amendment_history",
                f"Không có cạnh sửa đổi/bổ sung/bãi bỏ được ghi nhận cho {identity} "
                f"tính đến {as_of.isoformat()}.",
            )
        lines = []
        sources: dict[str, Hit] = {}
        for amendment in amendments:
            source = _render_reference(amendment["source_uid"])
            target = _render_reference(amendment["target_uid"])
            lines.append(f"- {source} {amendment['amend_type']} {target}.")
            for side in ("source", "target"):
                uid = amendment.get(f"{side}_uid")
                doc_identity = amendment.get(f"{side}_doc")
                label = amendment.get(f"{side}_label")
                content = amendment.get(f"{side}_content") or ""
                if (
                    uid
                    and doc_identity
                    and label in {"Article", "Clause", "Point"}
                    and content.strip()
                ):
                    sources.setdefault(
                        uid,
                        Hit(
                            uid=uid,
                            score=1.0,
                            doc_identity=doc_identity,
                            label=label,
                            content=content,
                        ),
                    )
        return GraphAnswer(
            "amendment_history",
            f"Các cạnh AMENDS của {identity} tính đến {as_of.isoformat()}:\n" + "\n".join(lines),
            tuple(sources.values()),
        )


def _render_reference(uid: str) -> str:
    return uid_to_citation(uid) if "::" in uid else uid
