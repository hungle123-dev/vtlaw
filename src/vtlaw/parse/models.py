"""Domain model: documents, provisions, and the UID that ties them to citations.

A provision UID is document-scoped and hierarchical:

    168/2024/NĐ-CP::article::6::clause::3::point::a

Every decree has an "Điều 1", a "Khoản 1" and a "điểm a", so a UID that is not
scoped to its document collides across the corpus. The UID is also the natural
key for idempotent graph writes and the thing a citation renders from.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

Level = Literal["article", "clause", "point"]

_LEVEL_VI = {"article": "Điều", "clause": "Khoản", "point": "Điểm"}


def article_uid(doc_identity: str, number: str) -> str:
    return f"{doc_identity}::article::{number}"


def clause_uid(doc_identity: str, article: str, number: str) -> str:
    return f"{doc_identity}::article::{article}::clause::{number}"


def point_uid(doc_identity: str, article: str, clause: str, letter: str) -> str:
    return f"{doc_identity}::article::{article}::clause::{clause}::point::{letter}"


def uid_to_citation(uid: str) -> str:
    """Render a UID the way a lawyer writes it: smallest unit first.

    ``168/2024/NĐ-CP::article::6::clause::3::point::a``
    -> ``Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP``

    A ``#n`` disambiguation suffix is stripped. It is an internal key for
    source documents that number two distinct provisions identically, and it is
    not part of a legal citation.
    """
    if "::" not in uid:
        return uid
    doc, *rest = uid.split("::")
    parts = [
        f"{_LEVEL_VI.get(rest[i], rest[i])} {rest[i + 1].split('#')[0]}"
        for i in range(0, len(rest) - 1, 2)
    ]
    return " ".join([*reversed(parts), doc])


def _to_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return datetime.fromisoformat(text).date() if text else None


class Document(BaseModel):
    """A legal document plus the metadata retrieval needs.

    ``effect_date`` and ``expire_date`` are here rather than tucked away because
    temporal filtering reads them on every query: the same offence carries
    different penalties depending on when it happened.
    """

    model_config = ConfigDict(frozen=True)

    doc_guid: str
    doc_identity: str
    doc_name: str
    doc_type: str | None = None
    effect_status: str | None = None
    issue_date: date | None = None
    effect_date: date | None = None
    expire_date: date | None = None
    fields: tuple[str, ...] = ()
    organizations: tuple[str, ...] = ()
    signers: tuple[str, ...] = ()

    @classmethod
    def from_metadata(cls, doc_guid: str, raw: dict) -> Document:
        """Build from the source API's ``tomtat`` payload."""
        def names(key: str, name_key: str) -> tuple[str, ...]:
            items = raw.get(key) or []
            return tuple(
                item[name_key] for item in items
                if isinstance(item, dict) and item.get(name_key)
            )

        def nested_name(key: str, name_key: str) -> str | None:
            obj = raw.get(key)
            return obj.get(name_key) if isinstance(obj, dict) else None

        return cls(
            doc_guid=doc_guid,
            doc_identity=raw.get("docIdentity") or "",
            doc_name=raw.get("docName") or "",
            doc_type=nested_name("docType", "docTypeName"),
            effect_status=nested_name("effectStatus", "effectStatusName"),
            issue_date=_to_date(raw.get("issueDate")),
            effect_date=_to_date(raw.get("effectDate")),
            expire_date=_to_date(raw.get("expireDate")),
            fields=names("fields", "fieldName"),
            organizations=names("organs", "organName"),
            signers=names("signers", "signerName"),
        )


class Provision(BaseModel):
    """One retrievable unit of law: an Article, Clause, or Point."""

    model_config = ConfigDict(frozen=True)

    uid: str
    level: Level
    doc_identity: str
    number: str                      # clause/article number, or point letter
    content: str
    title: str | None = None         # articles only
    parent_uid: str | None = None    # exact parent UID, suffix-aware
    parent_article: str | None = None
    parent_clause: str | None = None
    ordinal: int = 0                  # position in document order
    # >1 when the source numbers two distinct provisions identically. Real case:
    # 118/2025/QH15 Điều 5 contains two different "khoản 4", each amending a
    # different law. Without this the two collapse into one node and legal text
    # is lost.
    occurrence: int = 1

    @property
    def citation(self) -> str:
        return uid_to_citation(self.uid)

    @property
    def has_duplicate_numbering(self) -> bool:
        return self.occurrence > 1

    def embedding_text(self) -> str:
        """Text to embed.

        An article's title carries strong topical signal ("Xử phạt người điều
        khiển xe ô tô…") and many articles have no body of their own, so the
        title is prepended rather than dropped.
        """
        if self.title:
            return f"{self.title}\n{self.content}".strip()
        return self.content.strip()


class ParsedDocument(BaseModel):
    """Everything one source document yields after parsing."""

    model_config = ConfigDict(frozen=True)

    document: Document
    provisions: tuple[Provision, ...]
    preamble: str = ""
    footer: str = ""

    def by_level(self, level: Level) -> tuple[Provision, ...]:
        return tuple(p for p in self.provisions if p.level == level)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "article": len(self.by_level("article")),
            "clause": len(self.by_level("clause")),
            "point": len(self.by_level("point")),
        }
