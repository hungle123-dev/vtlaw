"""Turn snapshot text into provisions.

The parser walks the document once, tracking where it is in the hierarchy, and
emits one :class:`Provision` per Điều / Khoản / Điểm.

Three things it gets right that a naive line-matching pass does not:

1. **Quoted amendments are content, not structure.** An amending document quotes
   the text it replaces verbatim: 592 lines in this corpus open with a quote
   character. Parsing hierarchy inside those quotes invents provisions that do
   not exist in the document being parsed.
2. **Letter-suffixed numbering is real.** "Điều 18a" and "khoản 2a" are how
   amendments insert provisions without renumbering. See
   :mod:`vtlaw.parse.patterns`.
3. **Duplicate numbering must not merge.** When a document numbers two distinct
   provisions identically, both are kept under distinct UIDs.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from vtlaw.parse.models import (
    Document,
    ParsedDocument,
    Provision,
    article_uid,
    clause_uid,
    point_uid,
)
from vtlaw.parse.patterns import (
    RE_ARTICLE,
    RE_CHAPTER,
    RE_CLAUSE,
    RE_FOOTER,
    RE_PART,
    RE_POINT,
    RE_SECTION,
    closes_quote,
    opens_quote,
)

log = logging.getLogger(__name__)


@dataclass
class ParseWarnings:
    """Structural oddities worth surfacing instead of silently absorbing."""

    clause_before_article: int = 0
    point_before_clause: int = 0
    duplicate_numbering: list[str] = field(default_factory=list)
    empty_articles: int = 0

    @property
    def any(self) -> bool:
        return bool(
            self.clause_before_article
            or self.point_before_clause
            or self.duplicate_numbering
            or self.empty_articles
        )

    def summary(self) -> str:
        bits = []
        if self.clause_before_article:
            bits.append(f"clause_before_article={self.clause_before_article}")
        if self.point_before_clause:
            bits.append(f"point_before_clause={self.point_before_clause}")
        if self.duplicate_numbering:
            bits.append(f"duplicate_numbering={len(self.duplicate_numbering)}")
        if self.empty_articles:
            bits.append(f"empty_articles={self.empty_articles}")
        return " ".join(bits) or "none"


class _Builder:
    """Accumulates provisions while the parser walks the document.

    Content arriving after a header belongs to the deepest open provision, so the
    builder keeps mutable buffers and freezes them into immutable Provisions at
    the end.
    """

    def __init__(self, doc_identity: str) -> None:
        self._doc = doc_identity
        self._rows: list[dict] = []
        self._by_uid: dict[str, dict] = {}
        self._seen: Counter[str] = Counter()
        # First occurrence wins for parent lookup. Verified on this corpus: the
        # only duplicate-numbered provisions (118/2025/QH15 Điều 5) have no child
        # points, so no child can be attached to the wrong parent.
        self._article_uid: dict[str, str] = {}
        self._clause_uid: dict[tuple[str, str], str] = {}
        self.open_uid: str | None = None
        self.warnings = ParseWarnings()

    def _suffix(self, key: str, number: str) -> tuple[str, int]:
        self._seen[key] += 1
        n = self._seen[key]
        return (number if n == 1 else f"{number}#{n}"), n

    def add_article(self, number: str, title: str) -> None:
        suffixed, n = self._suffix(f"a:{number}", number)
        uid = article_uid(self._doc, suffixed)
        self._article_uid.setdefault(number, uid)
        if n > 1:
            self.warnings.duplicate_numbering.append(uid)
        row = {
            "uid": uid,
            "level": "article",
            "doc_identity": self._doc,
            "number": number,
            "title": title.strip() or None,
            "content": "",
            "parent_uid": None,
            "parent_article": None,
            "parent_clause": None,
            "ordinal": len(self._rows),
            "occurrence": n,
        }
        self._rows.append(row)
        self._by_uid[uid] = row
        self.open_uid = uid

    def add_clause(self, parent_article: str, number: str, content: str) -> None:
        suffixed, n = self._suffix(f"c:{parent_article}:{number}", number)
        uid = clause_uid(self._doc, parent_article, suffixed)
        self._clause_uid.setdefault((parent_article, number), uid)
        if n > 1:
            self.warnings.duplicate_numbering.append(uid)
        row = {
            "uid": uid,
            "level": "clause",
            "doc_identity": self._doc,
            "number": number,
            "title": None,
            "content": content.strip(),
            "parent_uid": self._article_uid.get(parent_article),
            "parent_article": parent_article,
            "parent_clause": None,
            "ordinal": len(self._rows),
            "occurrence": n,
        }
        self._rows.append(row)
        self._by_uid[uid] = row
        self.open_uid = uid

    def add_point(
        self, parent_article: str, parent_clause: str, letter: str, content: str
    ) -> None:
        key = f"p:{parent_article}:{parent_clause}:{letter}"
        suffixed, n = self._suffix(key, letter)
        uid = point_uid(self._doc, parent_article, parent_clause, suffixed)
        if n > 1:
            self.warnings.duplicate_numbering.append(uid)
        row = {
            "uid": uid,
            "level": "point",
            "doc_identity": self._doc,
            "number": letter,
            "title": None,
            "content": content.strip(),
            "parent_uid": self._clause_uid.get((parent_article, parent_clause)),
            "parent_article": parent_article,
            "parent_clause": parent_clause,
            "ordinal": len(self._rows),
            "occurrence": n,
        }
        self._rows.append(row)
        self._by_uid[uid] = row
        self.open_uid = uid

    def append_content(self, line: str) -> None:
        """Attach a continuation line to the deepest open provision."""
        if self.open_uid is None:
            return
        row = self._by_uid[self.open_uid]
        row["content"] = f"{row['content']}\n{line}".strip() if row["content"] else line

    def finish(self) -> tuple[Provision, ...]:
        self.warnings.empty_articles = sum(
            1 for r in self._rows if r["level"] == "article" and not r["content"]
        )
        return tuple(Provision(**row) for row in self._rows)


def split_footer(text: str) -> tuple[str, str]:
    """Split legal content from the signature block / distribution list / appendix.

    Measured on this corpus: the footer is 8,621 characters (0.5%) and contains no
    "Điều" header, so nothing normative is discarded.
    """
    match = RE_FOOTER.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.start() :]


def parse_text(text: str, document: Document) -> ParsedDocument:
    """Parse one document's plain text into provisions."""
    normalised = unicodedata.normalize("NFC", text)
    main, footer = split_footer(normalised)

    builder = _Builder(document.doc_identity)
    preamble: list[str] = []
    current_article: str | None = None
    current_clause: str | None = None
    in_quote = False

    for raw_line in main.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # --- quoted amendment text -------------------------------------
        # An amending document reproduces the provision it replaces verbatim.
        # Inside a quote, "Điều 5." is a reference to another law, not a heading
        # in this one, so hierarchy matching is suspended until the quote closes.
        if not in_quote and opens_quote(line):
            in_quote = True
        if in_quote:
            builder.append_content(line)
            if closes_quote(line):
                in_quote = False
            continue

        # --- Part / Chapter / Section ----------------------------------
        # These are organisational only: they carry no retrievable text of their
        # own, so they reset the article context and are otherwise skipped.
        if RE_PART.match(line) or RE_CHAPTER.match(line) or RE_SECTION.match(line):
            current_article = current_clause = None
            builder.open_uid = None
            continue

        # --- Điều -------------------------------------------------------
        if m := RE_ARTICLE.match(line):
            current_article = m.group(1)
            current_clause = None
            builder.add_article(current_article, m.group(2))
            continue

        # --- Khoản ------------------------------------------------------
        if m := RE_CLAUSE.match(line):
            if current_article is None:
                # A numbered line before any Điều is prose (e.g. a preamble
                # list), not a clause. Recording it as one would fabricate a
                # provision with no parent.
                builder.warnings.clause_before_article += 1
                if builder.open_uid is None:
                    preamble.append(line)
                else:
                    builder.append_content(line)
                continue
            current_clause = m.group(1)
            builder.add_clause(current_article, current_clause, m.group(2))
            continue

        # --- Điểm -------------------------------------------------------
        if m := RE_POINT.match(line):
            if current_article is None or current_clause is None:
                builder.warnings.point_before_clause += 1
                if builder.open_uid is None:
                    preamble.append(line)
                else:
                    builder.append_content(line)
                continue
            builder.add_point(current_article, current_clause, m.group(1), m.group(2))
            continue

        # --- continuation ----------------------------------------------
        if builder.open_uid is None:
            preamble.append(line)
        else:
            builder.append_content(line)

    provisions = builder.finish()
    if builder.warnings.any:
        log.info(
            "%s parse warnings: %s",
            document.doc_identity, builder.warnings.summary(),
        )

    return ParsedDocument(
        document=document,
        provisions=provisions,
        preamble="\n".join(preamble).strip(),
        footer=footer.strip(),
    )
