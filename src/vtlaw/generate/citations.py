"""Check that legal citations in an answer are supported by retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vtlaw.parse.models import article_uid, clause_uid, point_uid
from vtlaw.parse.patterns import DOC_IDENTITY
from vtlaw.retrieve.search import Hit

CitationStatus = Literal["verified", "missing", "unsupported"]

_CITATION = re.compile(
    r"(?:(?:điểm\s+(?P<point>[a-zđ])\s+)?"
    r"(?:khoản\s+(?P<clause>\d+[a-z]?)\s+)?)?"
    r"điều\s+(?P<article>\d+[a-z]?)\s+"
    r"(?:(?:nghị định|luật)\s+)?"
    # The trailing boundary matters: without it "168/2024/NĐ-CPx" matches as a
    # citation to 168/2024/NĐ-CP, so a typo'd or hallucinated document identity
    # reads as verified against a real source.
    rf"(?P<document>{DOC_IDENTITY})(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationCheck:
    status: CitationStatus
    cited: list[str]
    unsupported: list[str]


def assess_citations(answer: str, sources: list[Hit]) -> CitationCheck:
    """Return whether every explicit provision citation is covered by a source.

    A child hit supports citing its parent Article/Clause, but a broad Article
    hit cannot support a more specific Point the retrieval did not return.
    """
    cited = list(dict.fromkeys(_citation_uid(match) for match in _CITATION.finditer(answer)))
    if not cited:
        return CitationCheck("missing", [], [])
    source_uids = [source.uid for source in sources]
    unsupported = [
        uid
        for uid in cited
        if not any(source == uid or source.startswith(uid + "::") for source in source_uids)
    ]
    return CitationCheck("unsupported" if unsupported else "verified", cited, unsupported)


def _citation_uid(match: re.Match[str]) -> str:
    document = match.group("document").upper()
    article = match.group("article").lower()
    clause = match.group("clause")
    point = match.group("point")
    if clause is None:
        return article_uid(document, article)
    if point is None:
        return clause_uid(document, article, clause.lower())
    return point_uid(document, article, clause.lower(), point.lower())
