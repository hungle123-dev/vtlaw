"""Parse every document in a snapshot."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from vtlaw.parse.models import Document, ParsedDocument
from vtlaw.parse.parser import parse_text
from vtlaw.scrape import Snapshot

log = logging.getLogger(__name__)


@dataclass
class CorpusStats:
    documents: int = 0
    articles: int = 0
    clauses: int = 0
    points: int = 0
    letter_suffixed: int = 0
    duplicate_numbering: int = 0
    failed: list[str] = field(default_factory=list)

    @property
    def provisions(self) -> int:
        return self.articles + self.clauses + self.points

    def summary(self) -> str:
        return (
            f"{self.documents} documents -> {self.provisions:,} provisions "
            f"(articles={self.articles:,} clauses={self.clauses:,} "
            f"points={self.points:,}); letter-suffixed={self.letter_suffixed} "
            f"duplicate-numbered={self.duplicate_numbering}"
        )


def parse_document(snapshot: Snapshot, doc_guid: str) -> ParsedDocument:
    """Parse one snapshot document by GUID."""
    metadata = snapshot.read_metadata(doc_guid)
    document = Document.from_metadata(doc_guid, metadata)
    return parse_text(snapshot.read_text(doc_guid), document)


def iter_corpus(snapshot: Snapshot) -> Iterator[ParsedDocument]:
    """Parse every document tracked in the snapshot manifest.

    Iterating the manifest rather than globbing the directory means an untracked
    stray file cannot silently enter the corpus — `scrape verify` is what decides
    what belongs.
    """
    for doc_guid in sorted(snapshot.load_manifest()):
        yield parse_document(snapshot, doc_guid)


def parse_corpus(snapshot: Snapshot) -> tuple[list[ParsedDocument], CorpusStats]:
    stats = CorpusStats()
    parsed: list[ParsedDocument] = []

    for doc_guid in sorted(snapshot.load_manifest()):
        try:
            document = parse_document(snapshot, doc_guid)
        except (OSError, ValueError) as exc:
            # One malformed document must not abort the corpus.
            stats.failed.append(f"{doc_guid}: {exc}")
            log.warning("failed to parse %s: %s", doc_guid, exc)
            continue

        parsed.append(document)
        counts = document.counts
        stats.documents += 1
        stats.articles += counts["article"]
        stats.clauses += counts["clause"]
        stats.points += counts["point"]
        stats.letter_suffixed += sum(
            1 for p in document.provisions
            if p.level in ("article", "clause") and not p.number.isdigit()
        )
        stats.duplicate_numbering += sum(
            1 for p in document.provisions if p.has_duplicate_numbering
        )

    return parsed, stats
