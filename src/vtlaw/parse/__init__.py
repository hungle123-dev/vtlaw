"""Stage 2 — parse snapshot text into provisions.

Reads :class:`vtlaw.scrape.Snapshot`, writes nothing: the output is in-memory
domain objects for the graph stage. No network.
"""

from vtlaw.parse.corpus import CorpusStats, iter_corpus, parse_corpus, parse_document
from vtlaw.parse.models import (
    Document,
    ParsedDocument,
    Provision,
    article_uid,
    clause_uid,
    point_uid,
    uid_to_citation,
)
from vtlaw.parse.parser import ParseWarnings, parse_text, split_footer

__all__ = [
    "CorpusStats",
    "Document",
    "ParseWarnings",
    "ParsedDocument",
    "Provision",
    "article_uid",
    "clause_uid",
    "iter_corpus",
    "parse_corpus",
    "parse_document",
    "parse_text",
    "point_uid",
    "split_footer",
    "uid_to_citation",
]
