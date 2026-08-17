"""Stage 1 — scraping.

Everything that touches the network lives here and nowhere else. The rest of the
pipeline reads :class:`Snapshot`, which is what keeps it reproducible while the
upstream service is unavailable.
"""

from vtlaw.scrape.client import (
    AccessBlockedError,
    ClientStats,
    LegalDocumentClient,
    SearchPage,
    SourceError,
    UpstreamError,
)
from vtlaw.scrape.scraper import Scraper, ScrapeReport
from vtlaw.scrape.snapshot import (
    DocumentRecord,
    IndexReport,
    Snapshot,
    VerifyReport,
)
from vtlaw.scrape.text import html_to_text, sha256_text

__all__ = [
    "AccessBlockedError",
    "ScrapeReport",
    "Scraper",
    "ClientStats",
    "DocumentRecord",
    "IndexReport",
    "LegalDocumentClient",
    "SearchPage",
    "Snapshot",
    "SourceError",
    "UpstreamError",
    "VerifyReport",
    "html_to_text",
    "sha256_text",
]
