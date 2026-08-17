"""Text extraction and hashing for scraped documents.

Kept separate from the HTTP client so that both the crawler and the offline
snapshot tools can produce byte-identical text without importing httpx.
"""

from __future__ import annotations

import hashlib
import unicodedata

from bs4 import BeautifulSoup

# Block-level tags become newlines before text extraction. Without this,
# BeautifulSoup's get_text() welds "Điều 6" onto the end of the previous
# sentence and the parser can no longer see where a provision begins.
_BLOCK_TAGS = (
    "p", "div", "br", "tr", "li", "table",
    "h1", "h2", "h3", "h4", "h5", "h6",
)


def html_to_text(html: str) -> str:
    """Flatten document HTML to plain text, preserving line structure.

    Output is NFC-normalised so Vietnamese diacritics have one byte
    representation everywhere downstream — hashes, regex matching, embeddings.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(list(_BLOCK_TAGS)):
        tag.insert_before("\n")
    lines = [line.rstrip() for line in soup.get_text().splitlines()]
    body = unicodedata.normalize("NFC", "\n".join(lines)).strip()
    return f"{body}\n" if body else ""


def sha256_text(text: str) -> str:
    """Content hash used for change detection and snapshot verification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
