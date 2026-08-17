"""Drive the client to fill the snapshot.

Change detection is by **content hash**, never by ``updDateTime``. The timestamp
is recorded because it is useful for ordering work, but a document is rewritten
only when its text actually differs — so re-running acquisition converges instead
of churning the corpus.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from vtlaw.scrape.client import AccessBlockedError, LegalDocumentClient, UpstreamError
from vtlaw.scrape.snapshot import DocumentRecord, Snapshot
from vtlaw.scrape.text import html_to_text, sha256_text

log = logging.getLogger(__name__)


@dataclass
class ScrapeReport:
    discovered: int = 0
    written: int = 0
    unchanged: int = 0
    failed: int = 0
    blocked: bool = False
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"discovered={self.discovered}",
            f"written={self.written}",
            f"unchanged={self.unchanged}",
            f"failed={self.failed}",
        ]
        if self.blocked:
            parts.append("BLOCKED")
        return " ".join(parts)


class Scraper:
    def __init__(
        self,
        client: LegalDocumentClient,
        snapshot: Snapshot,
        *,
        doc_group_ids: list[int] | None = None,
        field_ids: list[int] | None = None,
    ) -> None:
        self._client = client
        self._snapshot = snapshot
        self._doc_group_ids = doc_group_ids
        self._field_ids = field_ids

    def discover(
        self,
        keywords: str,
        *,
        max_documents: int | None = None,
        row_amount: int = 100,
    ) -> list[dict[str, Any]]:
        """Page through search results until the catalog is exhausted.

        An empty page means end-of-catalog *only* because the client raises
        :class:`UpstreamError` first when the envelope reported an error. That
        distinction is what stops an upstream failure from being recorded as a
        complete, empty corpus.
        """
        found: list[dict[str, Any]] = []
        page_index = 0

        while True:
            page = self._client.search(
                keywords,
                page_index=page_index,
                row_amount=row_amount,
                doc_group_ids=self._doc_group_ids,
                field_ids=self._field_ids,
            )
            if page.is_empty:
                log.info(
                    "end of catalog at page %d (rowCount=%d, collected=%d)",
                    page_index, page.row_count, len(found),
                )
                break

            found.extend(page.docs)
            log.info(
                "page %d: +%d (collected %d / rowCount %d)",
                page_index, len(page.docs), len(found), page.row_count,
            )

            if max_documents is not None and len(found) >= max_documents:
                return found[:max_documents]
            if page.row_count and len(found) >= page.row_count:
                break
            page_index += 1

        return found

    def scrape_document(self, doc: dict[str, Any]) -> tuple[DocumentRecord, bool]:
        """Scrape one document. Returns ``(record, wrote_to_disk)``.

        Content is fetched first: when its hash already matches what is stored
        there is nothing to write, and the metadata request is skipped entirely.
        """
        guid = doc["docGUId"]
        content = self._client.get_content(guid)
        text = html_to_text(content.get("docContent") or "")
        if not text.strip():
            raise UpstreamError(
                f"empty docContent for {guid}", error="empty_content", status=200
            )

        content_hash = sha256_text(text)
        if self._snapshot.content_hash_on_disk(guid) == content_hash:
            metadata_raw = self._snapshot.metadata_path(guid).read_text(
                encoding="utf-8"
            )
            metadata = json.loads(metadata_raw)
            record = DocumentRecord(
                doc_guid=guid,
                doc_identity=metadata.get("docIdentity") or doc.get("docIdentity", ""),
                doc_name=metadata.get("docName") or doc.get("docName", ""),
                upd_datetime=metadata.get("updDateTime"),
                content_sha256=content_hash,
                metadata_sha256=sha256_text(metadata_raw),
                text_chars=len(text),
                scraped_at=self._snapshot.load_manifest()
                .get(guid, {})
                .get("scraped_at", ""),
            )
            return record, False

        metadata = self._client.get_metadata(guid)
        record = self._snapshot.write_document(
            guid,
            text=text,
            metadata=metadata,
            fallback_identity=doc.get("docIdentity", ""),
            fallback_name=doc.get("docName", ""),
        )
        return record, True

    def run(self, keywords: str, *, max_documents: int | None = None) -> ScrapeReport:
        report = ScrapeReport()
        manifest = self._snapshot.load_manifest()

        try:
            docs = self.discover(keywords, max_documents=max_documents)
        except AccessBlockedError as exc:
            report.blocked = True
            report.errors.append(str(exc))
            log.error("%s", exc)
            return report
        except UpstreamError as exc:
            report.errors.append(str(exc))
            log.error("discovery failed: %s", exc)
            return report

        report.discovered = len(docs)

        for doc in docs:
            guid = doc.get("docGUId")
            if not guid:
                report.failed += 1
                report.errors.append("search result without docGUId")
                continue

            try:
                record, wrote = self.scrape_document(doc)
            except AccessBlockedError as exc:
                # Refused mid-run: stop and keep what is already on disk.
                report.blocked = True
                report.errors.append(f"{guid}: {exc}")
                log.error("access blocked at %s; stopping", guid)
                break
            except (UpstreamError, OSError, json.JSONDecodeError) as exc:
                report.failed += 1
                report.errors.append(f"{guid}: {exc}")
                log.warning("failed %s: %s", guid, exc)
                continue

            manifest[guid] = record.to_json()
            if wrote:
                report.written += 1
            else:
                report.unchanged += 1

        # Persist whatever was achieved, including after a block: a partial
        # snapshot with an accurate manifest beats none at all.
        self._snapshot.save_manifest(manifest)
        return report
