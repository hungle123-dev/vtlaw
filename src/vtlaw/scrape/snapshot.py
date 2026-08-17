"""The snapshot: an immutable, hash-verified copy of the source corpus.

    data/snapshot/documents/{doc_guid}.txt    plain text, NFC, line structure kept
    data/snapshot/metadata/{doc_guid}.json    the `tomtat` payload
    data/snapshot/manifest.json               hashes + provenance per document

Only :mod:`vtlaw.scrape` writes here, and only ``crawl`` needs the network.
Every later stage reads the snapshot, which is what keeps the pipeline
reproducible while the upstream API is unavailable — as it was during
development (``HTTP 200`` + ``error``, later ``403``).

Files are keyed by ``docGUId`` (a UUID) rather than ``docIdentity``: identity
strings such as ``100/2019/NĐ-CP`` are not unique across source revisions and
would collide on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vtlaw.scrape.text import sha256_text

MANIFEST_VERSION = 1
DEFAULT_ROOT = Path("data/snapshot")


@dataclass(frozen=True)
class DocumentRecord:
    """One document's provenance. Serialised into the manifest."""

    doc_guid: str
    doc_identity: str
    doc_name: str
    upd_datetime: str | None
    content_sha256: str
    metadata_sha256: str
    text_chars: int
    scraped_at: str

    def to_json(self) -> dict[str, Any]:
        return {
            "doc_guid": self.doc_guid,
            "doc_identity": self.doc_identity,
            "doc_name": self.doc_name,
            "upd_datetime": self.upd_datetime,
            "content_sha256": self.content_sha256,
            "metadata_sha256": self.metadata_sha256,
            "text_chars": self.text_chars,
            "scraped_at": self.scraped_at,
        }


@dataclass
class VerifyReport:
    documents: int = 0
    ok: int = 0
    content_mismatch: list[str] = field(default_factory=list)
    metadata_mismatch: list[str] = field(default_factory=list)
    missing_text: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    untracked: list[str] = field(default_factory=list)
    invalid_metadata: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.content_mismatch
            or self.metadata_mismatch
            or self.missing_text
            or self.missing_metadata
            or self.untracked
            or self.invalid_metadata
        )

    def problems(self) -> list[tuple[str, list[str]]]:
        return [
            ("content hash mismatch", self.content_mismatch),
            ("metadata hash mismatch", self.metadata_mismatch),
            ("missing text file", self.missing_text),
            ("missing metadata file", self.missing_metadata),
            ("on disk but untracked", self.untracked),
            ("metadata is not valid JSON", self.invalid_metadata),
        ]


@dataclass
class IndexReport:
    """Result of (re)building the manifest from files already on disk."""

    indexed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"indexed={self.indexed} failed={self.failed}"


class Snapshot:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.documents_dir = self.root / "documents"
        self.metadata_dir = self.root / "metadata"
        self.manifest_path = self.root / "manifest.json"

    # -- paths -------------------------------------------------------------

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def text_path(self, doc_guid: str) -> Path:
        return self.documents_dir / f"{doc_guid}.txt"

    def metadata_path(self, doc_guid: str) -> Path:
        return self.metadata_dir / f"{doc_guid}.json"

    def guids_on_disk(self) -> list[str]:
        return sorted(p.stem for p in self.documents_dir.glob("*.txt"))

    # -- manifest ----------------------------------------------------------

    def load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {entry["doc_guid"]: entry for entry in payload.get("documents", [])}

    def save_manifest(self, records: dict[str, dict[str, Any]]) -> None:
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "document_count": len(records),
            "total_text_chars": sum(r["text_chars"] for r in records.values()),
            "documents": sorted(records.values(), key=lambda r: r["doc_identity"]),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    # -- reads -------------------------------------------------------------

    def read_text(self, doc_guid: str) -> str:
        return self.text_path(doc_guid).read_text(encoding="utf-8")

    def read_metadata(self, doc_guid: str) -> dict[str, Any]:
        return json.loads(self.metadata_path(doc_guid).read_text(encoding="utf-8"))

    def content_hash_on_disk(self, doc_guid: str) -> str | None:
        path = self.text_path(doc_guid)
        return sha256_text(path.read_text(encoding="utf-8")) if path.exists() else None

    # -- writes ------------------------------------------------------------

    def write_document(
        self,
        doc_guid: str,
        *,
        text: str,
        metadata: dict[str, Any],
        fallback_identity: str = "",
        fallback_name: str = "",
    ) -> DocumentRecord:
        """Persist one document and return its provenance record."""
        metadata_raw = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        self.ensure_dirs()
        self.text_path(doc_guid).write_text(text, encoding="utf-8")
        self.metadata_path(doc_guid).write_text(metadata_raw, encoding="utf-8")
        return DocumentRecord(
            doc_guid=doc_guid,
            doc_identity=metadata.get("docIdentity") or fallback_identity,
            doc_name=metadata.get("docName") or fallback_name,
            upd_datetime=metadata.get("updDateTime"),
            content_sha256=sha256_text(text),
            metadata_sha256=sha256_text(metadata_raw),
            text_chars=len(text),
            scraped_at=datetime.now(UTC).isoformat(),
        )

    # -- offline maintenance ----------------------------------------------

    def reindex(self) -> IndexReport:
        """Rebuild the manifest from whatever is on disk.

        Needed for a corpus scraped before this tool existed: it computes the
        same hashes a fresh crawl would, so provenance is recorded and later
        stages cannot tell the difference.
        """
        self.ensure_dirs()
        report = IndexReport()
        records: dict[str, dict[str, Any]] = {}

        for text_path in sorted(self.documents_dir.glob("*.txt")):
            guid = text_path.stem
            metadata_path = self.metadata_path(guid)
            if not metadata_path.exists():
                report.failed += 1
                report.errors.append(f"{guid}: text present, metadata missing")
                continue

            metadata_raw = metadata_path.read_text(encoding="utf-8")
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError as exc:
                report.failed += 1
                report.errors.append(f"{guid}: metadata is not valid JSON ({exc})")
                continue

            text = text_path.read_text(encoding="utf-8")
            records[guid] = DocumentRecord(
                doc_guid=guid,
                doc_identity=metadata.get("docIdentity") or "",
                doc_name=metadata.get("docName") or "",
                upd_datetime=metadata.get("updDateTime"),
                content_sha256=sha256_text(text),
                metadata_sha256=sha256_text(metadata_raw),
                text_chars=len(text),
                scraped_at=datetime.fromtimestamp(
                    text_path.stat().st_mtime, tz=UTC
                ).isoformat(),
            ).to_json()
            report.indexed += 1

        self.save_manifest(records)
        return report

    def verify(self) -> VerifyReport:
        """Hash-check every tracked document. No network, no side effects."""
        report = VerifyReport()
        manifest = self.load_manifest()

        for guid, record in manifest.items():
            report.documents += 1
            text_path = self.text_path(guid)
            metadata_path = self.metadata_path(guid)

            if not text_path.exists():
                report.missing_text.append(guid)
                continue
            if not metadata_path.exists():
                report.missing_metadata.append(guid)
                continue

            metadata_raw = metadata_path.read_text(encoding="utf-8")
            try:
                json.loads(metadata_raw)
            except json.JSONDecodeError:
                report.invalid_metadata.append(guid)
                continue

            content_ok = (
                sha256_text(text_path.read_text(encoding="utf-8"))
                == record["content_sha256"]
            )
            metadata_ok = sha256_text(metadata_raw) == record["metadata_sha256"]

            if not content_ok:
                report.content_mismatch.append(guid)
            if not metadata_ok:
                report.metadata_mismatch.append(guid)
            if content_ok and metadata_ok:
                report.ok += 1

        tracked = set(manifest)
        report.untracked = [g for g in self.guids_on_disk() if g not in tracked]
        return report
