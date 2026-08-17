"""HTTP client for the phapluat.gov.vn legal-document API.

Endpoints
---------
    POST /api/legal-documents                        search / list
    GET  /api/legal-documents/detail?tabName=tomtat  metadata
    GET  /api/legal-documents/detail?tabName=noidung full content (HTML)

Four behaviours, each traced to something observed against the live service on
2026-08-15:

1. **TLS verification stays on.** Probed: ``verify=True`` returns 200 and the
   certificate validates, so disabling it buys nothing while letting a
   man-in-the-middle rewrite a penalty amount under a correct-looking citation.
2. **Every request has a timeout.** Without one a hung server hangs the whole
   ingest with no error and no log line.
3. **Bounded, jittered retry for transient failures only.** ``403`` is a
   decision, not a fault, and is never retried.
4. **The envelope is validated, not just the HTTP status.** Probed: ``HTTP 200``
   with ``{"data": {"rowCount": 0, "docs": []}, "error": "QTDC API error: 500"}``.
   ``raise_for_status()`` passes that, so a caller checking only the status reads
   an upstream failure as "end of catalog" and stops paginating with a
   complete-looking empty result.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://phapluat.gov.vn/api/legal-documents"

# The API answers requests shaped like the site's own frontend. These headers
# mirror that; they do not defeat access control. When the service refuses (403)
# this client stops rather than trying to get around it.
DEFAULT_HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "origin": "https://phapluat.gov.vn",
    "referer": "https://phapluat.gov.vn/he-thong-van-ban-phap-luat",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36"
    ),
}

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_BACKOFF_S = 8.0


class SourceError(RuntimeError):
    """Base class for source-API failures."""


class UpstreamError(SourceError):
    """HTTP 200 whose envelope carries a non-null ``error``.

    Deliberately distinct from an empty page: here an empty ``docs[]`` means the
    upstream failed, not that the catalog is exhausted.
    """

    def __init__(self, message: str, *, error: str, status: int) -> None:
        super().__init__(message)
        self.error = error
        self.status = status


class AccessBlockedError(SourceError):
    """The service refused the request (403 / WAF challenge).

    Raised so acquisition stops immediately. Retrying or circumventing a refusal
    is out of scope: back off and use the committed snapshot instead.
    """


@dataclass(frozen=True)
class SearchPage:
    """One page of search results, envelope already validated."""

    docs: tuple[dict[str, Any], ...]
    row_count: int
    page_index: int

    @property
    def is_empty(self) -> bool:
        """No documents.

        Safe to read as end-of-catalog *only* because the client raises
        :class:`UpstreamError` before a page is ever constructed when the
        envelope reported an error.
        """
        return not self.docs


@dataclass
class ClientStats:
    requests: int = 0
    retries: int = 0
    sleep_s: float = 0.0
    statuses: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.requests} requests, {self.retries} retries, "
            f"{self.sleep_s:.1f}s throttle+backoff, statuses={self.statuses}"
        )


class LegalDocumentClient:
    """Verifying, throttled, bounded-retry client for the source API."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = 30.0,
        connect_timeout_s: float = 10.0,
        throttle_s: float = 1.2,
        max_attempts: int = 4,
        verify_tls: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not verify_tls:
            # An explicit, loud escape hatch for a broken local CA store.
            # Never the default, never silent.
            log.warning(
                "TLS verification DISABLED — responses cannot be trusted; "
                "do not use this for a real scraping run"
            )
        self._base_url = base_url
        self._throttle_s = throttle_s
        self._max_attempts = max(1, max_attempts)
        self._last_request_at = 0.0
        self.verify_tls = verify_tls
        self.stats = ClientStats()
        self._client = httpx.Client(
            headers=DEFAULT_HEADERS,
            timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s),
            verify=verify_tls,
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LegalDocumentClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _throttle(self) -> None:
        """Keep at least ``throttle_s`` between requests.

        A courtesy limit on a public government service, not an optimisation:
        bursting has no upside and risks a block.
        """
        if self._last_request_at:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self._throttle_s:
                pause = self._throttle_s - elapsed
                time.sleep(pause)
                self.stats.sleep_s += pause
        self._last_request_at = time.monotonic()

    def _backoff(self, attempt: int, *, reason: str) -> None:
        # Exponential with jitter so concurrent retries do not resynchronise.
        delay = min(2.0 ** (attempt - 1), MAX_BACKOFF_S) * (0.5 + random.random())
        self.stats.retries += 1
        self.stats.sleep_s += delay
        log.warning(
            "retry %d/%d after %s; sleeping %.1fs",
            attempt, self._max_attempts, reason, delay,
        )
        time.sleep(delay)

    def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        last: Exception | None = None

        for attempt in range(1, self._max_attempts + 1):
            self._throttle()
            self.stats.requests += 1
            try:
                response = self._client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                if attempt == self._max_attempts:
                    break
                self._backoff(attempt, reason=type(exc).__name__)
                continue

            self.stats.statuses[response.status_code] = (
                self.stats.statuses.get(response.status_code, 0) + 1
            )

            if response.status_code == 403:
                raise AccessBlockedError(
                    f"403 from {url} — the service refused this request "
                    "(WAF / bot protection). Stopping; use the committed snapshot."
                )

            if response.status_code in RETRYABLE_STATUS:
                last = SourceError(f"HTTP {response.status_code}")
                if attempt == self._max_attempts:
                    break
                self._backoff(attempt, reason=f"HTTP {response.status_code}")
                continue

            response.raise_for_status()
            return self._validate_envelope(response)

        raise SourceError(
            f"{method} {url} failed after {self._max_attempts} attempts: {last}"
        ) from last

    @staticmethod
    def _validate_envelope(response: httpx.Response) -> dict[str, Any]:
        """Reject an envelope whose ``error`` is non-null, even on HTTP 200."""
        try:
            envelope = response.json()
        except ValueError as exc:
            raise SourceError(
                f"non-JSON body from {response.request.url} "
                f"(status={response.status_code}, "
                f"content-type={response.headers.get('content-type')})"
            ) from exc

        if not isinstance(envelope, dict):
            raise SourceError(f"unexpected envelope type {type(envelope).__name__}")

        error = envelope.get("error")
        if error:
            raise UpstreamError(
                f"upstream reported an error while returning HTTP "
                f"{response.status_code}: {error!r}",
                error=str(error),
                status=response.status_code,
            )
        return envelope

    # -- endpoints ---------------------------------------------------------

    def search(
        self,
        keywords: str = "",
        *,
        page_index: int = 0,
        row_amount: int = 100,
        date_from: str = "01/01/1945",
        date_to: str = "31/12/2026",
        doc_group_ids: list[int] | None = None,
        field_ids: list[int] | None = None,
        doc_type_ids: list[int] | None = None,
        effect_status_ids: list[int] | None = None,
        organ_ids: list[int] | None = None,
        signer_ids: list[int] | None = None,
        sort_by: str = "crDateTime",
        sort_order: str = "desc",
    ) -> SearchPage:
        """One page of search results.

        The total-count key is ``data.rowCount`` — not ``total`` or ``totalRow`` —
        confirmed against the live envelope.
        """
        body = {
            "keywords": keywords,
            "isSearchExact": 0,
            "dateFrom": date_from,
            "dateTo": date_to,
            "pageIndex": page_index,
            "rowAmount": row_amount,
            "searchOptions": 1,
            "searchByDate": "crDateTime",
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "docGroupIds": doc_group_ids or [],
            "fieldIds": field_ids or [],
            "effectStatusIds": effect_status_ids or [],
            "signerIds": signer_ids or [],
            "organIds": organ_ids or [],
            "docTypeIds": doc_type_ids or [],
            "languageId": 1,
        }
        envelope = self._request("POST", self._base_url, json=body)
        data = envelope.get("data") or {}
        return SearchPage(
            docs=tuple(data.get("docs") or ()),
            row_count=int(data.get("rowCount") or 0),
            page_index=page_index,
        )

    def get_metadata(self, doc_guid: str) -> dict[str, Any]:
        """``tomtat`` tab: identity, dates, status, fields, organs, related docs."""
        envelope = self._request(
            "GET",
            f"{self._base_url}/detail",
            params={"docGUId": doc_guid, "tabName": "tomtat"},
        )
        return envelope.get("data") or {}

    def get_content(self, doc_guid: str) -> dict[str, Any]:
        """``noidung`` tab: full document body as HTML under ``docContent``."""
        envelope = self._request(
            "GET",
            f"{self._base_url}/detail",
            params={"docGUId": doc_guid, "tabName": "noidung"},
        )
        return envelope.get("data") or {}
