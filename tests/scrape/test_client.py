"""Stage 1 tests — no network.

The client is driven through an httpx MockTransport, so every behaviour that
matters (envelope validation, retry policy, refusal handling) is proven without
touching the live service.
"""

from __future__ import annotations

import json

import httpx
import pytest

from vtlaw.scrape import (
    AccessBlockedError,
    LegalDocumentClient,
    Snapshot,
    SourceError,
    UpstreamError,
    html_to_text,
    sha256_text,
)


def make_client(handler, **kw) -> LegalDocumentClient:
    """Client on a mock transport, throttling disabled for speed."""
    kw.setdefault("throttle_s", 0.0)
    return LegalDocumentClient(transport=httpx.MockTransport(handler), **kw)


def ok_envelope(docs=(), row_count=0) -> httpx.Response:
    return httpx.Response(
        200, json={"data": {"rowCount": row_count, "docs": list(docs)}, "error": None}
    )


# ---------------------------------------------------------------------------
# Envelope validation — the defect a status-only check misses
# ---------------------------------------------------------------------------


def test_http_200_with_envelope_error_raises():
    """Observed live 2026-08-15:

        HTTP 200, {"data": {"rowCount": 0, "docs": []},
                   "error": "QTDC API error: 500"}

    raise_for_status() passes this, so a status-only check reads an upstream
    failure as "end of catalog" and stops paginating with an empty corpus.
    """
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {"rowCount": 0, "docs": [], "searchOptions": 0},
                "error": "QTDC API error: 500",
            },
        )

    with make_client(handler) as client, pytest.raises(UpstreamError) as exc:
        client.search("giao thông")

    assert exc.value.status == 200
    assert "QTDC API error: 500" in exc.value.error


def test_empty_docs_with_null_error_is_a_normal_empty_page():
    """The inverse case: genuinely no results. Must not raise."""
    with make_client(lambda _r: ok_envelope()) as client:
        page = client.search("nothing matches this")

    assert page.is_empty
    assert page.row_count == 0


def test_row_count_is_the_total_key():
    """The total lives at data.rowCount, not total or totalRow."""
    docs = [{"docGUId": "g1", "docIdentity": "1/2020/NĐ-CP"}]
    with make_client(lambda _r: ok_envelope(docs=docs, row_count=137)) as client:
        page = client.search("x")

    assert page.row_count == 137
    assert page.docs[0]["docGUId"] == "g1"


def test_non_json_body_raises_source_error():
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>maintenance</html>", headers={"content-type": "text/html"}
        )

    with make_client(handler) as client, pytest.raises(SourceError, match="non-JSON"):
        client.search("x")


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def test_transient_500_is_retried_then_succeeds():
    calls = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"data": None, "error": "boom"})
        return ok_envelope(docs=[{"docGUId": "g"}], row_count=1)

    with make_client(handler, max_attempts=4) as client:
        page = client.search("x")

    assert calls["n"] == 3
    assert client.stats.retries == 2
    assert page.row_count == 1


def test_retries_are_bounded():
    calls = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"data": None, "error": "unavailable"})

    with make_client(handler, max_attempts=3) as client, pytest.raises(
        SourceError, match="after 3 attempts"
    ):
        client.search("x")

    assert calls["n"] == 3, "must not retry forever"


def test_timeout_is_treated_as_transient():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return ok_envelope(docs=[{"docGUId": "g"}], row_count=1)

    with make_client(handler, max_attempts=3) as client:
        page = client.search("x")

    assert calls["n"] == 2
    assert page.row_count == 1


def test_403_is_not_retried():
    """A refusal is a decision, not a fault. Retrying it is useless, and
    circumventing it is out of scope."""
    calls = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403, text="blocked")

    with make_client(handler, max_attempts=4) as client, pytest.raises(
        AccessBlockedError, match="refused"
    ):
        client.search("x")

    assert calls["n"] == 1


def test_404_is_not_retried():
    calls = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"data": None, "error": None})

    with make_client(handler, max_attempts=4) as client, pytest.raises(httpx.HTTPStatusError):
        client.get_metadata("missing")

    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_tls_verification_is_on_by_default():
    client = LegalDocumentClient(throttle_s=0.0)
    try:
        assert client.verify_tls is True
    finally:
        client.close()


def test_detail_endpoints_send_expected_tab_names():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(
            200, json={"data": {"docIdentity": "1/2020/NĐ-CP"}, "error": None}
        )

    with make_client(handler) as client:
        client.get_metadata("guid-1")
        client.get_content("guid-1")

    assert seen[0] == {"docGUId": "guid-1", "tabName": "tomtat"}
    assert seen[1] == {"docGUId": "guid-1", "tabName": "noidung"}


def test_search_body_carries_filters():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return ok_envelope()

    with make_client(handler) as client:
        client.search(
            "giao thông", doc_group_ids=[1], field_ids=[7, 9],
            page_index=2, row_amount=50,
        )

    assert captured["docGroupIds"] == [1]
    assert captured["fieldIds"] == [7, 9]
    assert captured["pageIndex"] == 2
    assert captured["rowAmount"] == 50
    assert captured["keywords"] == "giao thông"


# ---------------------------------------------------------------------------
# html_to_text
# ---------------------------------------------------------------------------


def test_block_tags_become_newlines():
    """Without inserted newlines, get_text() welds "Điều 6" onto the previous
    sentence and the parser can no longer find where a provision starts."""
    html = "<p>Điều 5. Nội dung trước</p><p>Điều 6. Nội dung sau</p>"

    lines = [ln for ln in html_to_text(html).splitlines() if ln.strip()]

    assert lines == ["Điều 5. Nội dung trước", "Điều 6. Nội dung sau"]


def test_text_is_nfc_normalised():
    decomposed = "Ðìều 6"  # combining grave accents
    text = html_to_text(f"<p>{decomposed}</p>")

    assert "̀" not in text, "combining marks must be normalised away"


def test_empty_html_yields_empty_string():
    assert html_to_text("") == ""
    assert html_to_text("<div></div>").strip() == ""


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_reindex_records_hashes(tmp_path):
    snap = Snapshot(tmp_path / "snapshot")
    snap.ensure_dirs()
    snap.text_path("guid-a").write_text("Điều 1. Nội dung\n", encoding="utf-8")
    snap.metadata_path("guid-a").write_text(
        json.dumps({"docIdentity": "1/2020/NĐ-CP", "docName": "Test",
                    "updDateTime": "2020-01-01T00:00:00"}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = snap.reindex()
    assert report.indexed == 1

    record = snap.load_manifest()["guid-a"]
    assert record["doc_identity"] == "1/2020/NĐ-CP"
    assert record["content_sha256"] == sha256_text("Điều 1. Nội dung\n")
    assert record["text_chars"] == len("Điều 1. Nội dung\n")


def test_verify_detects_tampered_text(tmp_path):
    snap = Snapshot(tmp_path / "snapshot")
    snap.ensure_dirs()
    snap.text_path("g").write_text("original\n", encoding="utf-8")
    snap.metadata_path("g").write_text('{"docIdentity": "x"}', encoding="utf-8")
    snap.reindex()
    assert snap.verify().is_clean

    snap.text_path("g").write_text("tampered\n", encoding="utf-8")
    report = snap.verify()

    assert not report.is_clean
    assert report.content_mismatch == ["g"]


def test_verify_flags_untracked_file(tmp_path):
    snap = Snapshot(tmp_path / "snapshot")
    snap.ensure_dirs()
    snap.text_path("known").write_text("a\n", encoding="utf-8")
    snap.metadata_path("known").write_text("{}", encoding="utf-8")
    snap.reindex()

    snap.text_path("stray").write_text("b\n", encoding="utf-8")
    report = snap.verify()

    assert report.untracked == ["stray"]
    assert not report.is_clean


def test_reindex_reports_text_without_metadata(tmp_path):
    snap = Snapshot(tmp_path / "snapshot")
    snap.ensure_dirs()
    snap.text_path("orphan").write_text("x\n", encoding="utf-8")

    report = snap.reindex()

    assert report.failed == 1
    assert "metadata missing" in report.errors[0]


def test_write_document_returns_provenance(tmp_path):
    snap = Snapshot(tmp_path / "snapshot")

    record = snap.write_document(
        "g1",
        text="Điều 1. X\n",
        metadata={"docIdentity": "9/2024/NĐ-CP", "docName": "N",
                  "updDateTime": "2024-01-01T00:00:00"},
    )

    assert record.doc_identity == "9/2024/NĐ-CP"
    assert record.content_sha256 == sha256_text("Điều 1. X\n")
    assert snap.read_text("g1") == "Điều 1. X\n"
    assert snap.read_metadata("g1")["docIdentity"] == "9/2024/NĐ-CP"
