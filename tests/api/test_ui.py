"""The portfolio UI is served by the same deployable FastAPI application."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from vtlaw.api.app import ChatRequest, app


def test_portfolio_ui_is_available_at_the_root_without_initialising_the_pipeline():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "vtlaw" in response.text
    assert "Legal Graph RAG" in response.text


def test_ui_offers_every_retrieval_profile_the_api_accepts():
    """A profile the UI cannot select is a profile no reviewer will ever see."""
    page = TestClient(app).get("/").text
    accepted = ChatRequest.model_fields["profile"].annotation.__args__

    for profile in accepted:
        assert f'value="{profile}"' in page, profile


def test_quality_profile_is_the_default_selected_by_the_ui_and_api():
    page = TestClient(app).get("/").text

    assert ChatRequest(question="q").profile == "quality"
    assert '<option value="quality" selected>' in page


def test_ui_discloses_when_a_requested_rerank_was_not_applied():
    """A requested rerank must not silently present a baseline fallback as reranked."""
    page = TestClient(app).get("/").text

    assert "rerank skipped" in page


def test_ui_ships_no_external_runtime_dependency():
    """The demo must render offline; a CDN outage cannot break the portfolio.

    Asserting `"http://" not in page` passed for the wrong reason: it allowed
    `<script src="//cdn.example/x.js">`, an uppercase scheme, or a same-host
    absolute URL that is still a runtime fetch. Parse the actual load-bearing
    URLs instead.
    """
    page = TestClient(app).get("/").text

    fetched = [
        *re.findall(r"""<script[^>]+\bsrc\s*=\s*["']([^"']+)""", page, re.IGNORECASE),
        *re.findall(
            r"""<link[^>]+\brel\s*=\s*["']stylesheet["'][^>]*\bhref\s*=\s*["']([^"']+)""",
            page,
            re.IGNORECASE,
        ),
        *re.findall(r"""@import\s+(?:url\()?["']([^"']+)""", page, re.IGNORECASE),
        *re.findall(r"""<(?:img|iframe|source)[^>]+\bsrc\s*=\s*["']([^"']+)""",
                    page, re.IGNORECASE),
    ]

    # Nothing at all is the only passing state: every style and script is inline.
    assert fetched == [], fetched


def test_ui_uses_local_time_for_the_default_legal_date():
    """`toISOString()` is UTC — in Vietnam (UTC+7) it yields yesterday until 07:00.

    `as_of` drives the effective-date filter, so a UTC default silently queries
    the wrong legal day for the first seven hours of every day.
    """
    page = TestClient(app).get("/").text

    assert "getTimezoneOffset" in page


def test_ui_guards_against_a_second_concurrent_submission():
    """Enter reaches requestSubmit() regardless of the button's disabled state.

    Two in-flight requests both serialize the same history and push their turns
    in completion order, so the transcript can end up out of order.
    """
    page = TestClient(app).get("/").text

    assert "inFlight" in page


def test_ui_does_not_shadow_the_browser_history_api():
    page = TestClient(app).get("/").text

    assert not re.search(r"^\s*const history\b", page, re.MULTILINE)
    assert "chatHistory" in page


def test_ui_uses_the_streaming_contract_and_current_benchmark_copy():
    page = TestClient(app).get("/").text

    assert '"/chat/stream"' in page
    assert "getReader()" in page
    assert re.search(r"0\.9025.*QA_NLP Recall@8", page, re.DOTALL)
    assert re.search(r"0\.6691.*QA Part 2–5 Recall@8", page, re.DOTALL)
    assert "0.86" not in page


def test_ui_uses_a_conversational_shell_with_a_real_new_chat_action():
    """The portfolio should feel like a chat product, not a benchmark landing page."""
    page = TestClient(app).get("/").text

    assert 'class="app-layout"' in page
    assert 'class="app-sidebar"' in page
    assert 'id="thread-list"' in page
    assert 'id="new-chat"' in page
    assert "function clearConversation()" in page
    assert '$("messages").replaceChildren()' in page
    assert "chatHistory.length = 0" in page


def test_ui_keeps_retrieval_trace_available_without_overwhelming_the_answer():
    page = TestClient(app).get("/").text

    assert 'el("details", "answer-inspect")' in page
    assert "Chi tiết truy xuất" in page
    assert "Nguồn đã viện dẫn" in page


def test_ui_translates_machine_citation_statuses_for_the_reader():
    page = TestClient(app).get("/").text

    assert "Không cần trích dẫn cho dạng tra cứu này" in page
    assert '"Trích dẫn " + data.citation_status' not in page


def test_screenshot_harness_waits_for_the_visible_final_state():
    harness = (Path(__file__).parents[2] / "scripts" / "capture_screenshots.py").read_text(
        encoding="utf-8"
    )

    assert 'page.wait_for_selector("#messages .answer-inspect"' in harness


def test_public_docs_label_current_and_historical_benchmarks_truthfully():
    readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")

    assert "git_dirty: false" in readme
    assert "QA_NLP (94), query composition" in readme
    assert re.search(r"0\.9025.*0\.1769.*0\.7061", readme, re.DOTALL)
    assert re.search(r"0\.6691.*0\.2037.*0\.5655", readme, re.DOTALL)
    assert "0.2766" not in readme


def test_ui_renders_safe_chunks_without_draft_or_replace_logic():
    page = TestClient(app).get("/").text

    assert 'event.name === "status"' in page
    assert 'event.name === "delta"' in page
    assert 'event.name === "replace"' not in page
    assert "has-draft" not in page
    assert "Bản nháp" not in page
    assert "finishAssistant(pending, event.data)" in page


def test_ui_renders_retrieved_and_cited_sources_from_the_current_contract():
    page = TestClient(app).get("/").text

    assert "data.retrieved_candidates" in page
    assert "data.cited_sources" in page
    assert "Nguồn đã truy xuất" in page
    assert "Nguồn đã viện dẫn" in page
    assert "data.sources" not in page
