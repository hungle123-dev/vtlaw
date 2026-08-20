"""The portfolio UI is served by the same deployable FastAPI application."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from vtlaw.api.app import app


def test_portfolio_ui_is_available_at_the_root_without_initialising_the_pipeline():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "vtlaw" in response.text
    assert "Legal Graph RAG" in response.text


def test_ui_offers_every_retrieval_profile_the_api_accepts():
    """A profile the UI cannot select is a profile no reviewer will ever see."""
    from vtlaw.api.app import ChatRequest

    page = TestClient(app).get("/").text
    accepted = ChatRequest.model_fields["profile"].annotation.__args__

    for profile in accepted:
        assert f'value="{profile}"' in page, profile


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
