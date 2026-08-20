"""Capture portfolio screenshots from the running vtlaw API.

Dev tool, not shipped: run with
    uv run --with playwright==1.49.1 python scripts/capture_screenshots.py
against `vtlaw api serve` on 127.0.0.1:18080. Uses the system Edge channel so
no Playwright browser download is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:18080"
OUT = Path("docs/images")
VIEWPORT = {"width": 1280, "height": 900}

# (filename, question, profile, history) — one per retrieval path worth showing.
SHOTS = [
    ("ui-hybrid-answer", "Vượt đèn đỏ xe máy bị phạt bao nhiêu?", "baseline", []),
    ("ui-exact-citation", "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP", "baseline", []),
    ("ui-graph-template", "Nghị định 168/2024/NĐ-CP có bao nhiêu điều?", "baseline", []),
    (
        "ui-followup-rewrite",
        "Còn xe ô tô thì sao?",
        "baseline",
        [
            ("user", "Vượt đèn đỏ xe máy bị phạt bao nhiêu?"),
            ("assistant", "Mức phạt là từ 1.000.000 đồng đến 2.000.000 đồng."),
        ],
    ),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        for name, question, profile, history in SHOTS:
            page.goto(BASE, wait_until="load")
            page.select_option("#profile", profile)
            # Seed multi-turn state the way a real session would have it.
            for role, content in history:
                page.evaluate(
                    "([r, c]) => chatHistory.push({role: r, content: c})", [role, content]
                )
                page.evaluate("([r, c]) => addMessage(c, r === 'user' ? 'user' : 'assistant')",
                              [role, content])

            page.fill("#question", question)
            page.click("#send")
            # The pending bubble carries .thinking until the response replaces it.
            page.wait_for_selector(".thinking", state="detached", timeout=90_000)
            page.wait_for_timeout(400)

            # The transcript scrolls internally, so full_page alone captures a
            # box already scrolled past the user's question. Expand it for the
            # shot only — the max-height is right for real use.
            page.evaluate(
                "() => { const m = document.getElementById('messages');"
                " m.style.maxHeight = 'none'; m.style.overflow = 'visible'; }"
            )
            page.wait_for_timeout(200)
            target = OUT / f"{name}.png"
            page.screenshot(path=str(target), full_page=True)
            print(f"wrote {target}")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
