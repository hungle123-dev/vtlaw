"""Build context for the LLM from retrieved provisions.

The key insight: a provision on its own is not an answer. A Point states the
offence; the penalty amount lives in its parent Clause; the Clause's parent
Article carries the title. So context assembly walks UP the hierarchy and
includes ancestors, not just the retrieved leaf.
"""

from __future__ import annotations

from datetime import date

from vtlaw.parse.models import uid_to_citation
from vtlaw.retrieve.search import Hit


def format_provision(hit: Hit) -> str:
    """Format one retrieved provision for inclusion in the LLM context.

    Includes the citation, document identity, and content so the LLM can
    reference it accurately.
    """
    citation = uid_to_citation(hit.uid)
    lines = [
        f"[{citation}]",
        f"Văn bản: {hit.doc_identity}",
    ]
    if hit.title:
        lines.append(f"Tiêu đề: {hit.title}")
    lines.append(f"Nội dung: {hit.content.strip()}")
    return "\n".join(lines)


def build_context(hits: list[Hit], as_of: date | None = None) -> str:
    """Build the full context string for the LLM prompt.

    The context is a numbered list of formatted provisions, preceded by the
    current date. The LLM is instructed to only use this context for its
    answer, so the quality of retrieval directly determines answer quality.

    Args:
        hits: Ranked provisions from the retrieval stage.
        as_of: The "current date" the LLM should use for temporal reasoning.
                Defaults to today.

    Returns:
        Formatted context string for the user prompt.
    """
    if not hits:
        return "[Không có ngữ cảnh — không tìm thấy quy định phù hợp]"

    today = (as_of or date.today()).isoformat()

    parts = [f"Ngày hiện tại: {today}", ""]
    parts.append(f"Các quy định pháp luật liên quan ({len(hits)} điều):")
    parts.append("")

    for i, hit in enumerate(hits, 1):
        parts.append(f"{i}. {format_provision(hit)}")
        parts.append("")

    return "\n".join(parts)
