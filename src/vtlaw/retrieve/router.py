"""Route the original user question before conversation rewriting or retrieval."""

from __future__ import annotations

import json
import logging
from typing import Literal

from vtlaw.generate.llm_client import LLMClient

log = logging.getLogger(__name__)

Intent = Literal["direct_answer", "retrieve", "reject", "cypher_query"]
_INTENTS: set[str] = {"direct_answer", "retrieve", "reject", "cypher_query"}

_SYSTEM_PROMPT = """Bạn là router cho hệ thống tra cứu pháp luật giao thông Việt Nam.
Phân loại câu hỏi vào đúng một intent:
- direct_answer: chào hỏi, hỏi bot là gì/làm được gì, giao tiếp không cần tra cứu.
- cypher_query: đếm, thống kê, liệt kê metadata đồ thị như số điều, người ký,
  hoặc lịch sử sửa đổi/bãi bỏ của văn bản.
- retrieve: hỏi nội dung, mức phạt, điều kiện, thủ tục hoặc quy định pháp luật.
- reject: hoàn toàn ngoài pháp luật giao thông Việt Nam.
Chỉ trả về JSON hợp lệ: {"intent":"direct_answer|retrieve|reject|cypher_query"}."""


class QueryRouter:
    """LLM classifier with a fail-open retrieval fallback."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def route(self, question: str) -> Intent:
        try:
            raw = self._llm.complete(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                temperature=0,
                max_tokens=64,
            )
        except Exception as exc:  # noqa: BLE001 - legal retrieval must remain available
            log.warning("intent routing failed; using retrieve: %s", exc)
            return "retrieve"
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> Intent:
        text = raw.strip()
        if text.startswith("```") and text.endswith("```"):
            text = "\n".join(text.splitlines()[1:-1]).strip()
        try:
            intent = str(json.loads(text).get("intent", "")).lower()
        except (json.JSONDecodeError, AttributeError):
            return "retrieve"
        return intent if intent in _INTENTS else "retrieve"  # type: ignore[return-value]
