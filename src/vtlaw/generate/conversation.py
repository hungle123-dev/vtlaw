"""Conversation-aware query rewriting for follow-up legal questions."""

from __future__ import annotations

import logging
from typing import TypedDict

from vtlaw.generate.llm_client import LLMClient

log = logging.getLogger(__name__)


class ChatTurn(TypedDict):
    role: str
    content: str


_SYSTEM_PROMPT = """Viết lại câu hỏi tiếp theo thành truy vấn pháp luật giao thông Việt Nam độc lập.
Giữ nguyên phương tiện, chủ thể, hành vi, số liệu và mốc thời gian từ hội thoại.
Không trả lời, giải thích hoặc thêm dữ kiện. Chỉ trả về truy vấn đã viết lại."""

_USER_TEMPLATE = """Hội thoại trước đó:
{transcript}

Câu hỏi tiếp theo cần viết lại: {question}

Truy vấn độc lập:"""


class ConversationRewriter:
    """Resolve a follow-up against a bounded, client-provided conversation."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def rewrite(self, history: list[ChatTurn], question: str) -> str:
        if not history:
            return question

        # The transcript goes inside ONE user message rather than being replayed
        # as real chat turns. Replayed, the model sees a dialogue ending on a
        # user question and completes it as the assistant: the follow-up "Còn xe
        # ô tô thì sao?" came back as "Mức phạt ... là từ 3.000.000 đồng đến
        # 5.000.000 đồng" — an invented figure, which then became the retrieval
        # query. The system prompt loses to the message structure; quoting the
        # history as data does not give it one to continue.
        transcript = "\n".join(
            f"{'Người dùng' if turn['role'] == 'user' else 'Trợ lý'}: {turn['content']}"
            for turn in history
        )
        prompt = _USER_TEMPLATE.format(transcript=transcript, question=question)
        try:
            rewritten = self._llm.complete(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                # A rewritten query is one line. Capping it means a model that
                # starts answering anyway gets truncated instead of shipping a
                # paragraph of invented law into retrieval.
                max_tokens=120,
            ).strip()
        except Exception as exc:  # noqa: BLE001 - retrieval must remain available
            log.warning("conversation rewrite failed; using original query: %s", exc)
            return question
        return rewritten or question
