"""Query rewriter: rewrites follow-up queries into standalone queries.

When a user asks "nếu đi xe ô tô thì sao?" after asking about xe máy penalties,
the rewriter uses chat history to produce a standalone query:
"mức phạt xe ô tô vượt đèn đỏ là bao nhiêu"

If there is no chat history, the original query is returned immediately
(no LLM call — saves latency and tokens).
"""

from __future__ import annotations

import logging
from typing import TypedDict

from vtlaw.generate.llm_client import LLMClient

log = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: str


# ruff: noqa: E501 — prompt text contains long Vietnamese lines that cannot be
#   wrapped without breaking the prompt structure.

_REWRITE_SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên xử lý ngôn ngữ tự nhiên.

Nhiệm vụ: Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng, hãy viết lại câu hỏi thành một câu truy vấn ĐỘC LẬP, HOÀN CHỈNH, có thể hiểu được mà KHÔNG cần đọc lịch sử hội thoại.

Nguyên tắc:
1. NẾU câu hỏi mới nhất đã rõ ràng và tự nó đã mang đầy đủ ý nghĩa (ví dụ: "số lượng văn bản theo từng năm", "ai là người ký văn bản mới nhất") → PHẢI trả về y nguyên văn câu hỏi đó, TUYỆT ĐỐI KHÔNG thêm thắt từ ngữ.
2. NẾU câu hỏi mới nhất có sử dụng đại từ nhân xưng, từ thay thế (ví dụ: "văn bản đó", "ông ấy", "nếu đi ô tô thì sao?") hoặc dựa vào ngữ cảnh câu trước → TÌM và THAY THẾ từ đó bằng đối tượng cụ thể từ lịch sử hội thoại để câu hỏi trở nên độc lập.
3. TUYỆT ĐỐI KHÔNG tự động thêm các từ khóa chuyên ngành như "pháp luật", "giao thông đường bộ", "luật" vào câu hỏi viết lại nếu câu hỏi gốc (và các câu liên quan trong lịch sử) không đề cập đến. Mục tiêu chỉ là lấp đầy các thông tin bị thiếu do tham chiếu chéo.
4. KHÔNG trả lời câu hỏi. Chỉ viết lại câu hỏi hoặc giữ nguyên.
5. KHÔNG giải thích. Chỉ trả về câu hỏi đã viết lại, không bọc trong dấu ngoặc kép hay markdown.

Ví dụ:
---
Lịch sử: User hỏi "vượt đèn đỏ chạy xe máy bị phạt thế nào", Bot trả lời về mức phạt xe máy.
Câu hỏi mới: "nếu đi xe ô tô thì sao?"
→ Viết lại: mức phạt xe ô tô vượt đèn đỏ
---
Lịch sử: User hỏi "ai là người ký văn bản mới nhất", Bot trả lời "Ông Trần Thanh Mẫn".
Câu hỏi mới: "số lượng văn bản theo từng năm"
→ Viết lại: số lượng văn bản theo từng năm (Vì câu hỏi đã tự đầy đủ ý nghĩa, không tham chiếu đến lịch sử)
---
Lịch sử: User hỏi "Nghị định 100 có bao nhiêu điều", Bot trả lời "Nghị định 100 có 86 điều".
Câu hỏi mới: "văn bản đó do ai ký?"
→ Viết lại: Nghị định 100 do ai ký?
---
Lịch sử: Không có.
Câu hỏi mới: "chạy quá tốc độ 20km/h bị xử phạt thế nào"
→ Viết lại: chạy quá tốc độ 20km/h
"""


class QueryRewriter:
    """Rewrites follow-up queries into standalone queries using chat history."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def rewrite(
        self,
        chat_history: list[ChatMessage],
        query: str,
        max_history_turns: int = 10,
    ) -> str:
        """Rewrite query into a standalone query given chat_history.

        If there is no history, the original query is returned immediately
        (no LLM call — saves latency and tokens).

        Args:
            chat_history: List of {"role": "user"|"assistant", "content": "..."}
            query: The latest user query (potentially a follow-up).
            max_history_turns: Maximum number of turns to include.

        Returns:
            A self-contained query string suitable for the retrieval pipeline.
        """
        if not chat_history:
            return query

        # Truncate history to avoid exceeding context window
        recent = chat_history[-max_history_turns * 2 :]

        # Build history as text (truncated assistant messages)
        history_parts = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "assistant" and len(content) > 200:
                content = content[:200] + "..."
            history_parts.append(f"{role}: {content}")

        history_text = "\n".join(history_parts)

        try:
            rewritten = self.llm.complete(
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Lịch sử:\n{history_text}\n\nCâu hỏi mới: {query}"},
                ],
                temperature=0,
                max_tokens=256,
            )
            return rewritten.strip()
        except Exception as e:
            log.warning("Rewrite failed: %s. Using original query.", e)
            return query
