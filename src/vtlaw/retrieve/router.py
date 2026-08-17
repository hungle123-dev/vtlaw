"""Query router: classifies user queries into 4 intents.

Determines the appropriate response strategy for each query:
  - direct_answer: Simple greeting or direct question that doesn't need retrieval
  - retrieve: Complex question requiring provision retrieval
  - reject: Off-topic query (not traffic law)
  - cypher_query: Analytical query that can be answered via Cypher
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from vtlaw.generate.llm_client import LLMClient

log = logging.getLogger(__name__)

IntentType = Literal["direct_answer", "retrieve", "reject", "cypher_query"]

# ruff: noqa: E501 — prompt text contains long Vietnamese lines that cannot be
#   wrapped without breaking the prompt structure.

_ROUTER_SYSTEM_PROMPT = """Bạn là một hệ thống phân loại câu hỏi (Router) cho một Chatbot Pháp luật Giao thông Đường bộ Việt Nam.

Nhiệm vụ của bạn là phân loại câu hỏi của người dùng vào một trong bốn loại (intent) sau:

1. "direct_answer": Câu hỏi chào hỏi, giao tiếp cơ bản với bot (Ví dụ: "bạn là ai", "chào bot", "bạn làm được gì"), hoặc các câu logic đơn giản không yêu cầu tra cứu luật pháp.
2. "cypher_query": Các câu hỏi phân tích dữ liệu, đếm số lượng, tổng hợp hoặc thống kê thông tin từ cơ sở dữ liệu (Ví dụ: "có bao nhiêu văn bản", "Nghị định 100 có bao nhiêu điều", "ai ký luật này", "liệt kê các văn bản do ông X ký"), nói chung là những tác vụ không thể chỉ dựa vào truy xuất dữ liệu, mà phải thao tác trên các đỉnh và cạnh của cơ sở dữ liệu đồ thị.
3. "retrieve": Các câu hỏi liên quan đến nội dung của luật giao thông đường bộ, quy tắc, mức phạt vi phạm, thủ tục hành chính, yêu cầu phải tra cứu cơ sở dữ liệu pháp luật để trả lời chính xác.
4. "reject": Các câu hỏi về các lĩnh vực hoàn toàn không liên quan đến luật giao thông (ví dụ: y tế, lập trình, nấu ăn, toán học phức tạp, chính trị, luật hình sự...). Đối với những câu này, Chatbot sẽ từ chối trả lời.

Quy tắc đầu ra:
- CHỈ trả về một JSON object với cấu trúc: {"intent": "<loại_intent>"}
- KHÔNG giải thích, KHÔNG thêm bất kỳ văn bản nào khác.
- Tuyệt đối không sử dụng code fence hay bọc markdown (VD: KHÔNG dùng ```json ... ```). Output phải là raw text JSON hợp lệ.

Ví dụ:
Input: "Xin chào bạn" -> Output: {"intent": "direct_answer"}
Input: "Nghị định 168 có bao nhiêu điều?" -> Output: {"intent": "cypher_query"}
Input: "Vượt đèn đỏ bị phạt bao nhiêu tiền?" -> Output: {"intent": "retrieve"}
Input: "Hướng dẫn tôi cách nấu món phở bò" -> Output: {"intent": "reject"}
"""

_ROUTER_USER_PROMPT = """Câu hỏi của người dùng:
{query}
"""


class QueryRouter:
    """Classifies user queries to determine the appropriate response strategy."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def route(self, query: str) -> IntentType:
        """Classify the user query. Fallback to 'retrieve' on failure."""
        try:
            raw_response = self.llm.complete(
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": _ROUTER_USER_PROMPT.format(query=query)},
                ],
                temperature=0,
                max_tokens=64,
            )
            return self._parse_intent(raw_response)
        except Exception as e:
            log.warning("Router error: %s. Falling back to 'retrieve'.", e)
            return "retrieve"

    def _parse_intent(self, response_text: str) -> IntentType:
        """Parse the JSON response to extract the intent."""
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            intent = data.get("intent", "retrieve").lower()
            if intent in ["direct_answer", "retrieve", "reject", "cypher_query"]:
                return intent
        except json.JSONDecodeError:
            # Simple fallback parsing using string matching
            if '"intent": "direct_answer"' in text or "'intent': 'direct_answer'" in text:
                return "direct_answer"
            elif '"intent": "reject"' in text or "'intent': 'reject'" in text:
                return "reject"
            elif '"intent": "cypher_query"' in text or "'intent': 'cypher_query'" in text:
                return "cypher_query"

        return "retrieve"
