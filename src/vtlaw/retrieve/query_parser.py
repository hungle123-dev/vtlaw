"""Query decomposition: split complex queries into sub-queries.

A multi-violation question like "vừa vượt đèn đỏ vừa không đội mũ bảo hiểm
phạt bao nhiêu?" should be split into separate sub-queries so each violation
gets its own retrieval pass. The results are then merged via RRF.
"""

from __future__ import annotations

import json
import logging
from typing import TypedDict

from vtlaw.generate.llm_client import LLMClient

log = logging.getLogger(__name__)


class SubQuery(TypedDict):
    query: str


# ruff: noqa: E501 — prompt text contains long Vietnamese lines that cannot be
#   wrapped without breaking the prompt structure.
#
# The JSON examples below use SINGLE braces. This prompt is passed to the model
# verbatim — only the user prompt goes through .format() — so `{{` is not an
# escape here, it is two literal braces. It was written as `{{"query": ...}}`
# and the model faithfully copied the doubling into its output, which then failed
# json.loads and sent every question down the fallback path. Measured: 19 of 19
# decompositions lost that way, silently, because the fallback returns the
# original query and logs at WARNING.

_DECOMPOSE_SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế truy vấn cho hệ thống RAG tra cứu Luật Giao thông đường bộ Việt Nam.

Nhiệm vụ:
Chuyển câu hỏi của người dùng thành một tập hợp sub-query tối ưu cho truy xuất văn bản pháp luật giao thông. Mục tiêu chính là rút trúng đầy đủ các quy tắc giao thông, điều kiện phương tiện, và khung xử phạt vi phạm hành chính/hình sự cần thiết.

ĐỂ ĐẢM BẢO CHẤT LƯỢNG, BẠN PHẢI TUÂN THỦ NGHIÊM NGẶT 9 NGUYÊN TẮC SAU ĐÂY, CHIA LÀM 3 NHÓM:

=== NHÓM 1: RÀ SOÁT TỪ VỰNG VÀ CHỦ THỂ (PHẢI LÀM ĐẦU TIÊN) ===

1. Xử lý thiếu hụt loại phương tiện (QUAN TRỌNG):
   - Nếu câu hỏi KHÔNG nói rõ loại phương tiện, BẮT BUỘC tạo các sub-query riêng biệt cho các loại phương tiện thông dụng (cụ thể là "xe mô tô" và "xe ô tô") để đảm bảo độ phủ dữ liệu, miễn là hành vi đó có thể áp dụng cho loại xe đó.

2. Chuẩn hóa thuật ngữ giao thông có chọn lọc (BẮT BUỘC DỊCH):
   - Chuyển từ lóng sang thuật ngữ Luật Giao thông. Ví dụ:
     + "vượt đèn đỏ" -> "không chấp hành hiệu lệnh của đèn tín hiệu giao thông"
     + "bằng lái" -> "giấy phép lái xe"
     + "cà vẹt" -> "giấy đăng ký xe"
     + "say xỉn", "nhậu" -> "điều khiển phương tiện mà trong máu hoặc hơi thở có nồng độ cồn"
     + "lấn tuyến", "đi sai làn" -> "đi không đúng phần đường, làn đường"
     + "xe máy" -> "xe mô tô, xe gắn máy"
     + "không đội mũ bảo hiểm" -> "người điều khiển, người ngồi trên xe mô tô không đội mũ bảo hiểm"

3. Không tự bịa thêm dữ kiện:
   - Chỉ tạo sub-query dựa trên hành vi, loại xe, độ tuổi có thật hoặc hàm ý trực tiếp trong câu gốc.
   - Không tự động thêm các vi phạm (như không xi nhan, thiếu gương) nếu người dùng không nhắc tới.

=== NHÓM 2: BÓC TÁCH VÀ BẢO TOÀN NGỮ NGHĨA ===

4. Ưu tiên truy xuất đầy đủ thông tin pháp lý cần thiết:
   - Nếu câu hỏi chứa một chuỗi nhiều lỗi vi phạm độc lập (ví dụ: vừa không mũ, vừa vượt đèn đỏ, vừa không bằng lái), bắt buộc tách mỗi lỗi thành một sub-query.
   - Đối với câu hỏi phân biệt (ví dụ: các loại biển báo, các loại xe), tách riêng từng đối tượng.
   - Đối với câu hỏi đơn giản, thì không cần trả nhiều subquery, 1 subquery là đủ (hoặc 2 nếu cần tách riêng hành vi và mức phạt).

5. Không làm mất hoặc thay đổi các yếu tố pháp lý quan trọng:
   - Giữ nguyên và phân biệt rõ loại phương tiện.
   - Giữ nguyên các tình tiết định khung định lượng: độ tuổi, vận tốc vượt quá (km/h), mức độ nồng độ cồn, hậu quả, loại đường.

6. Giữ mục tiêu tra cứu chế tài của câu hỏi:
   - Nếu hỏi về tiền phạt, giữ các từ khóa: "mức xử phạt", "xử phạt vi phạm hành chính".
   - Nếu hỏi về hình phạt bổ sung: "tước quyền sử dụng giấy phép lái xe", "tạm giữ phương tiện".
   - Nếu hỏi về hậu quả nghiêm trọng: "truy cứu trách nhiệm hình sự vi phạm quy định về tham gia giao thông".

7. Khi nào được tổng quát hóa chủ thể:
   - Tuyệt đối không được làm mờ/gom chung các vai trò pháp lý đặc thù như: "người điều khiển phương tiện", "chủ phương tiện", "người ngồi trên xe", "người giao xe". Sự khác biệt giữa người lái và chủ xe là cực kỳ quan trọng.

=== NHÓM 3: TỐI ƯU CHO TÌM KIẾM VÀ ĐẦU RA ===

8. Tối ưu cho tìm kiếm, không phải diễn giải:
   - Mỗi sub-query phải là một cụm từ tìm kiếm ngắn, rõ, giàu từ khóa. Không dùng từ nghi vấn (bao nhiêu tiền, thế nào).

9. Số lượng và loại bỏ trùng lặp:
   - Tối thiểu 1, tối đa 6 sub-query. Loại bỏ các sub-query trùng ý hoặc quá gần nhau.

=== QUY TẮC ĐẦU RA BẮT BUỘC ===
- Chỉ trả về JSON array hợp lệ. Mỗi phần tử có đúng một khóa: "query".
- Tuyệt đối không bọc trong markdown (KHÔNG dùng ```json).
- Không giải thích, không thêm bất kỳ văn bản nào khác.
- Dùng đúng cú pháp JSON: một dấu ngoặc nhọn cho mỗi object, ví dụ {"query": "..."}.

Một số ví dụ:
Input: "Lỗi vượt đèn đỏ phạt thế nào?"
Output:
[
  {"query": "người điều khiển xe mô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông"},
  {"query": "người điều khiển xe ô tô không chấp hành hiệu lệnh của đèn tín hiệu giao thông"},
  {"query": "mức xử phạt vi phạm hành chính"}
]

Input: "Lỗi chạy xe máy không đội mũ bảo hiểm bị phạt gì"
Output:
[
  {"query": "người điều khiển, người ngồi trên xe mô tô không đội mũ bảo hiểm"},
  {"query": "xử phạt vi phạm hành chính"}
]

Input: "Uống 1 lon bia rồi chạy xe điện có bị phạt không?"
Output:
[
  {"query": "điều khiển xe điện khi trong máu hoặc hơi thở có nồng độ cồn"},
  {"query": "xử phạt vi phạm hành chính đối với hành vi điều khiển xe điện có nồng độ cồn"}
]
"""

_DECOMPOSE_USER_PROMPT = """Câu hỏi giao thông cần phân tích:
{query}

Yêu cầu:
- Trả về đúng một JSON array định dạng hợp lệ.
- Bắt đầu bằng [ và kết thúc bằng ].
- Không giải thích, không bọc code fence.
"""


class QueryDecomposer:
    """Handles parsing and decomposing user queries into sub-queries."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def decompose(self, query: str) -> list[SubQuery]:
        """Decompose a user query into a list of SubQuery dictionaries."""
        try:
            raw_str = self.llm.complete(
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
                    {"role": "user", "content": _DECOMPOSE_USER_PROMPT.format(query=query)},
                ],
                temperature=0,
            )
        except Exception as e:
            log.warning("Decompose error: %s. Using original query.", e)
            return [{"query": query}]

        fallback = self._parse_json_fallback(raw_str)
        validated = [
            {"query": str(item["query"])}
            for item in fallback
            if isinstance(item, dict) and "query" in item
        ]

        if not validated:
            log.warning("LLM generated invalid subqueries. Raw output: %s", raw_str[:200])
            return [{"query": query}]

        return validated

    @staticmethod
    def _parse_json_fallback(text: str) -> list[dict]:
        """Try to extract JSON array from LLM output, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    json_str = text[start : end + 1]
                    return json.loads(json_str)
            except Exception:
                pass
            return []
