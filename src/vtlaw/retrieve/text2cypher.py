"""Text-to-Cypher: converts natural language queries to Cypher.

Uses schema introspection from the live Neo4j database to ground the LLM.
Supports self-healing: on CypherSyntaxError, feeds the error back to the LLM
for one retry.
"""

from __future__ import annotations

import logging

from neo4j.exceptions import CypherSyntaxError, Neo4jError

from vtlaw.generate.llm_client import LLMClient
from vtlaw.graph.client import GraphClient

log = logging.getLogger(__name__)

_NODE_PROPS_QUERY = """
MATCH (n)
WITH labels(n)[0] AS label, keys(n) AS propertyNames
UNWIND propertyNames AS prop
WITH label, collect(DISTINCT prop) AS props
WHERE label IS NOT NULL
RETURN {labels: label, properties: props} AS output
"""

_REL_PROPS_QUERY = """
MATCH ()-[r]->()
WITH type(r) AS relType, keys(r) AS propertyNames
UNWIND (case when propertyNames = [] then [null] else propertyNames end) AS prop
WITH relType, collect(DISTINCT prop) AS rawProps
RETURN {type: relType, properties: [p IN rawProps WHERE p IS NOT NULL]} AS output
"""

_REL_QUERY = """
CALL db.schema.visualization()
YIELD nodes, relationships
UNWIND relationships AS rel
WITH startNode(rel) AS source, endNode(rel) AS target, type(rel) AS relType
RETURN {source: labels(source)[0], relationship: relType, target: labels(target)[0]} AS output
"""


def _schema_text(node_props, rel_props, rels) -> str:
    return f"""
  Đây là schema của cơ sở dữ liệu Neo4j.
  Các thuộc tính của node bao gồm:
  {node_props}
  Các thuộc tính quan hệ (relationship) bao gồm:
  {rel_props}
  Hướng quan hệ từ source node đến target nodes như sau:
  {rels}
  Hãy đảm bảo tuân thủ đúng các loại quan hệ và hướng của chúng.
  """


# ruff: noqa: E501 — prompt text contains long Vietnamese lines that cannot be
#   wrapped without breaking the prompt structure.

_REWRITE_SYSTEM = """Bạn là chuyên gia xử lý ngôn ngữ tự nhiên cho các câu hỏi truy vấn đến cơ sở dữ liệu đồ thị Neo4j.

Nhiệm vụ: Viết lại câu hỏi thành câu truy vấn ngôn ngữ tự nhiên dựa theo schema, sao cho có thể chuyển sang Cypher chính xác.

Nguyên tắc:
- Giữ đúng nội dung câu hỏi, không tự phát sinh thông tin.
- Không nhắc đến tên loại quan hệ (relationship type) cụ thể.
- Sắp xếp thứ tự theo hướng quan hệ: source node trước target node.
- Thuộc tính định danh: dùng "bằng" + giá trị (chữ, số, La Mã).
- Thuộc tính văn bản: dùng "chứa" + giá trị.

Quy tắc đầu ra:
- Chỉ trả về câu truy vấn hợp lệ dưới dạng ngôn ngữ tự nhiên.
- Không thêm văn bản chào hỏi, giải thích, lưu ý hay phân tích.
"""


_SYSTEM_PROMPT = """Bạn là chuyên gia viết câu truy vấn Cypher cho Neo4j.

Nhiệm vụ: Tạo câu lệnh Cypher từ câu hỏi ngôn ngữ tự nhiên (Text-to-Cypher).

Nguyên tắc:
- Phải sử dụng mối quan hệ có độ dài biến đổi: (A)-[*1..]->(B)
- Tuyệt đối không sử dụng loại quan hệ cụ thể ở MATCH.
- Không thêm node trung gian nếu câu hỏi không yêu cầu.
- Tuân thủ hướng quan hệ theo schema.
- Thuộc tính định danh: dùng toán tử =.
- Thuộc tính văn bản: dùng CONTAINS + toLower().
- Không trả về toàn bộ node object. Chỉ RETURN thuộc tính liên quan.
- Nếu không thể tạo Cypher, giải thích lý do.

Quy tắc đầu ra:
- Chỉ trả về câu lệnh Cypher hợp lệ, KHÔNG bọc trong markdown.
- Không thêm văn bản chào hỏi, giải thích.
- Trường hợp không thể tạo: đưa ra nguyên nhân.

Ví dụ:
Input: Ai là người ký Luật đường bộ?
Output: MATCH (d:Document)-[*1..]->(s:Signer) WHERE toLower(d.doc_name) CONTAINS toLower('Luật đường bộ') RETURN s.name

Input: Nghị định 168 có bao nhiêu điều?
Output: MATCH (d:Document)-[*1..]->(a:Article) WHERE toLower(d.doc_name) CONTAINS toLower('Nghị định 168') RETURN count(a)
"""


class TextToCypher:
    """Converts natural language queries to Cypher using schema introspection."""

    def __init__(
        self,
        client: GraphClient,
        llm: LLMClient | None = None,
    ):
        self._client = client
        self.llm = llm or LLMClient()
        self.schema = self._generate_schema()

    def _generate_schema(self) -> str:
        with self._client.session() as session:
            node_props = session.run(_NODE_PROPS_QUERY).data()
            rel_props = session.run(_REL_PROPS_QUERY).data()
            rels = session.run(_REL_QUERY).data()
        return _schema_text(node_props, rel_props, rels)

    def refresh_schema(self) -> None:
        self.schema = self._generate_schema()

    def run(self, question: str, history: list[dict] | None = None) -> tuple[list, str]:
        """Convert a question to Cypher and execute it.

        Returns (results, cypher_string).
        On CypherSyntaxError, retries once with the error message fed back.
        """
        rewrite = self._rewrite_question(question)
        cypher = self._construct_cypher(question, rewrite, history)

        try:
            results = self._query_database(cypher)
            return results, cypher
        except (CypherSyntaxError, Neo4jError) as e:
            if history is not None:
                return [{"error": "Cypher syntax error", "detail": str(e)}], cypher
            log.warning("Cypher error, retrying with healing: %s", str(e)[:100])
            return self.run(
                question,
                [
                    {"role": "assistant", "content": cypher},
                    {
                        "role": "user",
                        "content": (
                            f"Câu truy vấn này trả về lỗi sau: {str(e)} "
                            "Hãy cung cấp câu truy vấn đã được cải thiện."
                        ),
                    },
                ],
            )

    def _rewrite_question(self, question: str) -> str:
        system = _REWRITE_SYSTEM + f"\nSchema:\n{self.schema}"
        response = self.llm.complete(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return response.strip()

    def _construct_cypher(
        self,
        question: str,
        rewrite: str,
        history: list[dict] | None = None,
    ) -> str:
        system = _SYSTEM_PROMPT + f"\nSchema:\n{self.schema}"
        user_msg = f"Câu hỏi gốc: {question}\nCâu truy vấn đã viết lại: {rewrite}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]
        if history:
            messages.extend(history)
        return self.llm.complete(messages=messages, temperature=0).strip()

    def _query_database(self, cypher: str, params: dict | None = None) -> list:
        with self._client.session() as session:
            result = session.run(cypher, params or {})
            output = [r.values() for r in result]
            output.insert(0, result.keys())
            return output
