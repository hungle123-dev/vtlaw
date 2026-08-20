"""A generated legal answer must not cite provisions absent from retrieval evidence."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vtlaw.config import Settings
from vtlaw.generate.answer import AnswerGenerator
from vtlaw.generate.citations import assess_citations
from vtlaw.retrieve.search import Hit

HIT = Hit(
    uid="168/2024/NĐ-CP::article::6::clause::3::point::a",
    score=1.0,
    doc_identity="168/2024/NĐ-CP",
    label="Point",
    content="Nội dung",
)


def test_accepts_a_source_citation_or_its_parent_provision():
    check = assess_citations(
        "Theo Điều 6 Nghị định 168/2024/NĐ-CP, người điều khiển bị xử phạt.", [HIT]
    )

    assert check.status == "verified"
    assert check.unsupported == []


def test_flags_a_citation_not_supported_by_the_retrieved_sources():
    check = assess_citations(
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 100/2019/NĐ-CP, người điều khiển bị xử phạt.",
        [HIT],
    )

    assert check.status == "unsupported"
    assert check.unsupported == ["100/2019/NĐ-CP::article::6::clause::3::point::a"]


def test_flags_a_legal_answer_that_omits_a_citation():
    check = assess_citations("Người điều khiển bị xử phạt tiền.", [HIT])

    assert check.status == "missing"


def test_a_document_identity_with_trailing_junk_is_not_read_as_a_real_citation():
    """"168/2024/NĐ-CPx" must not verify against 168/2024/NĐ-CP.

    Without a trailing word boundary the regex matched the valid prefix and a
    hallucinated document identity passed the provenance check.
    """
    check = assess_citations(
        "Theo Điều 6 Nghị định 168/2024/NĐ-CPx, người điều khiển bị xử phạt.", [HIT]
    )

    assert check.status == "missing"
    assert check.cited == []


def test_generator_repairs_an_unsupported_citation_before_returning_an_answer():
    llm = MagicMock()
    llm.complete.side_effect = [
        "Theo Điều 1 Nghị định 100/2019/NĐ-CP, bị phạt.",
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
    ]
    generator = AnswerGenerator(
        MagicMock(), MagicMock(), llm, Settings(neo4j_password="test")
    )
    with patch("vtlaw.generate.answer.build_full_context", return_value={HIT.uid: "context"}):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert "Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP" in answer
    assert llm.complete.call_count == 2
    assert "Nội dung" in llm.complete.call_args_list[1].kwargs["messages"][1]["content"]
    assert assess_citations(answer, [HIT]).status == "verified"


def test_generator_returns_a_grounded_fallback_when_repair_still_cites_outside_evidence():
    llm = MagicMock()
    llm.complete.side_effect = [
        "Theo Điều 1 Nghị định 100/2019/NĐ-CP, bị phạt.",
        "Theo Điều 2 Nghị định 100/2019/NĐ-CP, bị phạt.",
    ]
    generator = AnswerGenerator(
        MagicMock(), MagicMock(), llm, Settings(neo4j_password="test")
    )
    with patch("vtlaw.generate.answer.build_full_context", return_value={HIT.uid: "context"}):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert assess_citations(answer, [HIT]).status == "verified"
    assert "Dữ liệu đã truy xuất" in answer
