"""A generated legal answer must not cite provisions absent from retrieval evidence."""

from __future__ import annotations

from inspect import signature
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

ARTICLE = Hit(
    uid="168/2024/NĐ-CP::article::6",
    score=1.0,
    doc_identity="168/2024/NĐ-CP",
    label="Article",
    content="Điều 6",
)
DISPLAYED_POINT = "168/2024/NĐ-CP::article::6::clause::3::point::a"
SIBLING_POINT = "168/2024/NĐ-CP::article::6::clause::3::point::b"
UNDISPLAYED_POINT = "168/2024/NĐ-CP::article::6::clause::3::point::c"


def _rendered_hit_context(*_args, rendered_evidence, **_kwargs):
    rendered_evidence[HIT.uid] = HIT
    return {HIT.uid: "context"}


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


def test_accepts_a_displayed_point_expanded_from_an_article_hit():
    check = assess_citations(
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
        [ARTICLE],
        evidence_uids={ARTICLE.uid, DISPLAYED_POINT},
    )

    assert check.status == "verified"


def test_accepts_a_rendered_sibling_but_not_an_undisplayed_child():
    displayed = assess_citations(
        "Theo Điểm b Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
        [HIT],
        evidence_uids={HIT.uid, SIBLING_POINT},
    )
    undisplayed = assess_citations(
        "Theo Điểm c Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
        [ARTICLE],
        evidence_uids={ARTICLE.uid, DISPLAYED_POINT},
    )

    assert displayed.status == "verified"
    assert undisplayed.status == "unsupported"
    assert undisplayed.unsupported == [UNDISPLAYED_POINT]


def test_generator_rejects_a_parent_citation_without_its_rendered_evidence_record():
    llm = MagicMock()
    llm.complete.side_effect = [
        "Theo Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
    ]
    generator = AnswerGenerator(
        MagicMock(), MagicMock(), llm, Settings(neo4j_password="test")
    )

    with patch("vtlaw.generate.answer.build_full_context", side_effect=_rendered_hit_context):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert "Điểm a Khoản 3" in answer
    assert llm.complete.call_count == 2


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
    with patch("vtlaw.generate.answer.build_full_context", side_effect=_rendered_hit_context):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert "Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP" in answer
    assert llm.complete.call_count == 2
    assert "context" in llm.complete.call_args_list[1].kwargs["messages"][1]["content"]
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
    with patch("vtlaw.generate.answer.build_full_context", side_effect=_rendered_hit_context):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert assess_citations(answer, [HIT]).status == "verified"
    assert "Dữ liệu đã hiển thị" in answer


def test_generator_does_not_cite_a_hit_the_context_builder_did_not_record():
    llm = MagicMock()
    llm.complete.side_effect = [
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
    ]
    generator = AnswerGenerator(
        MagicMock(), MagicMock(), llm, Settings(neo4j_password="test")
    )

    with patch("vtlaw.generate.answer.build_full_context", return_value={HIT.uid: "graph context"}):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert "Điểm a Khoản 3 Điều 6" not in answer
    assert llm.complete.call_count == 2


def test_generator_repairs_against_the_rendered_context_and_evidence_sidecar():
    llm = MagicMock()
    llm.complete.side_effect = [
        "Theo Điều 1 Nghị định 100/2019/NĐ-CP, bị phạt.",
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt.",
    ]
    generator = AnswerGenerator(
        MagicMock(), MagicMock(), llm, Settings(neo4j_password="test")
    )
    evidence_uids: set[str] = set()
    rendered_evidence: dict[str, Hit] = {}

    def rendered_context(*_args, evidence_uids, rendered_evidence, **_kwargs):
        evidence_uids.update({ARTICLE.uid, DISPLAYED_POINT})
        rendered_evidence[ARTICLE.uid] = ARTICLE
        rendered_evidence[DISPLAYED_POINT] = HIT
        return {ARTICLE.uid: "BẰNG CHỨNG ĐIỂM A ĐÃ HIỂN THỊ"}

    with patch("vtlaw.generate.answer.build_full_context", side_effect=rendered_context):
        answer = generator.generate_from_hits(
            "Câu hỏi",
            [ARTICLE],
            evidence_uids=evidence_uids,
            rendered_evidence=rendered_evidence,
        )

    assert llm.complete.call_count == 2
    repair_prompt = llm.complete.call_args_list[1].kwargs["messages"][1]["content"]
    assert "BẰNG CHỨNG ĐIỂM A ĐÃ HIỂN THỊ" in repair_prompt
    assert evidence_uids == {ARTICLE.uid, DISPLAYED_POINT}
    assert rendered_evidence[DISPLAYED_POINT] == HIT
    assert assess_citations(answer, [ARTICLE], evidence_uids=evidence_uids).status == "verified"


def test_generator_has_no_browser_delta_callback_or_provider_draft_stream():
    llm = MagicMock()
    llm.complete_stream.return_value = iter(
        ["Theo Điều 1 Nghị định 100/2019/NĐ-CP, ", "bị phạt."]
    )
    llm.complete.return_value = (
        "Theo Điểm a Khoản 3 Điều 6 Nghị định 168/2024/NĐ-CP, bị phạt."
    )
    generator = AnswerGenerator(
        MagicMock(), MagicMock(), llm, Settings(neo4j_password="test")
    )

    with patch("vtlaw.generate.answer.build_full_context", side_effect=_rendered_hit_context):
        answer = generator.generate_from_hits("Câu hỏi", [HIT])

    assert "on_delta" not in signature(generator.generate_from_hits).parameters
    assert "Theo Điều 1 Nghị định 100/2019/NĐ-CP" not in answer
    assert assess_citations(answer, [HIT]).status == "verified"
    llm.complete_stream.assert_not_called()
