"""Safe text-to-graph operations must never execute model-generated Cypher."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from vtlaw.graph.query_templates import StructuredGraphQueries, detect_operation


def _client(rows: list[dict]) -> MagicMock:
    client = MagicMock()
    client.session.return_value.__enter__.return_value.run.return_value.data.return_value = rows
    return client


def test_detects_only_the_supported_graph_operations_with_a_document_identity():
    assert detect_operation("Nghị định 168/2024/NĐ-CP có bao nhiêu điều?") == "article_count"
    assert detect_operation("Ai ký Nghị định 168/2024/NĐ-CP?") == "signers"
    assert (
        detect_operation("Văn bản nào sửa đổi Nghị định 100/2019/NĐ-CP?")
        == "amendment_history"
    )
    assert detect_operation("Có bao nhiêu điều trong pháp luật?") is None
    assert detect_operation("MATCH (n) RETURN n") is None


def test_article_count_uses_a_parameterized_template_not_the_question_as_cypher():
    client = _client(
        [{"doc_identity": "168/2024/NĐ-CP", "doc_name": "Nghị định 168", "count": 89}]
    )

    result = StructuredGraphQueries(client).answer("Nghị định 168/2024/NĐ-CP có bao nhiêu điều?")

    assert result is not None
    assert result.operation == "article_count"
    assert "89 điều" in result.answer
    query = client.session.return_value.__enter__.return_value.run.call_args.args[0]
    assert "$doc_identity" in query
    assert "168/2024/NĐ-CP" not in query
    params = client.session.return_value.__enter__.return_value.run.call_args.kwargs
    assert params["doc_identity"] == "168/2024/NĐ-CP"


def test_every_operation_applies_the_requested_legal_date():
    """A count or signer answer for a date the document does not apply on is wrong.

    `article_count` and `signers` previously ignored `as_of` entirely, so the UI's
    date selector silently did nothing on two of the three graph routes.
    """
    for question in (
        "Nghị định 168/2024/NĐ-CP có bao nhiêu điều?",
        "Ai ký Nghị định 168/2024/NĐ-CP?",
        "Văn bản nào sửa đổi 168/2024/NĐ-CP?",
    ):
        client = _client([])

        StructuredGraphQueries(client).answer(question, as_of=date(2024, 1, 1))

        call = client.session.return_value.__enter__.return_value.run.call_args
        assert call.kwargs["as_of"] == "2024-01-01", question
        assert "effect_date" in call.args[0], question


def test_a_document_not_yet_applicable_is_reported_as_such_not_as_missing():
    client = _client(
        [{"doc_identity": "168/2024/NĐ-CP", "applicable": False, "count": 89}]
    )

    result = StructuredGraphQueries(client).answer(
        "Nghị định 168/2024/NĐ-CP có bao nhiêu điều?", as_of=date(2024, 1, 1)
    )

    assert result is not None
    assert "2024-01-01" in result.answer
    assert "không tồn tại" not in result.answer.casefold()


def test_unknown_document_names_the_supported_operation_and_corpus_absence():
    result = StructuredGraphQueries(_client([])).answer(
        "Nghị định 999/2099/NĐ-CP có bao nhiêu điều?"
    )

    assert result is not None
    assert result.operation == "article_count"
    assert "article_count" in result.answer
    assert "không có văn bản 999/2099/NĐ-CP trong corpus" in result.answer


def test_unknown_document_is_not_confused_with_empty_amendment_history():
    result = StructuredGraphQueries(_client([])).answer(
        "Văn bản nào sửa đổi Nghị định 999/2099/NĐ-CP?"
    )

    assert result is not None
    assert result.operation == "amendment_history"
    assert "amendment_history" in result.answer
    assert "không có văn bản 999/2099/NĐ-CP trong corpus" in result.answer


def test_present_document_with_no_amendments_reports_an_empty_operation():
    client = _client(
        [
            {
                "requested_doc": "168/2024/NĐ-CP",
                "applicable": True,
                "amendments": [],
            }
        ]
    )

    result = StructuredGraphQueries(client).answer(
        "Văn bản nào sửa đổi Nghị định 168/2024/NĐ-CP?"
    )

    assert result is not None
    assert "Không có cạnh" in result.answer
    assert "không có văn bản" not in result.answer.casefold()


def test_future_only_amendments_keep_the_existing_target_as_an_empty_history():
    client = MagicMock()
    session = client.session.return_value.__enter__.return_value

    def future_only_rows():
        query = session.run.call_args.args[0].casefold()
        if "collect(" in query:
            return [
                {
                    "requested_doc": "100/2019/NĐ-CP",
                    "applicable": True,
                    "amendments": [],
                }
            ]
        return []

    session.run.return_value.data.side_effect = future_only_rows

    result = StructuredGraphQueries(client).answer(
        "Văn bản nào sửa đổi Nghị định 100/2019/NĐ-CP?", as_of=date(2025, 1, 1)
    )

    assert result is not None
    assert "Không có cạnh" in result.answer
    assert "không có văn bản" not in result.answer.casefold()


def test_amendment_history_applies_effect_and_expiry_to_the_requested_document():
    client = MagicMock()
    session = client.session.return_value.__enter__.return_value

    def rows_for_target_date_predicate():
        query = session.run.call_args.args[0]
        if "d.effect_date" in query and "d.expire_date" in query:
            return [
                {
                    "requested_doc": "100/2019/NĐ-CP",
                    "applicable": False,
                    "amendments": [],
                }
            ]
        return [
            {
                "source_uid": "123/2021/NĐ-CP::article::2",
                "target_uid": "100/2019/NĐ-CP::article::2",
                "amend_type": "sửa đổi",
            }
        ]

    session.run.return_value.data.side_effect = rows_for_target_date_predicate

    result = StructuredGraphQueries(client).answer(
        "Văn bản nào sửa đổi Nghị định 100/2019/NĐ-CP?", as_of=date(2024, 1, 1)
    )

    assert result is not None
    assert "không áp dụng" in result.answer.casefold()
    call = session.run.call_args
    assert "d.effect_date" in call.args[0]
    assert "d.expire_date" in call.args[0]
    assert call.kwargs["as_of"] == "2024-01-01"


def test_amendment_history_is_date_bounded_and_cites_the_connected_provisions():
    client = _client(
        [
            {
                "requested_doc": "100/2019/NĐ-CP",
                "applicable": True,
                "amendments": [
                    {
                        "source_uid": "123/2021/NĐ-CP::article::2::clause::35::point::a",
                        "source_doc": "123/2021/NĐ-CP",
                        "source_label": "Point",
                        "source_content": "Bãi bỏ quy định tại điểm c khoản 2.",
                        "target_uid": "100/2019/NĐ-CP::article::2::clause::2::point::c",
                        "target_doc": "100/2019/NĐ-CP",
                        "target_label": "Point",
                        "target_content": "Quy định bị bãi bỏ.",
                        "amend_type": "bãi bỏ",
                    }
                ],
            }
        ]
    )

    result = StructuredGraphQueries(client).answer(
        "Văn bản nào bãi bỏ quy định của 100/2019/NĐ-CP?", as_of=date(2025, 1, 1)
    )

    assert result is not None
    assert result.operation == "amendment_history"
    assert "123/2021/NĐ-CP" in result.answer
    assert "Điểm a Khoản 35 Điều 2 123/2021/NĐ-CP" in result.answer
    assert {source.uid for source in result.sources} == {
        "123/2021/NĐ-CP::article::2::clause::35::point::a",
        "100/2019/NĐ-CP::article::2::clause::2::point::c",
    }
    assert all(source.content for source in result.sources)
    call = client.session.return_value.__enter__.return_value.run.call_args
    assert "source.content" in call.args[0]
    assert "target.content" in call.args[0]
    assert call.kwargs["doc_identity"] == "100/2019/NĐ-CP"
    assert call.kwargs["as_of"] == "2025-01-01"
