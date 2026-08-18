"""Tests for the optional LLM query decomposition path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vtlaw.retrieve.query_parser import QueryDecomposer


@pytest.fixture
def mock_llm():
    return MagicMock()


class TestQueryDecomposer:
    def test_decompose_single_query(self, mock_llm):
        mock_llm.complete.return_value = (
            '[{"query": "người điều khiển xe mô tô không chấp hành '
            'hiệu lệnh đèn tín hiệu"}, {"query": "xử phạt vi phạm hành chính"}]'
        )
        subqueries = QueryDecomposer(mock_llm).decompose(
            "Vượt đèn đỏ xe máy phạt thế nào?"
        )

        assert len(subqueries) == 2

    def test_decompose_multi_violation(self, mock_llm):
        mock_llm.complete.return_value = (
            '[{"query": "không đội mũ bảo hiểm"}, '
            '{"query": "vượt đèn đỏ"}, '
            '{"query": "xử phạt vi phạm hành chính"}]'
        )

        subqueries = QueryDecomposer(mock_llm).decompose(
            "Vừa không đội mũ bảo hiểm vừa vượt đèn đỏ phạt bao nhiêu?"
        )

        assert len(subqueries) == 3

    def test_decompose_falls_back_to_the_original_query(self, mock_llm):
        mock_llm.complete.side_effect = Exception("LLM unavailable")

        subqueries = QueryDecomposer(mock_llm).decompose("Some query")

        assert subqueries == [{"query": "Some query"}]

    def test_decompose_rejects_invalid_json(self, mock_llm):
        mock_llm.complete.return_value = "invalid json"

        subqueries = QueryDecomposer(mock_llm).decompose("Some query")

        assert subqueries == [{"query": "Some query"}]

    def test_decompose_handles_a_markdown_wrapper(self, mock_llm):
        mock_llm.complete.return_value = '```json\n[{"query": "test query"}]\n```'

        subqueries = QueryDecomposer(mock_llm).decompose("Some query")

        assert subqueries == [{"query": "test query"}]
