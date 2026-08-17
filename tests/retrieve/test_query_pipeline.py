"""Tests for query pipeline: router, parser, rewriter.

Tests cover:
- QueryRouter: classifies queries into 4 intents
- QueryDecomposer: splits complex queries into sub-queries
- QueryRewriter: rewrites follow-up queries using chat history
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vtlaw.retrieve.query_parser import QueryDecomposer
from vtlaw.retrieve.query_rewriter import QueryRewriter
from vtlaw.retrieve.router import QueryRouter


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class TestQueryRouter:
    def test_route_retrieve_intent(self, mock_llm):
        """Complex legal question should route to 'retrieve'."""
        mock_llm.complete.return_value = '{"intent": "retrieve"}'
        router = QueryRouter(llm=mock_llm)

        intent = router.route("Vượt đèn đỏ bị phạt bao nhiêu?")

        assert intent == "retrieve"
        mock_llm.complete.assert_called_once()

    def test_route_cypher_query_intent(self, mock_llm):
        """Counting/aggregation queries should route to 'cypher_query'."""
        mock_llm.complete.return_value = '{"intent": "cypher_query"}'
        router = QueryRouter(llm=mock_llm)

        intent = router.route("Nghị định 168 có bao nhiêu điều?")

        assert intent == "cypher_query"

    def test_route_direct_answer_intent(self, mock_llm):
        """Simple greeting should route to 'direct_answer'."""
        mock_llm.complete.return_value = '{"intent": "direct_answer"}'
        router = QueryRouter(llm=mock_llm)

        intent = router.route("Chào bạn")

        assert intent == "direct_answer"

    def test_route_reject_intent(self, mock_llm):
        """Off-topic query should route to 'reject'."""
        mock_llm.complete.return_value = '{"intent": "reject"}'
        router = QueryRouter(llm=mock_llm)

        intent = router.route("Hướng dẫn tôi cách nấu phở")

        assert intent == "reject"

    def test_route_fallback_on_error(self, mock_llm):
        """On LLM error, should fallback to 'retrieve'."""
        mock_llm.complete.side_effect = Exception("LLM unavailable")
        router = QueryRouter(llm=mock_llm)

        intent = router.route("Some query")

        assert intent == "retrieve"

    def test_route_fallback_on_invalid_json(self, mock_llm):
        """On invalid JSON response, should fallback to 'retrieve'."""
        mock_llm.complete.return_value = "invalid json response"
        router = QueryRouter(llm=mock_llm)

        intent = router.route("Some query")

        assert intent == "retrieve"


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------


class TestQueryDecomposer:
    def test_decompose_single_query(self, mock_llm):
        """Single violation should produce 1-2 sub-queries."""
        mock_llm.complete.return_value = (
            '[{"query": "người điều khiển xe mô tô không chấp hành '
            'hiệu lệnh đèn tín hiệu"}, {"query": "xử phạt vi phạm hành chính"}]'
        )
        decomposer = QueryDecomposer(llm=mock_llm)

        subqueries = decomposer.decompose("Vượt đèn đỏ xe máy phạt thế nào?")

        assert len(subqueries) == 2
        assert all("query" in sq for sq in subqueries)

    def test_decompose_multi_violation(self, mock_llm):
        """Multiple violations should produce multiple sub-queries."""
        mock_llm.complete.return_value = (
            '[{"query": "không đội mũ bảo hiểm"}, '
            '{"query": "vượt đèn đỏ"}, '
            '{"query": "xử phạt vi phạm hành chính"}]'
        )
        decomposer = QueryDecomposer(llm=mock_llm)

        subqueries = decomposer.decompose(
            "Vừa không đội mũ bảo hiểm vừa vượt đèn đỏ phạt bao nhiêu?"
        )

        assert len(subqueries) == 3
        assert all("query" in sq for sq in subqueries)

    def test_decompose_fallback_on_error(self, mock_llm):
        """On LLM error, should return original query."""
        mock_llm.complete.side_effect = Exception("LLM unavailable")
        decomposer = QueryDecomposer(llm=mock_llm)

        subqueries = decomposer.decompose("Some query")

        assert len(subqueries) == 1
        assert subqueries[0]["query"] == "Some query"

    def test_decompose_fallback_on_invalid_json(self, mock_llm):
        """On invalid JSON, should return original query."""
        mock_llm.complete.return_value = "invalid json"
        decomposer = QueryDecomposer(llm=mock_llm)

        subqueries = decomposer.decompose("Some query")

        assert len(subqueries) == 1
        assert subqueries[0]["query"] == "Some query"

    def test_decompose_handles_markdown_wrapper(self, mock_llm):
        """Should handle JSON wrapped in markdown code blocks."""
        mock_llm.complete.return_value = '```json\n[{"query": "test query"}]\n```'
        decomposer = QueryDecomposer(llm=mock_llm)

        subqueries = decomposer.decompose("Some query")

        assert len(subqueries) == 1
        assert subqueries[0]["query"] == "test query"


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------


class TestQueryRewriter:
    def test_rewrite_followup_query(self, mock_llm):
        """Follow-up query should be rewritten using chat history."""
        mock_llm.complete.return_value = "mức phạt xe ô tô vượt đèn đỏ"
        rewriter = QueryRewriter(llm=mock_llm)

        history = [
            {"role": "user", "content": "vượt đèn đỏ xe máy phạt bao nhiêu"},
            {"role": "assistant", "content": "Theo NĐ 100/2019..."},
        ]

        rewritten = rewriter.rewrite(history, "nếu đi xe ô tô thì sao?")

        assert rewritten == "mức phạt xe ô tô vượt đèn đỏ"
        mock_llm.complete.assert_called_once()

    def test_rewrite_no_history_returns_original(self, mock_llm):
        """Without chat history, should return original query (no LLM call)."""
        rewriter = QueryRewriter(llm=mock_llm)

        rewritten = rewriter.rewrite([], "vượt đèn đỏ phạt bao nhiêu")

        assert rewritten == "vượt đèn đỏ phạt bao nhiêu"
        mock_llm.complete.assert_not_called()

    def test_rewrite_fallback_on_error(self, mock_llm):
        """On LLM error, should return original query."""
        mock_llm.complete.side_effect = Exception("LLM unavailable")
        rewriter = QueryRewriter(llm=mock_llm)

        history = [{"role": "user", "content": "previous question"}]

        rewritten = rewriter.rewrite(history, "follow-up question")

        assert rewritten == "follow-up question"

    def test_rewrite_truncates_long_history(self, mock_llm):
        """Should truncate chat history to max_history_turns."""
        mock_llm.complete.return_value = "rewritten query"
        rewriter = QueryRewriter(llm=mock_llm)

        # Create 20 turns of history
        history = []
        for i in range(20):
            history.append({"role": "user", "content": f"question {i}"})
            history.append({"role": "assistant", "content": f"answer {i}"})

        rewritten = rewriter.rewrite(history, "follow-up", max_history_turns=5)

        assert rewritten == "rewritten query"
        # Should have truncated to last 10 messages (5 turns * 2)
        call_args = mock_llm.complete.call_args
        user_msg = call_args[1]["messages"][1]["content"]
        # Should contain last 10 messages worth of history
        assert "question 19" in user_msg or "answer 19" in user_msg
