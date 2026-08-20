"""Intent routing keeps non-legal chat out of the legal retrieval path."""

from __future__ import annotations

from unittest.mock import MagicMock

from vtlaw.retrieve.router import QueryRouter


def test_router_accepts_every_nlp_legalqa_intent():
    llm = MagicMock()
    router = QueryRouter(llm)

    for intent in ("direct_answer", "retrieve", "reject", "cypher_query"):
        llm.complete.return_value = f'{{"intent": "{intent}"}}'
        assert router.route("câu hỏi") == intent


def test_router_strips_a_json_fence_and_uses_a_small_deterministic_request():
    llm = MagicMock()
    llm.complete.return_value = '```json\n{"intent": "retrieve"}\n```'

    assert QueryRouter(llm).route("vượt đèn đỏ phạt bao nhiêu?") == "retrieve"
    assert llm.complete.call_args.kwargs["temperature"] == 0
    assert llm.complete.call_args.kwargs["max_tokens"] == 64


def test_router_falls_back_to_retrieval_for_invalid_or_failed_llm_output():
    llm = MagicMock()
    llm.complete.return_value = '{"intent": "delete_database"}'
    router = QueryRouter(llm)

    assert router.route("câu hỏi") == "retrieve"
    llm.complete.side_effect = RuntimeError("provider unavailable")
    assert router.route("câu hỏi") == "retrieve"
