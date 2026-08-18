"""Tests for AnswerGenerator's query-decomposition wiring.

The decomposer existed with tests and no callers for the whole project; these
cover the seam where it now attaches. The measured reason for the shape of that
seam — keeping the original phrasing as its own retrieval leg — is asserted here
so it cannot be quietly dropped.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vtlaw.config import Settings
from vtlaw.generate.answer import AnswerGenerator
from vtlaw.retrieve.search import Hit

HIT = Hit(
    uid="100/2019/NĐ-CP::article::6::clause::3::point::i",
    score=0.9,
    doc_identity="100/2019/NĐ-CP",
    label="Point",
    content="Không đội mũ bảo hiểm",
    title=None,
)


def _settings(**overrides):
    return Settings(neo4j_password="x" * 8, llm_api_key="test", **overrides)


@pytest.fixture
def generator_parts(monkeypatch):
    """An AnswerGenerator with retrieval, decomposition and the LLM all stubbed."""

    def build(**settings_overrides):
        llm = MagicMock()
        llm.complete.return_value = "Phạt tiền từ 400.000 đến 600.000 đồng."
        gen = AnswerGenerator(
            MagicMock(), MagicMock(), llm, _settings(**settings_overrides)
        )

        retriever = MagicMock()
        retriever.search_and_rerank.return_value = MagicMock(
            hits=[HIT], retrieval_score="hybrid", reranked=True
        )
        gen._retriever = retriever

        decomposer = MagicMock()
        decomposer.decompose.return_value = [
            {"query": "người điều khiển xe mô tô không đội mũ bảo hiểm"},
            {"query": "mức xử phạt vi phạm hành chính"},
        ]
        gen._decomposer = decomposer

        monkeypatch.setattr(
            "vtlaw.generate.answer.build_full_context", lambda *a, **k: {}
        )
        return gen, retriever, decomposer

    return build


class TestDecomposeWiring:
    def test_passes_sub_queries_to_retrieval(self, generator_parts):
        gen, retriever, _ = generator_parts(decompose_queries=True)

        gen.answer("không đội mũ bảo hiểm phạt bao nhiêu")

        assert retriever.search_and_rerank.call_args.kwargs["sub_queries"] is not None

    def test_keeps_the_original_phrasing_as_its_own_leg(self, generator_parts):
        """Measured: sub-queries alone lose at k=1 (0.332 vs 0.359); with the
        original included, recall@1 rises to 0.402. Dropping it costs precision
        at the top of the list."""
        gen, retriever, _ = generator_parts(decompose_queries=True)
        question = "không đội mũ bảo hiểm phạt bao nhiêu"

        gen.answer(question)

        subs = retriever.search_and_rerank.call_args.kwargs["sub_queries"]
        assert subs[0] == question
        assert len(subs) == 3  # original + two sub-queries

    def test_disabled_by_flag(self, generator_parts):
        gen, retriever, decomposer = generator_parts()

        gen.answer("q", decompose=False)

        assert retriever.search_and_rerank.call_args.kwargs["sub_queries"] is None
        decomposer.decompose.assert_not_called()

    def test_disabled_by_settings(self, generator_parts):
        gen, retriever, decomposer = generator_parts(decompose_queries=False)

        gen.answer("q")

        assert retriever.search_and_rerank.call_args.kwargs["sub_queries"] is None
        decomposer.decompose.assert_not_called()

    def test_explicit_flag_overrides_settings(self, generator_parts):
        gen, retriever, decomposer = generator_parts(decompose_queries=False)

        gen.answer("q", decompose=True)

        assert retriever.search_and_rerank.call_args.kwargs["sub_queries"] is not None
        decomposer.decompose.assert_called_once()

    def test_off_by_default(self, generator_parts):
        gen, _, decomposer = generator_parts()

        gen.answer("q")

        decomposer.decompose.assert_not_called()

    def test_answer_still_generated(self, generator_parts):
        """Decomposition must not disturb the generation path."""
        gen, _, _ = generator_parts()

        answer = gen.answer("q")

        assert answer.text == "Phạt tiền từ 400.000 đến 600.000 đồng."
        assert answer.sources == [HIT]
