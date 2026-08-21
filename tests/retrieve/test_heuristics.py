"""Tests for amends importer and heuristic rerank.

Tests cover:
- AmendStats dataclass
- _resolve_uid: label/uid resolution for different granularity levels
- apply_heuristic_rerank: demotion of superseded provisions, reordering
- Penalty constants: abolished > replaced > no penalty
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from vtlaw.graph.amends import AmendStats, _resolve_uid
from vtlaw.retrieve.heuristics import (
    ABOLISHED_PENALTY,
    REPLACED_PENALTY,
    apply_heuristic_rerank,
    fetch_abolished_uids,
)
from vtlaw.retrieve.search import Hit


def make_hit(
    uid: str = "test::1",
    score: float = 0.5,
    doc_identity: str = "test",
    label: str = "Clause",
    content: str = "content",
) -> Hit:
    return Hit(
        uid=uid,
        score=score,
        doc_identity=doc_identity,
        label=label,  # type: ignore[arg-type]
        content=content,
    )


# ---------------------------------------------------------------------------
# AmendStats
# ---------------------------------------------------------------------------


class TestAmendStats:
    def test_empty_stats(self):
        stats = AmendStats()
        assert stats.imported == 0
        assert stats.skipped == 0
        assert stats.failed == []
        assert "imported=0" in stats.summary()

    def test_with_failures(self):
        stats = AmendStats(imported=10, skipped=2, failed=["a->b", "c->d"])
        assert len(stats.failed) == 2
        assert "failed=2" in stats.summary()


# ---------------------------------------------------------------------------
# _resolve_uid
# ---------------------------------------------------------------------------


class TestResolveUid:
    def test_resolves_point(self):
        label, uid = _resolve_uid("168/2024/NĐ-CP", "6", "3", "a")
        assert label == "Point"
        assert uid == "168/2024/NĐ-CP::article::6::clause::3::point::a"

    def test_resolves_clause(self):
        label, uid = _resolve_uid("168/2024/NĐ-CP", "6", "3", None)
        assert label == "Clause"
        assert uid == "168/2024/NĐ-CP::article::6::clause::3"

    def test_resolves_article(self):
        label, uid = _resolve_uid("168/2024/NĐ-CP", "6", None, None)
        assert label == "Article"
        assert uid == "168/2024/NĐ-CP::article::6"

    def test_resolves_document_when_no_article(self):
        label, uid = _resolve_uid("168/2024/NĐ-CP", None, None, None)
        assert label == "Document"
        assert uid == "168/2024/NĐ-CP"

    def test_point_requires_article_and_clause(self):
        """If point is given but article is missing, falls back to document."""
        label, uid = _resolve_uid("doc", None, None, "a")
        assert label == "Document"


# ---------------------------------------------------------------------------
# Penalty constants
# ---------------------------------------------------------------------------


class TestPenaltyConstants:
    def test_abolished_penalty_is_larger_than_replaced(self):
        """A bãi bỏ provision should be penalised more than a thay thế one."""
        assert abs(ABOLISHED_PENALTY) > abs(REPLACED_PENALTY)

    def test_penalties_are_negative(self):
        assert ABOLISHED_PENALTY < 0
        assert REPLACED_PENALTY < 0


# ---------------------------------------------------------------------------
# apply_heuristic_rerank
# ---------------------------------------------------------------------------


class TestApplyHeuristicRerank:
    def test_empty_hits_returns_empty(self):
        mock_client = MagicMock()
        result = apply_heuristic_rerank([], mock_client)
        assert result == []


class TestTemporalAmendmentLookup:
    def test_uses_as_of_to_bound_amendment_effect(self):
        session = MagicMock()
        session.run.return_value.data.return_value = []
        client = MagicMock()
        client.session.return_value.__enter__.return_value = session

        fetch_abolished_uids(
            client,
            ["36/2024/QH15::article::6::clause::3"],
            as_of=date(2025, 1, 1),
        )

        assert session.run.call_args.kwargs["as_of"] == "2025-01-01"

    @pytest.mark.parametrize("amend_type", ("sửa đổi", "sửa đổi, bổ sung"))
    def test_modifying_provision_is_demoted_only_after_source_effect_date(
        self, amend_type: str
    ):
        old_uid = "old-doc::article::1"
        hits = [
            make_hit(uid=old_uid, score=0.9),
            make_hit(uid="current-doc::article::1", score=0.85),
        ]
        session = MagicMock()

        def lookup(query: str, **params: str) -> MagicMock:
            assert f"'{amend_type}'" in query
            assert "source_document.effect_date <= date($as_of)" in query
            rows = (
                [{"uid": old_uid, "types": [amend_type]}]
                if params["as_of"] >= "2025-01-01"
                else []
            )
            return MagicMock(data=lambda: rows)

        session.run.side_effect = lookup
        client = MagicMock()
        client.session.return_value.__enter__.return_value = session

        before_effect = apply_heuristic_rerank(hits, client, as_of=date(2024, 12, 31))
        after_effect = apply_heuristic_rerank(hits, client, as_of=date(2025, 1, 1))

        assert [hit.uid for hit in before_effect] == [old_uid, "current-doc::article::1"]
        assert [hit.uid for hit in after_effect] == ["current-doc::article::1", old_uid]

    def test_supplement_only_keeps_original_provision_rank(self):
        hits = [
            make_hit(uid="original", score=0.9),
            make_hit(uid="other", score=0.85),
        ]

        with patch(
            "vtlaw.retrieve.heuristics.fetch_abolished_uids",
            return_value={"original": ["bổ sung"]},
        ):
            adjusted = apply_heuristic_rerank(hits, MagicMock())

        assert [hit.uid for hit in adjusted] == ["original", "other"]

    def test_abolished_provision_sinks_below(self):
        """A provision marked bãi bỏ should be penalised and sink below
        a still-in-force provision with a lower original score."""
        hits = [
            make_hit(
                uid="100/2019/NĐ-CP::article::11::clause::3",
                score=0.9,
                doc_identity="100/2019/NĐ-CP",
            ),
            make_hit(
                uid="168/2024/NĐ-CP::article::12::clause::5::point::b",
                score=0.85,
                doc_identity="168/2024/NĐ-CP",
            ),
        ]

        mock_client = MagicMock()
        with patch(
            "vtlaw.retrieve.heuristics.fetch_abolished_uids",
            return_value={"100/2019/NĐ-CP::article::11::clause::3": ["bãi bỏ"]},
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        # The abolished provision (originally rank 1) should now be last.
        assert adjusted[-1].uid == "100/2019/NĐ-CP::article::11::clause::3"
        assert adjusted[0].uid == "168/2024/NĐ-CP::article::12::clause::5::point::b"

    def test_replaced_provision_penalised_less_than_abolished(self):
        """thay thế should penalise less than bãi bỏ."""
        hits = [
            make_hit(uid="abolished", score=0.9, doc_identity="old-doc"),
            make_hit(uid="replaced", score=0.9, doc_identity="mid-doc"),
            make_hit(uid="clean", score=0.9, doc_identity="new-doc"),
        ]

        mock_client = MagicMock()
        with patch(
            "vtlaw.retrieve.heuristics.fetch_abolished_uids",
            return_value={
                "abolished": ["bãi bỏ"],
                "replaced": ["thay thế"],
            },
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        # All have same original score; penalties determine order.
        scores = {h.uid: h.score for h in adjusted}
        assert scores["clean"] > scores["replaced"] > scores["abolished"]
        assert scores["abolished"] == 0.9 + ABOLISHED_PENALTY
        assert scores["replaced"] == 0.9 + REPLACED_PENALTY

    def test_preserves_hit_metadata(self):
        """All Hit fields (uid, doc_identity, label, content) must be preserved."""
        hits = [
            make_hit(
                uid="test::1",
                score=0.5,
                doc_identity="doc1",
                label="Article",
                content="original content",
            )
        ]

        mock_client = MagicMock()
        with patch("vtlaw.retrieve.heuristics.fetch_abolished_uids", return_value={}):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        assert adjusted[0].uid == "test::1"
        assert adjusted[0].doc_identity == "doc1"
        assert adjusted[0].label == "Article"
        assert adjusted[0].content == "original content"

    def test_returns_input_untouched_when_nothing_is_superseded(self):
        """No AMENDS edge among the hits means no reordering to do.

        The list comes in already sorted by retrieval score; re-sorting it would
        be a no-op at best, so the function returns early instead.
        """
        hits = [
            make_hit(uid="b", score=0.9),
            make_hit(uid="c", score=0.6),
            make_hit(uid="a", score=0.3),
        ]

        mock_client = MagicMock()
        with patch(
            "vtlaw.retrieve.heuristics.fetch_abolished_uids", return_value={}
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        assert [h.uid for h in adjusted] == ["b", "c", "a"]
        assert [h.score for h in adjusted] == [0.9, 0.6, 0.3]

    def test_scores_are_sorted_descending(self):
        """When something IS superseded, the output must be re-sorted."""
        hits = [
            make_hit(uid="a", score=0.3),
            make_hit(uid="b", score=0.9),
            make_hit(uid="c", score=0.6),
        ]

        mock_client = MagicMock()
        with patch(
            "vtlaw.retrieve.heuristics.fetch_abolished_uids",
            return_value={"b": ["bãi bỏ"]},
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        scores = [h.score for h in adjusted]
        assert scores == sorted(scores, reverse=True)
        assert adjusted[-1].uid == "b"  # the superseded one sinks

    def test_demotion_scales_with_the_score_span(self):
        """The penalty is a fraction of the list's span, not an absolute value.

        Absolute constants were the original bug: -5.0 against an RRF list whose
        span is ~0.3 does not demote a hit, it discards the ranking. The same
        constant must behave the same way on either scale.
        """
        rrf_like = [
            make_hit(uid="superseded", score=0.36),
            make_hit(uid="in_force", score=0.08),
        ]
        cross_encoder_like = [
            make_hit(uid="superseded", score=9.9),
            make_hit(uid="in_force", score=1.2),
        ]

        mock_client = MagicMock()
        with patch(
            "vtlaw.retrieve.heuristics.fetch_abolished_uids",
            return_value={"superseded": ["bãi bỏ"]},
        ):
            small = apply_heuristic_rerank(rrf_like, mock_client)
            large = apply_heuristic_rerank(cross_encoder_like, mock_client)

        assert [h.uid for h in small] == ["in_force", "superseded"]
        assert [h.uid for h in large] == ["in_force", "superseded"]
