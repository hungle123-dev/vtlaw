"""Tests for amends importer and heuristic rerank.

Tests cover:
- AmendStats dataclass
- _resolve_uid: label/uid resolution for different granularity levels
- apply_heuristic_rerank: penalty application, recency bonus, reordering
- Penalty constants: abolished > replaced > no penalty
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from vtlaw.graph.amends import AmendStats, _resolve_uid
from vtlaw.retrieve.heuristics import (
    ABOLISHED_PENALTY,
    RECENCY_DECAY_PER_YEAR,
    RECENCY_INITIAL_BONUS,
    REPLACED_PENALTY,
    apply_heuristic_rerank,
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

    def test_recency_constants_are_positive(self):
        assert RECENCY_INITIAL_BONUS > 0
        assert RECENCY_DECAY_PER_YEAR > 0


# ---------------------------------------------------------------------------
# apply_heuristic_rerank
# ---------------------------------------------------------------------------


class TestApplyHeuristicRerank:
    def test_empty_hits_returns_empty(self):
        mock_client = MagicMock()
        result = apply_heuristic_rerank([], mock_client)
        assert result == []

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
        with (
            patch(
                "vtlaw.retrieve.heuristics.fetch_abolished_uids",
                return_value={"100/2019/NĐ-CP::article::11::clause::3": ["bãi bỏ"]},
            ),
            patch(
                "vtlaw.retrieve.heuristics.fetch_doc_effect_dates",
                return_value={
                    "100/2019/NĐ-CP": "2020-01-01",
                    "168/2024/NĐ-CP": "2025-01-01",
                },
            ),
        ):
            adjusted = apply_heuristic_rerank(
                hits, mock_client, as_of=date(2026, 8, 16)
            )

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
        with (
            patch(
                "vtlaw.retrieve.heuristics.fetch_abolished_uids",
                return_value={
                    "abolished": ["bãi bỏ"],
                    "replaced": ["thay thế"],
                },
            ),
            patch(
                "vtlaw.retrieve.heuristics.fetch_doc_effect_dates",
                return_value={},
            ),
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        # All have same original score; penalties determine order.
        scores = {h.uid: h.score for h in adjusted}
        assert scores["clean"] > scores["replaced"] > scores["abolished"]
        assert scores["abolished"] == 0.9 + ABOLISHED_PENALTY
        assert scores["replaced"] == 0.9 + REPLACED_PENALTY

    def test_recency_bonus_boosts_newer_documents(self):
        """A document that took effect recently should get a higher score
        than one with the same original score but an older effect date."""
        hits = [
            make_hit(uid="old", score=0.5, doc_identity="old-doc"),
            make_hit(uid="new", score=0.5, doc_identity="new-doc"),
        ]

        mock_client = MagicMock()
        with (
            patch(
                "vtlaw.retrieve.heuristics.fetch_abolished_uids",
                return_value={},
            ),
            patch(
                "vtlaw.retrieve.heuristics.fetch_doc_effect_dates",
                return_value={
                    "old-doc": "2020-01-01",
                    "new-doc": "2025-01-01",
                },
            ),
        ):
            adjusted = apply_heuristic_rerank(
                hits, mock_client, as_of=date(2026, 8, 16)
            )

        # New document should rank first.
        assert adjusted[0].uid == "new"
        assert adjusted[0].score > adjusted[1].score

    def test_recency_bonus_floors_at_zero(self):
        """Very old documents should get no recency bonus (floor at 0)."""
        hits = [make_hit(uid="very-old", score=0.5, doc_identity="old-doc")]

        mock_client = MagicMock()
        with (
            patch(
                "vtlaw.retrieve.heuristics.fetch_abolished_uids",
                return_value={},
            ),
            patch(
                "vtlaw.retrieve.heuristics.fetch_doc_effect_dates",
                return_value={"old-doc": "2010-01-01"},
            ),
        ):
            adjusted = apply_heuristic_rerank(
                hits, mock_client, as_of=date(2026, 8, 16)
            )

        # 16.5 years old → bonus = max(0, 2.0 - 0.3 * 16.5) = max(0, -2.95) = 0
        assert adjusted[0].score == 0.5  # no bonus, no penalty

    def test_no_penalty_for_clean_provisions(self):
        """A provision with no amendments should keep its original score
        (plus recency bonus if applicable)."""
        hits = [make_hit(uid="clean", score=0.7, doc_identity="clean-doc")]

        mock_client = MagicMock()
        with (
            patch(
                "vtlaw.retrieve.heuristics.fetch_abolished_uids",
                return_value={},
            ),
            patch(
                "vtlaw.retrieve.heuristics.fetch_doc_effect_dates",
                return_value={"clean-doc": "2026-01-01"},
            ),
        ):
            adjusted = apply_heuristic_rerank(
                hits, mock_client, as_of=date(2026, 8, 16)
            )

        # ~0.6 years old → bonus = max(0, 2.0 - 0.3 * 0.6) ≈ 1.82
        assert adjusted[0].score > 0.7  # original + bonus
        assert adjusted[0].score < 0.7 + RECENCY_INITIAL_BONUS

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
        with (
            patch("vtlaw.retrieve.heuristics.fetch_abolished_uids", return_value={}),
            patch("vtlaw.retrieve.heuristics.fetch_doc_effect_dates", return_value={}),
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        assert adjusted[0].uid == "test::1"
        assert adjusted[0].doc_identity == "doc1"
        assert adjusted[0].label == "Article"
        assert adjusted[0].content == "original content"

    def test_scores_are_sorted_descending(self):
        """Output must be sorted by adjusted score, highest first."""
        hits = [
            make_hit(uid="a", score=0.3),
            make_hit(uid="b", score=0.9),
            make_hit(uid="c", score=0.6),
        ]

        mock_client = MagicMock()
        with (
            patch("vtlaw.retrieve.heuristics.fetch_abolished_uids", return_value={}),
            patch("vtlaw.retrieve.heuristics.fetch_doc_effect_dates", return_value={}),
        ):
            adjusted = apply_heuristic_rerank(hits, mock_client)

        scores = [h.score for h in adjusted]
        assert scores == sorted(scores, reverse=True)
