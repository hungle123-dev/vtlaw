"""Tests for the evaluation module.

Tests cover:
- is_relevant: UID prefix matching
- recall_at_k: fraction of relevant found
- precision_at_k: fraction of top-k that are relevant
- mrr: mean reciprocal rank
- compute_row_metrics: all metrics for a single row
- aggregate_metrics: averaged metrics across rows
- load_dataset: CSV parsing with comma-separated references
"""

from __future__ import annotations

import pytest

from vtlaw.eval.metrics import (
    RowMetrics,
    aggregate_metrics,
    compute_row_metrics,
    is_relevant,
    mrr,
    precision_at_k,
    recall_at_k,
)
from vtlaw.eval.runner import load_dataset

# ---------------------------------------------------------------------------
# is_relevant
# ---------------------------------------------------------------------------


class TestIsRelevant:
    def test_exact_match(self):
        assert is_relevant("168/2024/NĐ-CP::article::6", "168/2024/NĐ-CP::article::6")

    def test_descendant_match(self):
        """Retrieving a Clause answers a reference to its parent Article."""
        assert is_relevant(
            "168/2024/NĐ-CP::article::6::clause::3::point::a",
            "168/2024/NĐ-CP::article::6",
        )

    def test_no_match_different_doc(self):
        assert not is_relevant(
            "100/2019/NĐ-CP::article::5",
            "168/2024/NĐ-CP::article::6",
        )

    def test_no_match_different_article(self):
        assert not is_relevant(
            "168/2024/NĐ-CP::article::7",
            "168/2024/NĐ-CP::article::6",
        )

    def test_sibling_sharing_digit_prefix_is_not_relevant(self):
        """Điều 12 is not Điều 1, though its UID starts with the same characters.

        This is the case a plain startswith gets wrong, and it inflated recall
        for every Article-level reference in the dataset.
        """
        assert not is_relevant(
            "168/2024/NĐ-CP::article::12",
            "168/2024/NĐ-CP::article::1",
        )

    def test_letter_suffixed_sibling_is_not_relevant(self):
        """Điều 4a is a distinct provision from Điều 4, not a child of it."""
        assert not is_relevant(
            "100/2019/NĐ-CP::article::4a",
            "100/2019/NĐ-CP::article::4",
        )

    def test_deep_sibling_sharing_digit_prefix(self):
        assert not is_relevant(
            "168/2024/NĐ-CP::article::6::clause::30",
            "168/2024/NĐ-CP::article::6::clause::3",
        )

    def test_descendant_of_letter_suffixed_still_matches(self):
        assert is_relevant(
            "100/2019/NĐ-CP::article::4a::clause::1",
            "100/2019/NĐ-CP::article::4a",
        )

    def test_empty_retrieved(self):
        assert not is_relevant("", "168/2024/NĐ-CP::article::6")

    def test_empty_reference_is_never_relevant(self):
        """A malformed dataset row must not score as a perfect hit.

        An empty string is a prefix of everything, so the old prefix test
        returned True and rewarded bad data.
        """
        assert not is_relevant("anything", "")


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_all_found(self):
        assert recall_at_k(3, 3) == 1.0

    def test_partial(self):
        assert recall_at_k(2, 4) == 0.5

    def test_none_found(self):
        assert recall_at_k(0, 3) == 0.0

    def test_zero_total(self):
        assert recall_at_k(0, 0) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(5, 5) == 1.0

    def test_partial(self):
        assert precision_at_k(2, 5) == 0.4

    def test_none_relevant(self):
        assert precision_at_k(0, 5) == 0.0

    def test_zero_k(self):
        assert precision_at_k(0, 0) == 0.0


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_item_relevant(self):
        uids = ["168/2024/NĐ-CP::article::6", "other"]
        refs = ["168/2024/NĐ-CP::article::6"]
        assert mrr(uids, refs) == 1.0

    def test_second_item_relevant(self):
        uids = ["other", "168/2024/NĐ-CP::article::6"]
        refs = ["168/2024/NĐ-CP::article::6"]
        assert mrr(uids, refs) == 0.5

    def test_no_relevant(self):
        uids = ["other1", "other2"]
        refs = ["168/2024/NĐ-CP::article::6"]
        assert mrr(uids, refs) == 0.0

    def test_multiple_references(self):
        uids = ["168/2024/NĐ-CP::article::6", "100/2019/NĐ-CP::article::5"]
        refs = ["168/2024/NĐ-CP::article::6", "100/2019/NĐ-CP::article::5"]
        assert mrr(uids, refs) == 1.0


# ---------------------------------------------------------------------------
# compute_row_metrics
# ---------------------------------------------------------------------------


class TestComputeRowMetrics:
    def test_perfect_retrieval(self):
        """All references found at rank 1."""
        uids = ["168/2024/NĐ-CP::article::6", "168/2024/NĐ-CP::article::7"]
        refs = ["168/2024/NĐ-CP::article::6"]
        metrics = compute_row_metrics(uids, refs)

        assert metrics.recall_at_k[1] == 1.0
        assert metrics.recall_at_k[5] == 1.0
        assert metrics.recall_at_k[10] == 1.0
        assert metrics.precision_at_k[1] == 1.0
        assert metrics.mrr == 1.0

    def test_partial_retrieval(self):
        uids = ["other1", "168/2024/NĐ-CP::article::6", "other2"]
        refs = ["168/2024/NĐ-CP::article::6"]
        metrics = compute_row_metrics(uids, refs)

        assert metrics.recall_at_k[1] == 0.0
        assert metrics.recall_at_k[3] == 1.0
        assert metrics.precision_at_k[3] == pytest.approx(1 / 3)
        assert metrics.mrr == 0.5

    def test_no_relevant_found(self):
        uids = ["other1", "other2", "other3"]
        refs = ["168/2024/NĐ-CP::article::6"]
        metrics = compute_row_metrics(uids, refs)

        assert metrics.recall_at_k[1] == 0.0
        assert metrics.recall_at_k[10] == 0.0
        assert metrics.mrr == 0.0

    def test_multiple_references(self):
        uids = [
            "168/2024/NĐ-CP::article::6",
            "168/2024/NĐ-CP::article::7",
            "other",
        ]
        refs = [
            "168/2024/NĐ-CP::article::6",
            "168/2024/NĐ-CP::article::7",
        ]
        metrics = compute_row_metrics(uids, refs)

        assert metrics.recall_at_k[1] == 0.5
        assert metrics.recall_at_k[3] == 1.0
        assert metrics.mrr == 1.0

    def test_top_k_bounds_the_reported_cutoffs(self):
        """A top-5 run must not report recall@10.

        It would equal recall@5 and read as a plateau in the results table when
        it is only the list being truncated. The recorded benchmark JSON carried
        recall@7 and recall@10 for a `--top-k 5` run for exactly this reason.
        """
        uids = ["168/2024/NĐ-CP::article::6"]
        refs = ["168/2024/NĐ-CP::article::6"]

        metrics = compute_row_metrics(uids, refs, top_k=5)

        assert sorted(metrics.recall_at_k) == [1, 3, 5]
        assert sorted(metrics.precision_at_k) == [1, 3]

    def test_aggregate_omits_a_cutoff_no_row_measured(self):
        """Absent is not zero: averaging a missing k as 0.0 invents a regression."""
        row = compute_row_metrics(
            ["168/2024/NĐ-CP::article::6"], ["168/2024/NĐ-CP::article::6"], top_k=5
        )

        agg = aggregate_metrics([row])

        assert 10 not in agg.recall_at_k
        assert agg.recall_at_k[5] == 1.0

    def test_a_cutoff_only_some_rows_measured_averages_over_those_rows(self):
        """Mixed depths must not score the shallower rows as 0.0 at the deeper k."""
        shallow = compute_row_metrics(
            ["168/2024/NĐ-CP::article::6"], ["168/2024/NĐ-CP::article::6"], top_k=5
        )
        deep = compute_row_metrics(
            ["168/2024/NĐ-CP::article::6"], ["168/2024/NĐ-CP::article::6"], top_k=10
        )

        agg = aggregate_metrics([shallow, deep])

        # Only `deep` measured recall@10, and it was perfect. Dividing by both
        # rows would report 0.5 and read as a regression that never happened.
        assert agg.recall_at_k[10] == 1.0
        assert agg.recall_at_k[5] == 1.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_empty_list(self):
        agg = aggregate_metrics([])
        assert agg.total_rows == 0
        assert agg.mrr == 0.0

    def test_single_row(self):
        row = RowMetrics()
        row.recall_at_k = {1: 1.0, 5: 1.0, 10: 1.0}
        row.precision_at_k = {1: 1.0, 3: 0.5}
        row.mrr = 1.0

        agg = aggregate_metrics([row])
        assert agg.total_rows == 1
        assert agg.recall_at_k[1] == 1.0
        assert agg.mrr == 1.0

    def test_multiple_rows_average(self):
        row1 = RowMetrics()
        row1.recall_at_k = {1: 1.0, 5: 1.0, 10: 1.0}
        row1.mrr = 1.0

        row2 = RowMetrics()
        row2.recall_at_k = {1: 0.0, 5: 0.5, 10: 1.0}
        row2.mrr = 0.5

        agg = aggregate_metrics([row1, row2])
        assert agg.total_rows == 2
        assert agg.recall_at_k[1] == 0.5
        assert agg.mrr == 0.75


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


class TestLoadDataset:
    def test_load_valid_csv(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "id,question,answer,reference\n"
            '1,"Vượt đèn đỏ phạt bao nhiêu?","Answer","168/2024/NĐ-CP::article::7"\n'
            '2,"Mũ bảo hiểm phạt gì?","Answer2",'
            '"168/2024/NĐ-CP::article::12,100/2019/NĐ-CP::article::11"\n',
            encoding="utf-8",
        )

        rows = load_dataset(csv_path)

        assert len(rows) == 2
        assert rows[0]["question"] == "Vượt đèn đỏ phạt bao nhiêu?"
        assert len(rows[0]["references"]) == 1
        assert len(rows[1]["references"]) == 2

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("nonexistent.csv")

    def test_load_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("id,question,answer,reference\n", encoding="utf-8")

        rows = load_dataset(csv_path)
        assert rows == []

    def test_load_missing_columns(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("id,name\n1,test\n", encoding="utf-8")

        rows = load_dataset(csv_path)
        assert rows == []

    def test_load_strips_whitespace(self, tmp_path):
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            'id,question,answer,reference\n'
            '1,Question,Answer,"  168/2024/NĐ-CP::article::6  "\n',
            encoding="utf-8",
        )

        rows = load_dataset(csv_path)
        assert len(rows) == 1
        assert rows[0]["question"] == "Question"
        assert rows[0]["references"] == ["168/2024/NĐ-CP::article::6"]
