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

    def test_prefix_match(self):
        """Retrieving a Clause is relevant to an Article reference."""
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

    def test_empty_retrieved(self):
        assert not is_relevant("", "168/2024/NĐ-CP::article::6")

    def test_empty_reference(self):
        # Empty string is a prefix of everything, but that's a data error
        # not a code error. The function returns True, which is correct
        # behaviour for prefix matching.
        assert is_relevant("anything", "")


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
            '2,"Mũ bảo hiểm phạt gì?","Answer2","168/2024/NĐ-CP::article::12,100/2019/NĐ-CP::article::11"\n',
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
