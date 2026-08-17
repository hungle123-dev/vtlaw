"""Evaluation harness for RAG system.

Provides metrics (Recall@k, Precision@k, MRR) and evaluation runner
for benchmarking retrieval quality against labeled QA datasets.

Usage:
    vtlaw eval run --dataset data/evaluation/qa/QA_NLP.csv --top-k 10
"""

from vtlaw.eval.metrics import (
    AggregateMetrics,
    RowMetrics,
    aggregate_metrics,
    compute_row_metrics,
    is_relevant,
    mrr,
    precision_at_k,
    recall_at_k,
)
from vtlaw.eval.runner import load_dataset, run_eval

__all__ = [
    "AggregateMetrics",
    "RowMetrics",
    "aggregate_metrics",
    "compute_row_metrics",
    "is_relevant",
    "load_dataset",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "run_eval",
]
