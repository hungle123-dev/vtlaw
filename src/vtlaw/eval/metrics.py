"""Evaluation metrics for RAG retrieval.

Metrics computed per-row then aggregated:
    - Recall@k: fraction of relevant references found in top-k
    - Precision@k: fraction of top-k that are relevant
    - MRR: mean reciprocal rank (1/rank of first relevant item)

Relevance is determined by UID prefix matching — a retrieved UID is relevant
if it starts with the reference UID (so retrieving a Clause is relevant to
an Article reference if the Clause is a descendant).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RowMetrics:
    """Metrics for a single QA row."""
    recall_at_k: dict[int, float] = field(default_factory=dict)  # k -> recall
    precision_at_k: dict[int, float] = field(default_factory=dict)  # k -> precision
    mrr: float = 0.0


def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if retrieved_uid shares a prefix with reference.

    A retrieved UID is relevant if it starts with the reference UID,
    allowing hierarchical matching (e.g., retrieving a Clause when the
    reference is an Article).
    """
    return retrieved_uid.startswith(reference)


def recall_at_k(relevant_in_top_k: int, total_relevant: int) -> float:
    """Fraction of relevant references found in top-k."""
    if total_relevant == 0:
        return 0.0
    return relevant_in_top_k / total_relevant


def precision_at_k(relevant_in_top_k: int, k: int) -> float:
    """Fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    return relevant_in_top_k / k


def mrr(retrieved_uids: list[str], references: list[str]) -> float:
    """Mean Reciprocal Rank: 1 / rank of first relevant item (0 if none)."""
    for i, uid in enumerate(retrieved_uids, start=1):
        for ref in references:
            if is_relevant(uid, ref):
                return 1.0 / i
    return 0.0


def compute_row_metrics(retrieved_uids: list[str], references: list[str]) -> RowMetrics:
    """Compute all metrics for a single row."""
    metrics = RowMetrics()
    total_relevant = len(references)

    for k in [1, 3, 5, 7, 10]:
        top_k = retrieved_uids[:k]
        # Count unique references covered — a reference is "found" once
        found_refs = {ref for uid in top_k for ref in references if is_relevant(uid, ref)}
        rel_in_k = len(found_refs)
        metrics.recall_at_k[k] = recall_at_k(rel_in_k, total_relevant)
        if k in [1, 3]:
            metrics.precision_at_k[k] = precision_at_k(rel_in_k, k)

    metrics.mrr = mrr(retrieved_uids, references)

    return metrics


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all rows."""
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    total_rows: int = 0


def aggregate_metrics(row_metrics: list[RowMetrics]) -> AggregateMetrics:
    """Compute aggregate metrics across all rows."""
    agg = AggregateMetrics()
    agg.total_rows = len(row_metrics)
    if not row_metrics:
        return agg

    # Average recall/precision at each k
    for k in [1, 3, 5, 7, 10]:
        agg.recall_at_k[k] = sum(r.recall_at_k.get(k, 0.0) for r in row_metrics) / len(row_metrics)
        if k in [1, 3]:
            agg.precision_at_k[k] = sum(r.precision_at_k.get(k, 0.0) for r in row_metrics) / len(row_metrics)

    agg.mrr = sum(r.mrr for r in row_metrics) / len(row_metrics)

    return agg
