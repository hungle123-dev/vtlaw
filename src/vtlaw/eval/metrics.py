"""Evaluation metrics for RAG retrieval.

Metrics computed per-row then aggregated:
    - Recall@k: fraction of relevant references found in top-k
    - Precision@k: fraction of top-k that are relevant
    - MRR: mean reciprocal rank (1/rank of first relevant item)

A retrieved UID counts as relevant if it is the reference itself or a
descendant of it in the Điều / Khoản / Điểm hierarchy. See :func:`is_relevant`
for why that test is not a plain string prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# UID segments are joined by this. Descendant tests must land on a boundary:
# "article::1" is a prefix of "article::12" as a string but not as a provision.
UID_SEP = "::"


@dataclass
class RowMetrics:
    """Metrics for a single QA row."""
    recall_at_k: dict[int, float] = field(default_factory=dict)  # k -> recall
    precision_at_k: dict[int, float] = field(default_factory=dict)  # k -> precision
    mrr: float = 0.0


def is_relevant(retrieved_uid: str, reference: str) -> bool:
    """Return True if ``retrieved_uid`` is ``reference`` or a descendant of it.

    Retrieving a Clause answers a reference to its parent Article, so
    descendants count. A plain ``startswith`` almost expresses that but
    over-counts on two cases measured in this corpus:

    * Sibling numbers that share a digit prefix. ``article::1`` startswith-matches
      ``article::12``, ``article::15``, ``article::1a`` — different provisions.
    * An empty reference, which is a prefix of everything and would score a
      malformed dataset row as a perfect hit.

    Requiring the next character to be the ``::`` separator fixes both.
    """
    if not reference or not retrieved_uid:
        return False
    if retrieved_uid == reference:
        return True
    return retrieved_uid.startswith(reference + UID_SEP)


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
    n = len(row_metrics)
    for k in [1, 3, 5, 7, 10]:
        agg.recall_at_k[k] = sum(r.recall_at_k.get(k, 0.0) for r in row_metrics) / n
        if k in [1, 3]:
            agg.precision_at_k[k] = (
                sum(r.precision_at_k.get(k, 0.0) for r in row_metrics) / n
            )

    agg.mrr = sum(r.mrr for r in row_metrics) / n

    return agg
