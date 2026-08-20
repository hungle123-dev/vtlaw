"""Evaluation harness for RAG retrieval.

Runs the full retrieval pipeline against a labeled QA dataset and reports
recall@k, precision@k, MRR, and latency statistics.

Usage:
    vtlaw eval run --dataset data/evaluation/qa/QA_NLP.csv --top-k 10

The dataset CSV must have columns: question, reference (comma-separated UIDs).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

from vtlaw.config import get_settings
from vtlaw.eval.metrics import (
    AggregateMetrics,
    RowMetrics,
    aggregate_metrics,
    compute_row_metrics,
)

log = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load QA dataset from CSV.

    Required columns: question, reference (comma-separated UIDs).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize column names
            normalized = {k.strip().lower(): v for k, v in row.items()}
            if "question" not in normalized or "reference" not in normalized:
                continue
            question = normalized["question"].strip()
            ref_str = normalized["reference"].strip()
            if not question or not ref_str:
                continue
            # Parse references (comma-separated)
            references = [r.strip() for r in ref_str.split(",") if r.strip()]
            rows.append({
                "question": question,
                "references": references,
                "answer": normalized.get("answer", ""),
            })

    log.info("Loaded %d rows from %s", len(rows), path)
    return rows


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_eval(
    dataset_path: str | Path,
    *,
    top_k: int = 10,
    strategy: str = "hybrid",
    output_path: str | Path | None = None,
    rerank: bool = False,
    decompose: bool = False,
    fetch_k: int | None = None,
    limit: int | None = None,
    as_of: date | None = None,
) -> AggregateMetrics:
    """Run evaluation on a dataset.

    Args:
        dataset_path: Path to CSV with question/reference columns.
        top_k: Number of results to score against the references.
        strategy: Retrieval strategy ("hybrid", "vector", "bm25").
        output_path: If provided, write detailed results to this JSON file.
        rerank: Cross-encoder rerank the candidates before scoring. Retrieval
            fetches ``fetch_k`` candidates and the reranker picks ``top_k``, so
            this can raise recall@k for k < fetch_k, not just reorder.
        fetch_k: Candidate budget for each retrieval leg. With ``rerank``,
            the same number of fused candidates is scored by the cross-encoder.
            Defaults to 4x ``top_k`` for reranking; otherwise the configured
            retrieval budget is used.
        decompose: Use LLM-generated query phrasings alongside the original
            question, recording the exact phrasings in the result artifact.
        limit: Score only the first N rows. For a quick check, not a report.
        as_of: Legal-effective date applied to every retrieval. ``None`` is
            captured once as the day the benchmark starts and written to output.

    Returns:
        AggregateMetrics with averaged scores.
    """
    from vtlaw.embed import Embedder
    from vtlaw.graph.client import GraphClient
    from vtlaw.retrieve.search import HybridRetriever

    rows = load_dataset(dataset_path)
    if limit:
        rows = rows[:limit]
    if not rows:
        log.error("No valid rows found in dataset")
        return AggregateMetrics()

    effective_as_of = as_of or date.today()

    settings = get_settings()
    graph_client = GraphClient(settings=settings)
    embedder = Embedder(settings=settings)
    retriever = HybridRetriever(graph_client, embedder, settings)
    decomposer = None
    if decompose:
        from vtlaw.generate.llm_client import LLMClient
        from vtlaw.retrieve.query_parser import QueryDecomposer

        decomposer = QueryDecomposer(LLMClient(settings))
    candidates_k = fetch_k or top_k * 4

    all_row_metrics: list[RowMetrics] = []
    latencies: list[float] = []
    detailed_results: list[dict] = []
    # A row whose retrieval raised is dropped from the averages. Counting them
    # is the difference between "0.87 over 94 rows" and "0.87 over the 90 rows
    # that did not crash" — the second is a different claim.
    skipped: list[dict[str, str]] = []

    for i, row in enumerate(rows):
        start = time.time()
        question = row["question"]
        references = row["references"]
        sub_queries = None
        if decomposer:
            generated = [sub["query"] for sub in decomposer.decompose(question)]
            sub_queries = list(dict.fromkeys([question, *generated]))

        log.info("[%d/%d] Q: %s", i + 1, len(rows), question[:80])

        try:
            search_kwargs: dict[str, Any] = {
                "strategy": strategy,
                "as_of": effective_as_of,
            }
            if sub_queries is not None:
                search_kwargs["sub_queries"] = sub_queries
            if rerank:
                result = retriever.search_and_rerank(
                    question,
                    k=top_k,
                    rerank_top=candidates_k,
                    fetch_k=candidates_k,
                    rerank_enabled=True,
                    **search_kwargs,
                )
            else:
                if fetch_k is not None:
                    search_kwargs["fetch_k"] = fetch_k
                result = retriever.search(
                    question,
                    k=top_k,
                    **search_kwargs,
                )
        except Exception as e:
            log.error("Search failed for row %d: %s", i, e)
            skipped.append({"question": question, "error": f"{type(e).__name__}: {e}"})
            continue
        latency = time.time() - start
        latencies.append(latency)

        retrieved_uids = [h.uid for h in result.hits]
        row_metrics = compute_row_metrics(retrieved_uids, references, top_k=top_k)
        all_row_metrics.append(row_metrics)

        detailed_results.append({
            "question": question,
            "references": references,
            "retrieved": retrieved_uids,
            "latency_s": round(latency, 3),
            "recall_at_k": row_metrics.recall_at_k,
            "mrr": row_metrics.mrr,
            "sub_queries": sub_queries,
        })

    graph_client.close()

    # Aggregate
    agg = aggregate_metrics(all_row_metrics)

    # Latency stats
    latencies.sort()
    n = len(latencies)
    if n > 0:
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)] if n > 1 else latencies[0]
        p99 = latencies[int(n * 0.99)] if n > 1 else latencies[0]
    else:
        p50 = p95 = p99 = 0.0

    # Print summary
    label = strategy
    if rerank:
        label += "+rerank"
    if decompose:
        label += "+decompose"
    print(f"\n{'=' * 60}")
    print(f"Evaluation Results ({label}, k={top_k})")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_path}")
    print(f"As of: {effective_as_of.isoformat()}")
    print(f"Rows: {agg.total_rows}")
    if skipped:
        print(f"Skipped (retrieval error): {len(skipped)} — excluded from every score")
    print()
    for k, recall in sorted(agg.recall_at_k.items()):
        print(f"  Recall@{k:2d}: {recall:.4f}")
    print()
    for k, precision in sorted(agg.precision_at_k.items()):
        print(f"  Precision@{k}: {precision:.4f}")
    print()
    print(f"  MRR: {agg.mrr:.4f}")
    print()
    print(f"  Latency p50: {p50:.3f}s")
    print(f"  Latency p95: {p95:.3f}s")
    print(f"  Latency p99: {p99:.3f}s")
    print(f"  Avg latency: {sum(latencies) / max(n, 1):.3f}s")
    print(f"{'=' * 60}")

    # Write detailed results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_output = {
            "strategy": label,
            "top_k": top_k,
            "fetch_k": candidates_k if rerank else (fetch_k or settings.fetch_k),
            "dataset": str(dataset_path),
            "dataset_sha256": _file_sha256(dataset_path),
            "as_of": effective_as_of.isoformat(),
            "configuration": {
                "embed_model": settings.embed_model,
                "embed_model_revision": settings.embed_model_revision,
                "rerank_enabled": rerank,
                "rerank_model": settings.rerank_model,
                "rerank_model_revision": settings.rerank_model_revision,
                "query_decomposition": decompose,
                "llm_model": settings.llm_model if decompose else None,
                "rrf_k": settings.rrf_k,
                "rrf_vector_weight": settings.rrf_vector_weight,
                "rrf_bm25_weight": settings.rrf_bm25_weight,
                "overfetch_factor": settings.overfetch_factor,
            },
            "total_rows": agg.total_rows,
            "dataset_rows": len(rows),
            "skipped_rows": skipped,
            "aggregate": {
                "recall_at_k": agg.recall_at_k,
                "precision_at_k": agg.precision_at_k,
                "mrr": agg.mrr,
            },
            "latency": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "avg": sum(latencies) / max(n, 1),
            },
            "rows": detailed_results,
        }
        output_path.write_text(
            json.dumps(full_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Detailed results written to %s", output_path)

    return agg
