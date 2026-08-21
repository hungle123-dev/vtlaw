"""Evaluation harness for RAG retrieval.

Runs the full retrieval pipeline against a labeled QA dataset and reports
recall@k, precision@k, MRR, and latency statistics.

Usage:
    vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 8

The dataset CSV must have columns: question, reference (comma-separated UIDs).
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
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


def _optional_file_sha256(path: str | Path) -> str | None:
    try:
        return _file_sha256(path)
    except OSError:
        return None


def _verify_release_provenance(
    report: dict[str, Any],
    dataset_path: str | Path,
    snapshot_manifest_path: str | Path = "data/snapshot/manifest.json",
) -> None:
    """Reject release evidence whose input hashes are absent or stale."""
    dataset_sha256 = _optional_file_sha256(dataset_path)
    snapshot_sha256 = _optional_file_sha256(snapshot_manifest_path)
    checks = (
        ("dataset_sha256", report.get("dataset_sha256"), dataset_sha256),
        (
            "inputs.dataset.sha256",
            report.get("inputs", {}).get("dataset", {}).get("sha256"),
            dataset_sha256,
        ),
        (
            "snapshot_manifest_sha256",
            report.get("provenance", {}).get("snapshot_manifest_sha256"),
            snapshot_sha256,
        ),
        (
            "inputs.snapshot_manifest.sha256",
            report.get("inputs", {}).get("snapshot_manifest", {}).get("sha256"),
            snapshot_sha256,
        ),
    )
    for field, actual, expected in checks:
        if expected is None or actual != expected:
            raise ValueError(f"release evidence {field} is missing or incorrect")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    """Return whether the evaluated source tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _label_currency(
    rows: list[dict[str, Any]], client: Any, as_of: date
) -> dict[str, int | str]:
    """Describe how many evaluated reference UIDs are superseded at ``as_of``."""
    from vtlaw.retrieve.heuristics import fetch_abolished_uids

    reference_uids = sorted({uid for row in rows for uid in row["references"]})
    try:
        superseded = fetch_abolished_uids(client, reference_uids, as_of=as_of)
    except Exception as exc:  # noqa: BLE001 - record an incomplete audit, not a false zero
        log.warning("label-currency audit unavailable: %s", exc)
        return {"status": "unavailable", "error": type(exc).__name__}

    return {
        "status": "available",
        "unique_reference_uids": len(reference_uids),
        "superseded_reference_uids": len(superseded),
        "rows_with_any_superseded_reference": sum(
            any(uid in superseded for uid in row["references"]) for row in rows
        ),
        "rows_with_only_superseded_references": sum(
            bool(row["references"])
            and all(uid in superseded for uid in row["references"])
            for row in rows
        ),
    }


def run_eval(
    dataset_path: str | Path,
    *,
    top_k: int = 8,
    strategy: str = "hybrid",
    output_path: str | Path | None = None,
    rerank: bool = False,
    decompose: bool = False,
    fetch_k: int | None = None,
    limit: int | None = None,
    as_of: date | None = None,
    command: str | None = None,
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
        as_of: Legal-effective date applied to every retrieval. ``None`` runs a
            label-retrieval benchmark without temporal filtering or amendment
            demotion because the supplied QA rows have no date of fact.

    Returns:
        AggregateMetrics with averaged scores.
    """
    from vtlaw.embed import Embedder
    from vtlaw.graph.client import GraphClient
    from vtlaw.retrieve.search import HybridRetriever

    rows = load_dataset(dataset_path)
    loaded_rows = len(rows)
    if limit:
        rows = rows[:limit]
    if not rows:
        log.error("No valid rows found in dataset")
        return AggregateMetrics()

    temporal_graph = as_of is not None

    settings = get_settings()
    graph_client = GraphClient(settings=settings)
    embedder = Embedder(settings=settings)
    retriever = HybridRetriever(graph_client, embedder, settings)
    decomposer = None
    if decompose:
        from vtlaw.generate.llm_client import LLMClient
        from vtlaw.retrieve.query_parser import QueryDecomposer

        decomposer = QueryDecomposer(LLMClient(settings))
    candidate_k = max(top_k, fetch_k or settings.fetch_k)
    effective_fetch_k = candidate_k
    effective_rerank_top = candidate_k

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
                "as_of": as_of,
                "temporal": temporal_graph,
            }
            if sub_queries is not None:
                search_kwargs["sub_queries"] = sub_queries
            search_kwargs["fetch_k"] = candidate_k
            result = retriever.search_and_rerank(
                question,
                k=top_k,
                rerank_top=effective_rerank_top,
                rerank_enabled=rerank,
                heuristic_rerank=temporal_graph,
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
            "precision_at_k": row_metrics.precision_at_k,
            "mrr": row_metrics.mrr,
            "sub_queries": sub_queries,
        })

    label_currency = (
        _label_currency(rows, graph_client, as_of)
        if temporal_graph
        else {
            "status": "not_applicable",
            "reason": "benchmark_without_as_of",
        }
    )
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
    if temporal_graph:
        print(f"As of: {as_of.isoformat()}")
    else:
        print("Temporal graph: disabled (benchmark has no as_of date)")
    print(f"Rows: {agg.total_rows}")
    if skipped:
        print(f"Skipped (retrieval error): {len(skipped)} — excluded from every score")
    if label_currency["status"] == "available":
        print(
            "Superseded reference UIDs: "
            f"{label_currency['superseded_reference_uids']}/"
            f"{label_currency['unique_reference_uids']}"
        )
    print()
    for k, recall in sorted(agg.recall_at_k.items()):
        print(f"  Recall@{k:2d}: {recall:.4f}")
    print()
    for k, precision in sorted(agg.precision_at_k.items()):
        print(f"  Precision@{k}: {precision:.4f}")
    print()
    print(f"  MRR@{top_k}: {agg.mrr:.4f}")
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
        dataset_sha256 = _file_sha256(dataset_path)
        snapshot_manifest_path = Path("data/snapshot/manifest.json")
        snapshot_manifest_sha256 = _optional_file_sha256(snapshot_manifest_path)
        full_output = {
            "strategy": label,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "fetch_k": effective_fetch_k,
            "rerank_top": effective_rerank_top,
            "context_k": top_k,
            "command": command,
            "dataset": str(dataset_path),
            "dataset_sha256": dataset_sha256,
            "inputs": {
                "dataset": {"path": str(dataset_path), "sha256": dataset_sha256},
                "snapshot_manifest": {
                    "path": str(snapshot_manifest_path),
                    "sha256": snapshot_manifest_sha256,
                },
            },
            "as_of": as_of.isoformat() if as_of else None,
            "temporal_graph": temporal_graph,
            "label_currency": label_currency,
            "provenance": {
                "snapshot_manifest_sha256": snapshot_manifest_sha256,
                "git_commit": _git_commit(),
                "git_dirty": _git_dirty(),
            },
            "selection": {
                "limit": limit,
                "method": "first_n" if limit else "all_valid_rows",
                "loaded_rows": loaded_rows,
                "selected_rows": len(rows),
            },
            "configuration": {
                "embed_model": settings.embed_model,
                "embed_model_revision": settings.embed_model_revision,
                "rerank_enabled": rerank,
                "heuristic_rerank": temporal_graph,
                "temporal_graph": temporal_graph,
                "rerank_top": effective_rerank_top,
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
        if "release-evidence" in output_path.stem:
            if not command:
                raise ValueError("release evidence command is missing")
            _verify_release_provenance(
                full_output, dataset_path, snapshot_manifest_path
            )
        output_path.write_text(
            json.dumps(full_output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Detailed results written to %s", output_path)

    return agg
