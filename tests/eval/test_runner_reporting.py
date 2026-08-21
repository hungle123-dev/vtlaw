"""Evaluation runs must record the exact conditions behind a score."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from vtlaw.cli import build_parser, cmd_eval_run
from vtlaw.config import Settings
from vtlaw.eval.runner import run_eval
from vtlaw.retrieve.search import Hit, SearchResult


def test_eval_baseline_uses_served_amends_ranking(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "qa.csv"
    dataset.write_text(
        "question,reference\n"
        "Mức phạt?,168/2024/NĐ-CP::article::6\n",
        encoding="utf-8",
    )
    received: dict = {}

    class FakeGraphClient:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    class FakeEmbedder:
        def __init__(self, **_kwargs):
            pass

    class FakeRetriever:
        def __init__(self, *_args):
            pass

        def search_and_rerank(self, _question, **kwargs):
            received.update(kwargs)
            return SearchResult(
                query="Mức phạt?",
                strategy=kwargs["strategy"],
                hits=[
                    Hit(
                        uid="168/2024/NĐ-CP::article::6",
                        score=1.0,
                        doc_identity="168/2024/NĐ-CP",
                        label="Article",
                        content="Nội dung",
                    )
                ],
            )

    import vtlaw.embed
    import vtlaw.eval.runner
    import vtlaw.graph.client
    import vtlaw.retrieve.search

    settings = Settings(neo4j_password="test")
    monkeypatch.setattr(vtlaw.eval.runner, "get_settings", lambda: settings)
    monkeypatch.setattr(vtlaw.embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(vtlaw.graph.client, "GraphClient", FakeGraphClient)
    monkeypatch.setattr(vtlaw.retrieve.search, "HybridRetriever", FakeRetriever)

    run_eval(dataset, as_of=date(2025, 1, 1))

    report = capsys.readouterr().out
    assert received["k"] == 8
    assert received["as_of"] == date(2025, 1, 1)
    assert received["rerank_top"] == settings.rerank_top
    assert received["rerank_enabled"] is False
    assert received["heuristic_rerank"] is True
    assert received["fetch_k"] == 30
    assert "Recall@ 8" in report
    assert "Precision@8" in report
    assert "MRR@8" in report


def test_eval_records_dataset_config_and_as_of(tmp_path, monkeypatch):
    dataset = tmp_path / "qa.csv"
    dataset.write_text(
        "question,reference\n"
        "Mức phạt?,168/2024/NĐ-CP::article::6\n"
        "Đèn đỏ?,168/2024/NĐ-CP::article::6\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    seen: list[date | None] = []

    class FakeGraphClient:
        closed = False

        def __init__(self, **_kwargs):
            pass

        def close(self):
            self.closed = True

    graph = FakeGraphClient()

    class FakeEmbedder:
        def __init__(self, **_kwargs):
            pass

    class FakeRetriever:
        def __init__(self, *_args):
            pass

        def search_and_rerank(self, _question, *, k, strategy, as_of=None, **_kwargs):
            seen.append(as_of)
            return SearchResult(
                query="Mức phạt?",
                strategy=strategy,
                hits=[
                    Hit(
                        uid="168/2024/NĐ-CP::article::6",
                        score=1.0,
                        doc_identity="168/2024/NĐ-CP",
                        label="Article",
                        content="Nội dung",
                    )
                ][:k],
            )

    import vtlaw.embed
    import vtlaw.eval.runner
    import vtlaw.graph.client
    import vtlaw.retrieve.search

    settings = Settings(
        neo4j_password="test",
        embed_model_revision="embed-revision",
        rerank_model_revision="rerank-revision",
    )
    monkeypatch.setattr(vtlaw.eval.runner, "get_settings", lambda: settings)
    monkeypatch.setattr(vtlaw.embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(vtlaw.graph.client, "GraphClient", lambda **_kwargs: graph)
    monkeypatch.setattr(vtlaw.retrieve.search, "HybridRetriever", FakeRetriever)
    manifest = tmp_path / "data" / "snapshot" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"manifest_version": 1}', encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vtlaw.eval.runner, "_git_commit", lambda: "test-commit", raising=False)
    monkeypatch.setattr(vtlaw.eval.runner, "_git_dirty", lambda: None, raising=False)

    command = "python -m vtlaw eval run qa.csv --top-k 1 --output result.json"
    run_eval(
        dataset,
        top_k=1,
        limit=1,
        output_path=output,
        as_of=date(2025, 1, 1),
        command=command,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert seen == [date(2025, 1, 1)]
    assert graph.closed is True
    assert report["as_of"] == "2025-01-01"
    assert report["dataset_sha256"]
    assert report["command"] == command
    assert report["candidate_k"] == settings.fetch_k
    assert report["context_k"] == 1
    assert report["inputs"]["dataset"]["sha256"] == report["dataset_sha256"]
    assert report["inputs"]["snapshot_manifest"]["sha256"] == manifest_sha256
    assert report["configuration"]["embed_model_revision"] == "embed-revision"
    # Every setting that changes the result must be in the artifact, or the
    # artifact cannot reproduce the run it claims to describe.
    assert report["configuration"]["overfetch_factor"] == settings.overfetch_factor
    assert report["configuration"]["heuristic_rerank"] is True
    assert report["configuration"]["rerank_top"] == settings.rerank_top
    assert report["fetch_k"] == settings.fetch_k
    assert report["provenance"] == {
        "snapshot_manifest_sha256": manifest_sha256,
        "git_commit": "test-commit",
        "git_dirty": None,
    }
    assert report["selection"] == {
        "limit": 1,
        "method": "first_n",
        "loaded_rows": 2,
        "selected_rows": 1,
    }

    manifest.unlink()
    missing_output = tmp_path / "missing-local-provenance.json"
    monkeypatch.setattr(vtlaw.eval.runner, "_git_commit", lambda: None)
    run_eval(dataset, top_k=1, limit=1, output_path=missing_output, as_of=date(2025, 1, 1))

    missing_report = json.loads(missing_output.read_text(encoding="utf-8"))
    assert missing_report["provenance"] == {
        "snapshot_manifest_sha256": None,
        "git_commit": None,
        "git_dirty": None,
    }


def test_optional_provenance_handles_missing_local_sources(tmp_path, monkeypatch):
    import vtlaw.eval.runner

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vtlaw.eval.runner.subprocess, "run", no_git, raising=False)

    assert vtlaw.eval.runner._optional_file_sha256("data/snapshot/manifest.json") is None
    assert vtlaw.eval.runner._git_commit() is None


def test_git_dirty_reports_clean_dirty_or_unknown(monkeypatch):
    import vtlaw.eval.runner

    monkeypatch.setattr(
        vtlaw.eval.runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=" M src/vtlaw/eval/runner.py\n"),
    )
    assert vtlaw.eval.runner._git_dirty() is True

    monkeypatch.setattr(
        vtlaw.eval.runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=""),
    )
    assert vtlaw.eval.runner._git_dirty() is False

    def no_git(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(vtlaw.eval.runner.subprocess, "run", no_git)
    assert vtlaw.eval.runner._git_dirty() is None


def test_label_currency_counts_superseded_references(monkeypatch):
    import vtlaw.eval.runner
    import vtlaw.retrieve.heuristics

    def fake_fetch(_client, uids, *, as_of):
        assert uids == ["current", "old-a", "old-b"]
        assert as_of == date(2026, 8, 20)
        return {"old-a": ["bãi bỏ"], "old-b": ["sửa đổi"]}

    monkeypatch.setattr(
        vtlaw.retrieve.heuristics, "fetch_abolished_uids", fake_fetch
    )

    assert vtlaw.eval.runner._label_currency(
        [
            {"references": ["old-a", "current"]},
            {"references": ["old-b"]},
            {"references": ["current"]},
        ],
        object(),
        date(2026, 8, 20),
    ) == {
        "status": "available",
        "unique_reference_uids": 3,
        "superseded_reference_uids": 2,
        "rows_with_any_superseded_reference": 2,
        "rows_with_only_superseded_references": 1,
    }


def test_eval_cli_parses_as_of_date():
    args = build_parser().parse_args(
        ["eval", "run", "data/evaluation/qa/QA_NLP.csv", "--as-of", "2025-01-01"]
    )

    assert args.as_of == date(2025, 1, 1)


def test_eval_cli_defaults_to_served_k8():
    args = build_parser().parse_args(
        ["eval", "run", "data/evaluation/qa/QA_NLP.csv"]
    )

    assert args.top_k == 8


def test_settings_require_one_served_candidate_budget():
    with pytest.raises(ValueError, match="fetch_k and rerank_top must match"):
        Settings(neo4j_password="test", fetch_k=30, rerank_top=29)


def test_eval_cli_records_the_console_script_command(monkeypatch):
    import vtlaw.cli
    import vtlaw.eval

    received: dict = {}
    monkeypatch.setattr(
        vtlaw.eval,
        "run_eval",
        lambda *_args, **kwargs: received.update(kwargs)
        or SimpleNamespace(recall_at_k={8: 1.0}),
    )
    monkeypatch.setattr(
        vtlaw.cli.sys,
        "argv",
        [
            r"C:\repo\.venv\Scripts\vtlaw.exe",
            "eval",
            "run",
            "data/evaluation/qa/QA_NLP.csv",
            "--top-k",
            "8",
        ],
    )

    args = build_parser().parse_args(sys.argv[1:])

    assert cmd_eval_run(args) == 0
    assert received["command"] == (
        "vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 8"
    )


def test_eval_cli_returns_success_at_the_requested_nondefault_cutoff(monkeypatch):
    import vtlaw.eval

    monkeypatch.setattr(
        vtlaw.eval,
        "run_eval",
        lambda *_args, **_kwargs: SimpleNamespace(recall_at_k={5: 0.1}),
    )
    args = build_parser().parse_args(
        ["eval", "run", "data/evaluation/qa/QA_NLP.csv", "--top-k", "5"]
    )

    assert cmd_eval_run(args) == 0


def test_eval_decomposition_records_and_forwards_subqueries(tmp_path, monkeypatch):
    dataset = tmp_path / "qa.csv"
    dataset.write_text(
        "question,reference\n"
        "Mức phạt?,168/2024/NĐ-CP::article::6\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    received: list[list[str] | None] = []
    clock = [0.0]

    class FakeGraphClient:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    class FakeEmbedder:
        def __init__(self, **_kwargs):
            pass

    class FakeRetriever:
        def __init__(self, *_args):
            pass

        def search_and_rerank(
            self, _question, *, k, strategy, as_of=None, sub_queries=None, **_kwargs
        ):
            clock[0] += 1.0
            received.append(sub_queries)
            return SearchResult(
                query="Mức phạt?",
                strategy=strategy,
                hits=[
                    Hit(
                        uid="168/2024/NĐ-CP::article::6",
                        score=1.0,
                        doc_identity="168/2024/NĐ-CP",
                        label="Article",
                        content="Nội dung",
                    )
                ][:k],
            )

    class FakeDecomposer:
        def decompose(self, _question):
            clock[0] += 4.0
            return [{"query": "mức xử phạt xe mô tô"}]

    import vtlaw.embed
    import vtlaw.eval.runner
    import vtlaw.generate.llm_client
    import vtlaw.graph.client
    import vtlaw.retrieve.query_parser
    import vtlaw.retrieve.search

    monkeypatch.setattr(vtlaw.embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(vtlaw.graph.client, "GraphClient", FakeGraphClient)
    monkeypatch.setattr(vtlaw.retrieve.search, "HybridRetriever", FakeRetriever)
    monkeypatch.setattr(vtlaw.eval.runner.time, "time", lambda: clock[0])
    monkeypatch.setattr(vtlaw.generate.llm_client, "LLMClient", lambda _settings: object())
    monkeypatch.setattr(
        vtlaw.retrieve.query_parser, "QueryDecomposer", lambda _llm: FakeDecomposer()
    )

    run_eval(dataset, top_k=1, decompose=True, output_path=output)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert received == [["Mức phạt?", "mức xử phạt xe mô tô"]]
    assert report["configuration"]["query_decomposition"] is True
    assert report["rows"][0]["sub_queries"] == received[0]
    assert report["rows"][0]["latency_s"] == 5.0


def test_eval_records_rows_dropped_by_a_retrieval_error(tmp_path, monkeypatch):
    """A crashed row must be visible, not silently absent from the denominator."""
    dataset = tmp_path / "qa.csv"
    dataset.write_text(
        "question,reference\n"
        "Câu hỏi lỗi?,168/2024/NĐ-CP::article::6\n"
        "Mức phạt?,168/2024/NĐ-CP::article::6\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    class FakeGraphClient:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    class FakeEmbedder:
        def __init__(self, **_kwargs):
            pass

    class FakeRetriever:
        def __init__(self, *_args):
            pass

        def search_and_rerank(self, question, *, k, strategy, as_of=None, **_kwargs):
            if question.startswith("Câu hỏi lỗi"):
                raise RuntimeError("lucene exploded")
            return SearchResult(
                query=question,
                strategy=strategy,
                hits=[
                    Hit(
                        uid="168/2024/NĐ-CP::article::6",
                        score=1.0,
                        doc_identity="168/2024/NĐ-CP",
                        label="Article",
                        content="Nội dung",
                    )
                ][:k],
            )

    import vtlaw.embed
    import vtlaw.graph.client
    import vtlaw.retrieve.search

    monkeypatch.setattr(vtlaw.embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(vtlaw.graph.client, "GraphClient", FakeGraphClient)
    monkeypatch.setattr(vtlaw.retrieve.search, "HybridRetriever", FakeRetriever)

    run_eval(dataset, top_k=1, output_path=output)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["dataset_rows"] == 2
    assert report["total_rows"] == 1
    assert len(report["skipped_rows"]) == 1
    assert "lucene exploded" in report["skipped_rows"][0]["error"]


def test_eval_rerank_keeps_output_k_separate_from_candidate_pool(tmp_path, monkeypatch):
    dataset = tmp_path / "qa.csv"
    dataset.write_text(
        "question,reference\n"
        "Mức phạt?,168/2024/NĐ-CP::article::6\n",
        encoding="utf-8",
    )
    received: dict = {}

    class FakeGraphClient:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    class FakeEmbedder:
        def __init__(self, **_kwargs):
            pass

    class FakeRetriever:
        def __init__(self, *_args):
            pass

        def search_and_rerank(self, _question, **kwargs):
            received.update(kwargs)
            return SearchResult(
                query="Mức phạt?",
                strategy=kwargs["strategy"],
                hits=[
                    Hit(
                        uid="168/2024/NĐ-CP::article::6",
                        score=1.0,
                        doc_identity="168/2024/NĐ-CP",
                        label="Article",
                        content="Nội dung",
                    )
                ],
            )

    import vtlaw.embed
    import vtlaw.eval.runner
    import vtlaw.graph.client
    import vtlaw.retrieve.search

    monkeypatch.setattr(vtlaw.embed, "Embedder", FakeEmbedder)
    monkeypatch.setattr(vtlaw.graph.client, "GraphClient", FakeGraphClient)
    monkeypatch.setattr(vtlaw.retrieve.search, "HybridRetriever", FakeRetriever)

    output = tmp_path / "result.json"
    run_eval(dataset, rerank=True, output_path=output)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert received["k"] == 8
    assert received["rerank_top"] == 30
    assert received["fetch_k"] == 30
    assert report["candidate_k"] == 30
    assert report["fetch_k"] == 30
    assert report["rerank_top"] == 30
    assert report["context_k"] == 8
    assert report["configuration"]["rerank_top"] == 30


def test_release_evidence_rejects_missing_or_incorrect_hashes(tmp_path):
    import vtlaw.eval.runner

    dataset = tmp_path / "qa.csv"
    manifest = tmp_path / "manifest.json"
    dataset.write_text("question,reference\nq,r\n", encoding="utf-8")
    manifest.write_text('{"manifest_version": 1}', encoding="utf-8")
    valid = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "inputs": {
            "dataset": {
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()
            },
            "snapshot_manifest": {
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
            },
        },
        "provenance": {
            "snapshot_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        },
    }

    vtlaw.eval.runner._verify_release_provenance(valid, dataset, manifest)
    for field in ("dataset_sha256", "snapshot_manifest_sha256"):
        invalid = json.loads(json.dumps(valid))
        target = invalid if field == "dataset_sha256" else invalid["provenance"]
        for value in (None, "0" * 64):
            target[field] = value
            with pytest.raises(ValueError, match=field):
                vtlaw.eval.runner._verify_release_provenance(invalid, dataset, manifest)

    for input_name in ("dataset", "snapshot_manifest"):
        for value in (None, "0" * 64):
            invalid = json.loads(json.dumps(valid))
            invalid["inputs"][input_name]["sha256"] = value
            with pytest.raises(ValueError, match=input_name):
                vtlaw.eval.runner._verify_release_provenance(invalid, dataset, manifest)


def test_release_evidence_output_requires_snapshot_provenance(tmp_path, monkeypatch):
    import vtlaw.embed
    import vtlaw.graph.client
    import vtlaw.retrieve.search

    class FakeGraphClient:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    class FakeRetriever:
        def __init__(self, *_args):
            pass

        def search_and_rerank(self, question, *, strategy, **_kwargs):
            return SearchResult(query=question, hits=[], strategy=strategy)

    monkeypatch.setattr(vtlaw.embed, "Embedder", lambda **_kwargs: object())
    monkeypatch.setattr(vtlaw.graph.client, "GraphClient", FakeGraphClient)
    monkeypatch.setattr(vtlaw.retrieve.search, "HybridRetriever", FakeRetriever)
    dataset = tmp_path / "qa.csv"
    dataset.write_text("question,reference\nq,r\n", encoding="utf-8")
    output = tmp_path / "2026-08-21-release-evidence-k8.json"

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="snapshot_manifest_sha256"):
        run_eval(
            dataset,
            output_path=output,
            command=(
                "python -m vtlaw eval run data/evaluation/qa/QA_NLP.csv "
                "--top-k 8 --output "
                "data/evaluation/results/2026-08-21-release-evidence-k8.json"
            ),
        )
    assert not output.exists()


def test_public_docs_record_measured_currency_stages_and_historical_design():
    root = Path(__file__).parents[2]
    docs = {
        path: (root / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/benchmark.md", "docs/portfolio-case-study.md")
    }
    public_docs = "\n".join(docs.values())
    design = (
        root / "docs/superpowers/specs/2026-08-14-traffic-law-rag-design.md"
    ).read_text(encoding="utf-8")

    assert "16/31 → 8/31 → 2/31" in public_docs
    assert "2026-08-20-citation-currency-before.json" in public_docs
    assert "2026-08-20-citation-currency-after.json" in public_docs
    assert "2026-08-20-citation-currency-hierarchy-fixed.json" in public_docs
    assert "2026-08-20" in public_docs
    assert "historical" in design[:500].lower()
    assert "python -m vtlaw" not in public_docs
    assert "No `2026-08-21-release-evidence-k8.json` result is claimed here" not in public_docs
    assert "2026-08-21-release-evidence-k8.json" in docs["docs/benchmark.md"]
    assert "--as-of 2026-08-21" in docs["docs/benchmark.md"]
    assert "fetches 30 candidates per retrieval leg" in public_docs
    assert "fused candidate pool" in public_docs
    assert "Result JSON files are intentionally ignored by Git." not in docs["docs/benchmark.md"]
    assert (
        "Result JSON files are ignored by Git except "
        "`data/evaluation/results/2026-08-21-release-evidence-k8.json`."
    ) in docs["docs/benchmark.md"]
