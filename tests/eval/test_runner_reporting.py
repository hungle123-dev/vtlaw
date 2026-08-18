"""Evaluation runs must record the exact conditions behind a score."""

from __future__ import annotations

import json
from datetime import date

from vtlaw.cli import build_parser
from vtlaw.config import Settings
from vtlaw.eval.runner import run_eval
from vtlaw.retrieve.search import Hit, SearchResult


def test_eval_records_dataset_config_and_as_of(tmp_path, monkeypatch):
    dataset = tmp_path / "qa.csv"
    dataset.write_text(
        "question,reference\n"
        "Mức phạt?,168/2024/NĐ-CP::article::6\n",
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

        def search(self, _question, *, k, strategy, as_of=None):
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

    run_eval(dataset, top_k=1, output_path=output, as_of=date(2025, 1, 1))

    report = json.loads(output.read_text(encoding="utf-8"))
    assert seen == [date(2025, 1, 1)]
    assert graph.closed is True
    assert report["as_of"] == "2025-01-01"
    assert report["dataset_sha256"]
    assert report["configuration"]["embed_model_revision"] == "embed-revision"


def test_eval_cli_parses_as_of_date():
    args = build_parser().parse_args(
        ["eval", "run", "data/evaluation/qa/QA_NLP.csv", "--as-of", "2025-01-01"]
    )

    assert args.as_of == date(2025, 1, 1)
