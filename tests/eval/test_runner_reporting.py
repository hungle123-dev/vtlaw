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

        def search(self, _question, *, k, strategy, as_of=None, sub_queries=None):
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

    run_eval(dataset, top_k=1, rerank=True, fetch_k=4)

    assert received["k"] == 1
    assert received["rerank_top"] == 4
    assert received["fetch_k"] == 4
