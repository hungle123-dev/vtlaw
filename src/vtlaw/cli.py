"""Command-line entry point, one command group per pipeline stage.

    vtlaw scrape run          fetch from the source API (network required)
    vtlaw scrape reindex      rebuild data/snapshot/manifest.json from disk
    vtlaw scrape verify       hash-check the snapshot (offline)
    vtlaw scrape status       list what the snapshot holds

    vtlaw parse check         parse the snapshot and report provision counts
    vtlaw parse show UID      print one provision with its ancestors

    vtlaw graph init          create constraints and indexes
    vtlaw graph import        parse the snapshot and write it to Neo4j
    vtlaw graph status        counts read back from the database
    vtlaw graph wipe          delete all data, keep the schema

    vtlaw embed run           embed provisions missing a vector
    vtlaw embed status        show embedding coverage per label

    vtlaw retrieve search Q   hybrid retrieval, no LLM
    vtlaw generate answer Q   retrieval plus a grounded LLM answer
    vtlaw api serve           start the FastAPI service and chat UI
    vtlaw eval run CSV        score retrieval against a labelled QA dataset
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, datetime

from vtlaw.embed import Embedder, embed_corpus, embedding_coverage
from vtlaw.graph import GraphClient, count_graph, import_amends_directory, import_documents
from vtlaw.parse import parse_corpus
from vtlaw.scrape import LegalDocumentClient, Scraper, Snapshot

log = logging.getLogger("vtlaw")


def _iso_date(text: str) -> date:
    """Parse a YYYY-MM-DD argument. Shared by every command taking --as-of."""
    return datetime.strptime(text, "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# stage 1 — scrape
# ---------------------------------------------------------------------------


def cmd_scrape_run(args: argparse.Namespace) -> int:
    snapshot = Snapshot(args.snapshot)
    with LegalDocumentClient(throttle_s=args.throttle) as client:
        scraper = Scraper(
            client,
            snapshot,
            doc_group_ids=args.doc_group_id or None,
            field_ids=args.field_id or None,
        )
        report = scraper.run(args.keywords, max_documents=args.max_documents)
        print(report.summary())
        print(f"  {client.stats.summary()}")

    for error in report.errors[:10]:
        print(f"  ERROR {error}")

    if report.blocked:
        print(
            "\nThe service refused the request. Not retrying, not working around it.\n"
            "The committed snapshot stays usable for every later stage."
        )
        return 2
    return 1 if report.failed and not report.written else 0


def cmd_scrape_reindex(args: argparse.Namespace) -> int:
    snapshot = Snapshot(args.snapshot)
    report = snapshot.reindex()
    print(f"manifest written: {snapshot.manifest_path}")
    print(f"  {report.summary()}")
    for error in report.errors:
        print(f"  ERROR {error}")
    return 1 if report.failed else 0


def cmd_scrape_verify(args: argparse.Namespace) -> int:
    report = Snapshot(args.snapshot).verify()

    print(f"documents tracked : {report.documents}")
    print(f"verified ok       : {report.ok}")
    for label, items in report.problems():
        if items:
            print(f"{label:26}: {len(items)} -> {items[:5]}")

    if report.is_clean:
        print("snapshot is clean")
        return 0
    print("snapshot has problems")
    return 1


def cmd_scrape_status(args: argparse.Namespace) -> int:
    snapshot = Snapshot(args.snapshot)
    manifest = snapshot.load_manifest()
    if not manifest:
        print(f"no manifest at {snapshot.manifest_path} — run `vtlaw scrape reindex`")
        return 1

    records = sorted(manifest.values(), key=lambda r: r["doc_identity"])
    total = sum(r["text_chars"] for r in records)
    print(f"{len(records)} documents, {total:,} characters\n")
    for record in records:
        updated = (record["upd_datetime"] or "")[:10]
        print(
            f"  {record['doc_identity']:18} {record['text_chars']:>7,} chars  "
            f"upd={updated:10} {record['content_sha256'][:12]}"
        )
    return 0


# ---------------------------------------------------------------------------
# stage 2 — parse
# ---------------------------------------------------------------------------


def cmd_parse_check(args: argparse.Namespace) -> int:
    """Parse everything and report. Writes nothing — a dry run of stage 2."""
    documents, stats = parse_corpus(Snapshot(args.snapshot))

    print(f"{'document':18} {'articles':>9} {'clauses':>8} {'points':>7}  effect")
    for parsed in sorted(documents, key=lambda d: d.document.doc_identity):
        counts = parsed.counts
        effect = parsed.document.effect_date
        print(
            f"  {parsed.document.doc_identity:16} {counts['article']:>9} "
            f"{counts['clause']:>8} {counts['point']:>7}  {effect}"
        )

    print()
    print(stats.summary())

    for error in stats.failed:
        print(f"  ERROR {error}")
    return 1 if stats.failed else 0


def cmd_parse_show(args: argparse.Namespace) -> int:
    """Print one provision with its ancestors.

    Useful because a Point on its own is not an answer: the offence is in the
    Point and the amount is in its parent Clause.
    """
    documents, _ = parse_corpus(Snapshot(args.snapshot))
    by_uid = {p.uid: p for d in documents for p in d.provisions}

    provision = by_uid.get(args.uid)
    if provision is None:
        print(f"no provision with uid {args.uid!r}")
        return 1

    chain = [provision]
    while chain[-1].parent_uid is not None:
        parent = by_uid.get(chain[-1].parent_uid)
        if parent is None:
            break
        chain.append(parent)

    print(f"{provision.citation}\n")
    for node in reversed(chain):
        label = node.level.capitalize()
        heading = f"{label} {node.number}"
        if node.title:
            heading += f": {node.title}"
        print(heading)
        if node.content:
            for line in node.content.splitlines():
                print(f"  {line}")
        print()
    return 0


# ---------------------------------------------------------------------------
# stage 3 — graph
# ---------------------------------------------------------------------------


def _graph_client() -> GraphClient:
    client = GraphClient()
    client.verify()
    return client


def cmd_graph_init(_: argparse.Namespace) -> int:
    with _graph_client() as client:
        client.ensure_schema()
        client.await_indexes()
        for row in client.index_states():
            print(
                f"  {row['name']:26} {row['type']:9} {row['state']:8} "
                f"{row['populationPercent']:6.1f}%"
            )
    print("schema ready")
    return 0


def cmd_graph_import(args: argparse.Namespace) -> int:
    documents, parse_stats = parse_corpus(Snapshot(args.snapshot))
    print(f"parsed : {parse_stats.summary()}")
    if parse_stats.failed:
        for error in parse_stats.failed:
            print(f"  ERROR {error}")
        return 1

    with _graph_client() as client:
        if args.wipe:
            client.wipe()
        stats = import_documents(client, documents)
        amend_stats = import_amends_directory(client, "data/amends")
        client.await_indexes()
        counts = count_graph(client)

    print(f"imported: {stats.summary()}")
    print(f"amends : {amend_stats.summary()}")
    print(f"in graph: {counts.summary()}")

    # Read-back is the check: it catches a write that reported success without
    # landing. Parse, import and database counts must all agree.
    if counts.provisions != parse_stats.provisions:
        print(
            f"MISMATCH parsed {parse_stats.provisions:,} but graph holds "
            f"{counts.provisions:,}"
        )
        return 1
    if stats.failed or stats.missing_parent:
        return 1
    return 0


def cmd_graph_status(_: argparse.Namespace) -> int:
    with _graph_client() as client:
        counts = count_graph(client)
        print(counts.summary())
        print()
        with client.session() as session:
            amends_count = session.run(
                "MATCH ()-[r:AMENDS]->() RETURN count(r) AS n"
            ).single()["n"]
            amend_types = session.run(
                "MATCH ()-[r:AMENDS]->() RETURN r.type AS type, count(r) AS n "
                "ORDER BY n DESC"
            ).data()
        print(f"amends: {amends_count} edges")
        for row in amend_types:
            print(f"  {row['type']:20} {row['n']:>4}")
        print()
        print()
        with client.session() as session:
            rows = session.run(
                "MATCH (d:Document) "
                "OPTIONAL MATCH (d)-[:HAS_ARTICLE]->(a:Article) "
                "RETURN d.doc_identity AS identity, "
                "toString(d.effect_date) AS effect_date, "
                "d.effect_status AS status, count(a) AS articles "
                "ORDER BY identity"
            ).data()
        for row in rows:
            print(
                f"  {row['identity']:18} {row['articles']:>4} articles  "
                f"effect={row['effect_date'] or 'N/A':12} {row['status'] or ''}"
            )
    return 0


def cmd_graph_wipe(_: argparse.Namespace) -> int:
    with _graph_client() as client:
        client.wipe()
        print(f"deleted all data: {count_graph(client).summary()}")
    return 0


def cmd_graph_amends(args: argparse.Namespace) -> int:
    """Import amendment relationships from JSON files into the graph."""
    from pathlib import Path

    from vtlaw.graph import import_amends_directory

    amends_dir = Path(args.amends_dir)
    if not amends_dir.is_dir():
        print(f"amends directory not found: {amends_dir}")
        return 1

    with _graph_client() as client:
        stats = import_amends_directory(client, amends_dir)

    print(f"amends: {stats.summary()}")
    if stats.failed:
        print(f"  {len(stats.failed)} amendments skipped (source/target not in graph)")
        for uid_pair in stats.failed[:5]:
            print(f"    {uid_pair}")
    return 0


# ---------------------------------------------------------------------------
# stage 4 — embed
# ---------------------------------------------------------------------------


def cmd_embed_run(args: argparse.Namespace) -> int:
    """Embed provisions that are missing a vector.

    Idempotent: nodes already carrying an embedding are skipped, so a re-run
    after a partial failure resumes from where it left off.
    """
    from vtlaw.config import get_settings

    settings = get_settings()
    if args.batch_size:
        settings.embed_batch_size = args.batch_size

    with _graph_client() as client:
        embedder = Embedder(settings)
        stats = embed_corpus(client, embedder, batch_size=settings.embed_batch_size)
        client.await_indexes()
        coverage = embedding_coverage(client)

    print(stats.summary())
    print()
    for row in coverage:
        gap = row["total"] - row["embedded"]
        flag = "" if gap == 0 else f"  <-- {gap} missing"
        print(
            f"  {row['label']:8} total={row['total']:>5} "
            f"with_content={row['with_content']:>5} "
            f"title_only={row['title_only']:>5} "
            f"embedded={row['embedded']:>5}{flag}"
        )
    return 0


def cmd_embed_status(_: argparse.Namespace) -> int:
    with _graph_client() as client:
        coverage = embedding_coverage(client)

    for row in coverage:
        gap = row["total"] - row["embedded"]
        flag = "" if gap == 0 else f"  <-- {gap} missing"
        print(
            f"  {row['label']:8} total={row['total']:>5} "
            f"with_content={row['with_content']:>5} "
            f"title_only={row['title_only']:>5} "
            f"embedded={row['embedded']:>5}{flag}"
        )
    return 0


# ---------------------------------------------------------------------------
# stage 5 — retrieve
# ---------------------------------------------------------------------------


def cmd_retrieve_search(args: argparse.Namespace) -> int:
    """Search for provisions matching a query."""
    from vtlaw.config import get_settings
    from vtlaw.embed import Embedder
    from vtlaw.retrieve import HybridRetriever

    settings = get_settings()
    with _graph_client() as client:
        embedder = Embedder(settings)
        retriever = HybridRetriever(client, embedder, settings)

        if args.rerank:
            result = retriever.search_and_rerank(
                args.query,
                k=args.top_k,
                strategy=args.strategy,
                rerank_top=args.rerank_top,
                reranker_model=args.reranker_model,
                rerank_enabled=True,
                as_of=args.as_of,
            )
        else:
            result = retriever.search(
                args.query,
                k=args.top_k,
                strategy=args.strategy,
                as_of=args.as_of,
            )

    if args.rerank:
        print(f"reranked: {result.retrieval_score} -> {len(result.hits)} hits")
    else:
        print(result.summary())

    print()
    for i, hit in enumerate(result.hits, 1):
        title = f" ({hit.title})" if hit.title else ""
        print(f"{i:2}. [{hit.label:7}] {hit.uid}{title}")
        print(f"    score={hit.score:.4f} doc={hit.doc_identity}")
        snippet = hit.content[:120].replace("\n", " ")
        print(f"    {snippet}...")
        print()
    return 0


# ---------------------------------------------------------------------------
# stage 6 — generate
# ---------------------------------------------------------------------------


def cmd_generate_answer(args: argparse.Namespace) -> int:
    """Answer a question using retrieved provisions and LLM generation."""
    from vtlaw.config import get_settings
    from vtlaw.embed import Embedder
    from vtlaw.generate.answer import AnswerGenerator
    from vtlaw.generate.llm_client import LLMClient

    settings = get_settings()
    with _graph_client() as client:
        embedder = Embedder(settings)
        llm = LLMClient(settings)
        generator = AnswerGenerator(client, embedder, llm, settings)

        answer = generator.answer(
            args.question,
            strategy=args.strategy,
            as_of=args.as_of,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    print(f"Question: {args.question}")
    print(f"Strategy: {answer.strategy} (reranked={answer.reranked})")
    print(f"Sources: {len(answer.sources)} provisions")
    print()
    print("Answer:")
    print(answer.text)
    print()
    print("Sources:")
    for i, hit in enumerate(answer.sources, 1):
        print(f"  {i}. {hit.uid} ({hit.doc_identity}, score={hit.score:.3f})")
    return 0


# ---------------------------------------------------------------------------
# stage 7 — api
# ---------------------------------------------------------------------------


def cmd_api_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    from vtlaw.config import get_settings

    settings = get_settings()
    host = args.host or settings.api_host
    port = args.port or settings.api_port

    import uvicorn

    log.info("starting server on %s:%d", host, port)
    uvicorn.run(
        "vtlaw.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
    return 0


# ---------------------------------------------------------------------------
# stage 8 — eval
# ---------------------------------------------------------------------------


def cmd_eval_run(args: argparse.Namespace) -> int:
    """Run evaluation on a labeled QA dataset."""
    from vtlaw.eval import run_eval

    output = args.output or None
    agg = run_eval(
        args.dataset,
        top_k=args.top_k,
        strategy=args.strategy,
        output_path=output,
        rerank=args.rerank,
        decompose=args.decompose,
        fetch_k=args.fetch_k,
        limit=args.limit,
        as_of=args.as_of,
        command=subprocess.list2cmdline(["vtlaw", *sys.argv[1:]]),
    )

    # Exit code: 0 if the requested evaluation cutoff has a non-zero recall.
    return 0 if agg.recall_at_k.get(args.top_k, 0.0) > 0 else 1


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vtlaw")
    parser.add_argument(
        "--snapshot",
        default="data/snapshot",
        help="snapshot root (default: data/snapshot)",
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    # -- scrape ------------------------------------------------------------
    scrape = stages.add_parser(
        "scrape", help="stage 1 — source acquisition"
    ).add_subparsers(dest="action", required=True)

    run = scrape.add_parser("run", help="fetch documents from the source API")
    run.add_argument("--keywords", default="giao thông đường bộ")
    run.add_argument("--max-documents", type=int, default=None)
    run.add_argument(
        "--throttle", type=float, default=1.2,
        help="minimum seconds between requests (default: 1.2)",
    )
    run.add_argument("--doc-group-id", type=int, action="append", default=[])
    run.add_argument("--field-id", type=int, action="append", default=[])
    run.set_defaults(func=cmd_scrape_run)

    scrape.add_parser(
        "reindex", help="rebuild the manifest from files on disk"
    ).set_defaults(func=cmd_scrape_reindex)
    scrape.add_parser(
        "verify", help="hash-check the snapshot against its manifest"
    ).set_defaults(func=cmd_scrape_verify)
    scrape.add_parser(
        "status", help="list documents in the snapshot"
    ).set_defaults(func=cmd_scrape_status)

    # -- parse -------------------------------------------------------------
    parse = stages.add_parser(
        "parse", help="stage 2 — snapshot text to provisions"
    ).add_subparsers(dest="action", required=True)

    parse.add_parser(
        "check", help="parse everything and report counts (writes nothing)"
    ).set_defaults(func=cmd_parse_check)

    show = parse.add_parser("show", help="print one provision with its ancestors")
    show.add_argument("uid", help="e.g. 168/2024/NĐ-CP::article::6::clause::3::point::a")
    show.set_defaults(func=cmd_parse_show)

    # -- graph -------------------------------------------------------------
    graph = stages.add_parser(
        "graph", help="stage 3 — provisions to Neo4j"
    ).add_subparsers(dest="action", required=True)

    graph.add_parser(
        "init", help="create constraints and indexes"
    ).set_defaults(func=cmd_graph_init)

    graph_import = graph.add_parser("import", help="parse and write to Neo4j")
    graph_import.add_argument(
        "--wipe", action="store_true", help="delete existing data first"
    )
    graph_import.set_defaults(func=cmd_graph_import)

    graph.add_parser(
        "status", help="counts read back from the database"
    ).set_defaults(func=cmd_graph_status)
    graph.add_parser(
        "wipe", help="delete all data, keep the schema"
    ).set_defaults(func=cmd_graph_wipe)

    graph_amends = graph.add_parser(
        "import-amends", help="import amendment edges from JSON"
    )
    graph_amends.add_argument(
        "--amends-dir", default="data/amends",
        help="directory containing amends JSON files (default: data/amends)",
    )
    graph_amends.set_defaults(func=cmd_graph_amends)

    # -- embed -------------------------------------------------------------
    embed = stages.add_parser(
        "embed", help="stage 4 — vector embeddings"
    ).add_subparsers(dest="action", required=True)

    embed_run = embed.add_parser("run", help="embed provisions missing a vector")
    embed_run.add_argument(
        "--batch-size", type=int, default=None,
        help="override the configured batch size (default: from .env)",
    )
    embed_run.set_defaults(func=cmd_embed_run)

    embed.add_parser(
        "status", help="show embedding coverage per label"
    ).set_defaults(func=cmd_embed_status)

    # -- retrieve ----------------------------------------------------------
    retrieve = stages.add_parser(
        "retrieve", help="stage 5 — hybrid search + rerank"
    ).add_subparsers(dest="action", required=True)

    search = retrieve.add_parser("search", help="search for provisions")
    search.add_argument("query", help="the question to search for")
    search.add_argument(
        "--strategy",
        choices=["hybrid", "vector", "bm25"],
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    search.add_argument(
        "--top-k", type=int, default=10,
        help="number of results (default: 10)",
    )
    search.add_argument(
        "--rerank", action="store_true",
        help="rerank results with a cross-encoder",
    )
    search.add_argument(
        "--rerank-top", type=int, default=None,
        help="number of candidates to rerank (default: the configured rerank_top)",
    )
    search.add_argument(
        # No literal default. This used to default to an English
        # ms-marco-MiniLM, so `--rerank` from the CLI reranked Vietnamese legal
        # text with an English model — silently worse than the configured
        # Vietnamese cross-encoder the benchmark actually measured.
        "--reranker-model",
        default=None,
        help="cross-encoder model (default: the configured RERANK_MODEL)",
    )
    search.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        help="legal-effective date for retrieval (YYYY-MM-DD, default: today)",
    )
    search.set_defaults(func=cmd_retrieve_search)

    # -- generate ----------------------------------------------------------
    generate = stages.add_parser(
        "generate", help="stage 6 — answer questions with LLM"
    ).add_subparsers(dest="action", required=True)

    answer = generate.add_parser("answer", help="answer a question")
    answer.add_argument("question", help="the question to answer")
    answer.add_argument(
        "--strategy",
        choices=["hybrid", "vector", "bm25"],
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    answer.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        help="date for temporal reasoning (format: YYYY-MM-DD, default: today)",
    )
    answer.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM sampling temperature (default: 0.3)",
    )
    answer.add_argument(
        "--max-tokens", type=int, default=2048,
        help="maximum tokens in generated answer (default: 2048)",
    )
    answer.set_defaults(func=cmd_generate_answer)

    # -- api ---------------------------------------------------------------
    api = stages.add_parser(
        "api", help="stage 7 — HTTP API server"
    ).add_subparsers(dest="action", required=True)

    serve = api.add_parser("serve", help="start the FastAPI server")
    serve.add_argument(
        "--host", default=None,
        help="server host (default: from .env or 127.0.0.1)",
    )
    serve.add_argument(
        "--port", type=int, default=None,
        help="server port (default: from .env or 18080)",
    )
    serve.set_defaults(func=cmd_api_serve)

    # -- eval --------------------------------------------------------------
    eval_stage = stages.add_parser(
        "eval", help="stage 8 — evaluation harness"
    ).add_subparsers(dest="action", required=True)

    eval_run = eval_stage.add_parser("run", help="run evaluation on a dataset")
    eval_run.add_argument(
        "dataset",
        help="path to CSV with question,reference columns",
    )
    eval_run.add_argument(
        "--top-k", type=int, default=8,
        help="number of results to retrieve and score per question (default: 8)",
    )
    eval_run.add_argument(
        "--strategy",
        choices=["hybrid", "vector", "bm25"],
        default="hybrid",
        help="retrieval strategy (default: hybrid)",
    )
    eval_run.add_argument(
        "--output", default=None,
        help="output JSON file for detailed results (default: none)",
    )
    eval_run.add_argument(
        "--rerank", action="store_true",
        help="cross-encoder rerank the candidates before scoring",
    )
    eval_run.add_argument(
        "--decompose", action="store_true",
        help="expand each question with LLM-generated legal-search phrasings",
    )
    eval_run.add_argument(
        "--fetch-k", type=int, default=None,
        help="candidate budget per retrieval leg and reranker (default: configured fetch_k)",
    )
    eval_run.add_argument(
        "--limit", type=int, default=None,
        help="score only the first N rows (quick check, not a report)",
    )
    eval_run.add_argument(
        "--as-of",
        type=_iso_date,
        default=None,
        help="legal-effective date for every query (YYYY-MM-DD)",
    )
    eval_run.set_defaults(func=cmd_eval_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
