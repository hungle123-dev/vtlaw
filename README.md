# vtlaw — Vietnamese Legal Graph RAG

A production-minded, reproducible RAG system for Vietnamese road-traffic law.
It is built on the fixed corpus, QA labels, and amendment annotations supplied
by [NLP-LegalQA](https://github.com/n3sfan/NLP-LegalQA), while implementing its
own parser, Neo4j graph, hybrid retrieval, API, and evaluation workflow.

This is a portfolio system, not a legal-advice service. It is deliberately
honest about both what the graph proves and what the fixed source data cannot
prove.

For a concise project narrative, demo flow, and CV-ready bullets, see
[docs/portfolio-case-study.md](docs/portfolio-case-study.md).

## What is implemented

```text
NLP-LegalQA snapshot
        │
        ▼
Điều / Khoản / Điểm parser ──► Neo4j hierarchy + AMENDS annotations
        │                                      │
        ▼                                      ▼
exact citation lookup ──► vector + BM25 + weighted RRF ──► graph context
                                                               │
                                                               ▼
                                                   cited LLM answer (optional)
```

- 12 fixed source documents, parsed into 7,381 citable provisions.
- A graph hierarchy preserves `Document → Article → Clause → Point`; a Point
  can be expanded with its parent Clause, where a penalty amount often lives.
- 440 resolved `AMENDS` edges represent NLP-LegalQA amendment annotations.
- Every retrieval accepts `as_of`; it filters document effective/expiry dates.
  Amendment demotion is also date-bounded by the amending document's effective
  date.
- A full `Điều / Khoản / Điểm + document identity` citation is resolved exactly
  before approximate retrieval. Ambiguous citations stay on the hybrid path.
- The baseline is deterministic hybrid retrieval. LLM decomposition and the
  cross-encoder reranker are opt-in experiments, not hidden defaults.
- API includes `/health`, `/metrics`, request IDs, optional API-key protection,
  loopback binding by default, rate limiting, CORS allow-listing, and Redis
  caching keyed by corpus, model, retrieval configuration, and `as_of` date.

## Legal-temporal boundary

The graph can tell the system that a provision has a recorded `bãi bỏ` or
`thay thế` relationship as of a given date. It **cannot** reconstruct a new,
consolidated version of every amended provision: the corpus contains amendment
instructions, not a verified consolidated text. The system therefore never
pretends to synthesize a replacement rule; it cites the retrieved source and
keeps that limitation explicit.

The project uses only the committed NLP-LegalQA artifacts. Scraper code is kept
to make the original acquisition path inspectable, but normal operation is
offline and does not fetch PDFs or another legal source. The upstream endpoint
currently rejects or errors on requests, and the client stops on refusal rather
than trying to bypass it. See [data/README.md](data/README.md).

## Reproducible evaluation

Scores are meaningful only with their corpus, model revision, retrieval
settings, and legal date. `vtlaw eval run` writes all of those, plus a dataset
SHA-256, into its JSON result.

```bash
vtlaw eval run data/evaluation/qa/QA_NLP.csv \
  --top-k 5 --as-of 2026-08-18 \
  --output data/evaluation/results/qa-nlp.json

vtlaw eval run data/evaluation/qa/QA_Part2345.csv \
  --top-k 5 --as-of 2026-08-18 \
  --output data/evaluation/results/qa-part2345.json
```

`QA_Part2`–`QA_Part5` overlap with `QA_Part234` / `QA_Part2345`; they must not
be averaged as independent datasets. The current reproducible report and its
limits live in [docs/benchmark.md](docs/benchmark.md).

## Quick start

```bash
cp .env.example .env
# Set NEO4J_PASSWORD in .env (and LLM_API_KEY only if generation is wanted).

docker compose up -d

uv venv
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -e ".[dev,embed,serve,worker]"

vtlaw scrape verify             # hash-check only; no network access
vtlaw parse check               # 12 docs -> 7,381 provisions
vtlaw graph import --wipe       # imports hierarchy and AMENDS annotations
vtlaw embed run                 # model/provenance-aware vector refresh
vtlaw graph status              # read counts and amendment edges back
```

After a source re-import, provision vectors are cleared deliberately: their
input text may have changed. Run `vtlaw embed run` again; it refreshes only
vectors whose embedding fingerprint is stale or missing.

```bash
# Deterministic retrieval-only query
vtlaw retrieve search "không đội mũ bảo hiểm phạt bao nhiêu"

# Exact citation lookup (no vector search needed for this query)
vtlaw retrieve search "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP"

# LLM-backed answer, with a legally scoped date
vtlaw generate answer "vượt quá tốc độ phạt bao nhiêu" --as-of 2025-01-01

# HTTP API on http://127.0.0.1:18080
vtlaw api serve
```

Example API call:

```bash
curl -X POST http://127.0.0.1:18080/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"không đội mũ bảo hiểm phạt bao nhiêu","as_of":"2025-01-01"}'
```

## Development checks

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync pytest -m "not integration" -q
VTLAW_TEST_WIPE=1 uv run --no-sync pytest -m integration -q
```

Integration tests wipe Neo4j, so run them only against the isolated CI service
or an expendable local database. They require the explicit `VTLAW_TEST_WIPE=1`
opt-in. GitHub Actions runs lint, offline tests,
isolated Neo4j/Redis integration tests, and a non-root Docker image import
check.

## Scope and Definition of Done

The project is complete for its stated portfolio scope when all of the
following hold:

1. Snapshot hashes, parser counts, graph counts, AMENDS edges, and embedding
   coverage agree with their read-back checks.
2. Unit and isolated integration tests pass; Docker builds and imports.
3. The two non-overlapping reporting tracks are run with a recorded `as_of`
   date and their generated JSON is summarized in `docs/benchmark.md`.
4. API health is green with Neo4j and Redis, and a retrieval-only `/chat` call
   returns citable sources without requiring an LLM key.

That scope intentionally excludes live crawling, PDF/OCR ingestion, a frontend,
and legal consolidation. Those require authoritative source access and a
separate validation process rather than more retrieval code.

## Layout

```text
src/vtlaw/
  scrape/      reproducible acquisition client; not part of normal runtime
  parse/       Vietnamese legal-structure parser and citable UID model
  graph/       Neo4j schema, hierarchy importer, AMENDS importer
  embed/       pinned Vietnamese bi-encoder and provenance-aware indexer
  retrieve/    exact citation lookup, vector/BM25/RRF, temporal safeguards
  generate/    context construction and optional grounded LLM generation
  api/         FastAPI application and operational middleware
  eval/        Recall@k, MRR, latency, reproducibility metadata
  ingest/      optional ARQ maintenance worker

data/          committed NLP-LegalQA snapshot, labels, and annotations
tests/         offline unit tests and isolated Neo4j integration tests
```

## Licence

MIT. Corpus provenance and constraints are documented in
[data/README.md](data/README.md).
