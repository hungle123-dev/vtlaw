# vtlaw — Vietnamese Legal Graph RAG

Vietnamese road-traffic law as a queryable graph: 7,381 citable provisions in
Neo4j, answers that cite Điều/Khoản/Điểm or refuse to answer at all.
**Recall@5 0.5637 → 0.7713 across the two QA tracks with zero LLM calls in the
retrieval path**, and 0.86 on the primary track with an opt-in decomposition
profile. Every score ships with its corpus hash, pinned model revision, fusion
weights, and legal `as_of` date — the deterministic path reproduces to four
decimal places across runs, and the LLM-assisted one is reported with its
measured run-to-run spread instead of a single flattering figure.

Built on the fixed corpus, QA labels, and amendment annotations from
[NLP-LegalQA](https://github.com/n3sfan/NLP-LegalQA), with an independent
parser, graph, retrieval stack, API, and evaluation harness.

This is a portfolio system, not a legal-advice service. It is deliberately
honest about both what the graph proves and what the fixed source data cannot.

→ [Portfolio case study](docs/portfolio-case-study.md) ·
[Benchmark report](docs/benchmark.md) · [Data provenance](data/README.md)

## Measured against the upstream project

Both sets of numbers are retrieval metrics on the **same 200 questions**:
upstream reports per-part means over `QA_Part2`–`QA_Part5`, which is the same
question set, in the same order, as the `QA_Part2345.csv` track used here
(50 rows per part, so a mean of means equals the pooled mean). Same corpus,
same `bkai-foundation-models/vietnamese-bi-encoder`.

| Pipeline | Recall@5 | MRR | LLM calls per query |
|---|---:|---:|---|
| NLP-LegalQA — basic RAG | 0.4892 | 0.4626 | 0 |
| **vtlaw — baseline** (hybrid + weighted RRF) | **0.5637** | **0.5127** | 0 |
| **vtlaw — decomposition profile** | **0.6192** | **0.5637** | 1 |
| NLP-LegalQA — full pipeline (rerank top-30 + Borda/max aggregation) | 0.6711 | 0.7033 | 1+ |

Read honestly: vtlaw's *deterministic, zero-LLM* path beats the upstream
deterministic baseline by 15% relative Recall@5, and one decomposition call
closes most of the remaining gap. Upstream's full pipeline still wins,
particularly on MRR — it spends a cross-encoder pass over 30 candidates plus
rank aggregation to get there. That trade was measured here and rejected as a
default: the same reranker cost 7–9 s p50 and *regressed* the QA_NLP track
(see [benchmark.md](docs/benchmark.md)).

Caveats that belong next to the table: upstream's relevance test is a plain
`startswith`, this one requires the match to land on a `::` UID boundary — on
these rows the two agree to 4 decimal places, but they are not the same
predicate. Upstream's own numbers were not re-measured here; they are quoted
from its committed `eval_results/`.

## What is implemented

```text
NLP-LegalQA snapshot
        │
        ▼
conversation history ──► intent router ──► direct / reject / graph template
        │                                           │
        ▼                                           ▼
follow-up rewrite ──► exact citation or vector + BM25 + weighted RRF
                                      │
                                      ▼
                          graph context + cited LLM answer
                                      │
                                      ▼
                     citation check → one repair → evidence fallback
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
- The baseline is deterministic hybrid retrieval. LLM query decomposition is a
  measured quality profile; the cross-encoder reranker remains an explicit
  experiment because its full-track result is mixed and much slower.
- The router follows NLP-LegalQA's four intents (`direct_answer`, `retrieve`,
  `reject`, `cypher_query`). Its graph-query branch uses three parameterized,
  read-only templates (article count, signers, amendment history), never raw
  LLM-generated Cypher.
- Multi-turn follow-ups are rewritten to a standalone question only after the
  original message is routed. Generated answers are checked against retrieved
  evidence; an unverified answer gets one constrained repair, then a cited
  evidence fallback rather than an unsupported legal claim.
- API includes `/health`, `/metrics`, request IDs, optional API-key protection,
  loopback binding by default, rate limiting, CORS allow-listing, and Redis
  caching keyed by corpus, model, retrieval profile/configuration, and `as_of`
  date. A dependency-free chat UI is served at `/`.

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
SHA-256 and any rows a retrieval error dropped, into its JSON result.

```bash
vtlaw eval run data/evaluation/qa/QA_NLP.csv \
  --top-k 5 --as-of 2026-08-20 \
  --output data/evaluation/results/qa-nlp.json

vtlaw eval run data/evaluation/qa/QA_Part2345.csv \
  --top-k 5 --as-of 2026-08-20 \
  --output data/evaluation/results/qa-part2345.json
```

`QA_Part2`–`QA_Part5` concatenate exactly into `QA_Part2345`; they must not be
averaged as independent datasets. The current reproducible report and its
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

# Retrieval scoped to a past legal date
vtlaw retrieve search "vượt quá tốc độ" --as-of 2024-06-01

# LLM-backed answer, with a legally scoped date
vtlaw generate answer "vượt quá tốc độ phạt bao nhiêu" --as-of 2025-01-01

# HTTP API on http://127.0.0.1:18080
vtlaw api serve
```

Open <http://127.0.0.1:18080/> for the portfolio chat UI. It exposes the
retrieval profile, legal date, evidence, routing decision, citation status, and
timings without adding a frontend runtime dependency.

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
4. API health is green with Neo4j and Redis; baseline, decomposition, safe
   graph-template, and multi-turn `/chat` flows return the stated contracts.
5. Every generated API answer is either citation-verified against its retrieved
   evidence or replaced with a cited evidence fallback; this is a citation
   provenance check, not proof of legal correctness.
6. The root UI renders from the built Docker image as well as the source tree.

That scope intentionally excludes live crawling, PDF/OCR ingestion, arbitrary
LLM Cypher execution, and legal consolidation. Those require authoritative
source access or a separate validation process rather than more retrieval code.

## Layout

```text
src/vtlaw/
  scrape/      reproducible acquisition client; not part of normal runtime
  parse/       Vietnamese legal-structure parser and citable UID model
  graph/       Neo4j schema, hierarchy importer, AMENDS importer, safe templates
  embed/       pinned Vietnamese bi-encoder and provenance-aware indexer
  retrieve/    exact citation lookup, vector/BM25/RRF, router, temporal safeguards
  generate/    context construction, grounded generation, citation verification
  api/         FastAPI application, operational middleware, static chat UI
  eval/        Recall@k, MRR, latency, reproducibility metadata
  ingest/      optional ARQ maintenance worker

data/          committed NLP-LegalQA snapshot, labels, and annotations
tests/         offline unit tests and isolated Neo4j integration tests
```

## Licence

MIT. Corpus provenance and constraints are documented in
[data/README.md](data/README.md).
