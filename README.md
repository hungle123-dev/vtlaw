# vtlaw — Vietnamese Legal GraphRAG

An amendment-aware RAG system for Vietnamese traffic law. It returns a cited
answer only when the cited provision was present in the model's evidence;
otherwise it falls back to a transparent, source-backed response.

This is a portfolio system for retrieval and LLM engineering, not legal advice.

![vtlaw chat UI showing a verified citation](docs/images/ui-hybrid-answer.png)

## Why this project is interesting

Legal RAG fails in ways a generic document chatbot does not:

- a penalty amount may be in a parent clause while the offence is in a point;
- a full legal citation must resolve exactly, never by semantic similarity;
- amendments and effective dates can make a plausible retrieved rule unsafe;
- a fluent answer is not enough unless its citation is traceable to the actual
  retrieved context.

vtlaw turns those constraints into explicit retrieval, graph, and generation
contracts instead of hiding them inside a prompt.

## System at a glance

```text
question + conversation
        │
        ├── intent router ──► direct / reject / safe graph template
        │
        └── follow-up rewrite ──► exact citation lookup
                                  or vector + BM25 + weighted RRF
                                               │
                                               ▼
                         amendment-aware graph context (as_of)
                                               │
                                               ▼
                         grounded answer → citation verification
                                               │
                         verified answer or cited fallback
```

The baseline fetches 30 candidates per retrieval leg and fuses them into a
30-item fused candidate pool. It returns the best 8 contexts after temporal
and provenance safeguards.

## What I built

- **GraphRAG:** 12 legal documents become 7,381 stable
  `Document → Article → Clause → Point` nodes in Neo4j, with 440 recorded
  amendment edges.
- **Hybrid retrieval:** Vietnamese bi-encoder search + BM25 + weighted RRF;
  full `Điều / Khoản / Điểm + document` citations take an exact-lookup path.
- **Temporal guard:** every query accepts `as_of`; recorded repeal/replacement
  amendments demote affected provisions before top-k truncation.
- **Safe graph queries:** the LLM selects one of three parameterized,
  server-owned read-only templates (article count, signers, amendment history),
  never arbitrary Text-to-Cypher.
- **Grounded generation:** citations are verified against rendered evidence;
  one constrained repair is allowed before a cited fallback is returned.
- **Production basics:** FastAPI, SSE streaming, Redis cache, rate limits,
  API-key option, CORS allow-list, Prometheus metrics, readiness checks,
  non-root Docker image, and isolated Neo4j/Redis integration tests.

## Measured results

| Track | Recall | Precision | MRR | p50 latency | Decision |
|---|---:|---:|---:|---:|---|
| Current hash-verified QA_NLP baseline, k=8 (94 rows) | Recall@8 **0.2766** | P@8 0.0638 | **0.2289** | 0.348s | served default |
| Current QA_NLP GPU reranker, k=8 | Recall@8 0.2606 | P@8 **0.0691** | 0.1838 | 1.362s | opt-in experiment |
| Historical QA_Part2345 baseline, k=5 (200 rows) | Recall@5 **0.6012** | — | 0.5885 | — | historical only |

The reranker improved Precision@8 slightly but reduced Recall@8 and MRR while
adding latency, so it is deliberately **not** the default. The historical k=5
result is retained as historical evidence, not presented as the current
contract. Full commands, caveats, and reproducible artifacts are in
[docs/benchmark.md](docs/benchmark.md).

## Try it locally

Prerequisites: Docker, Python 3.12, and [`uv`](https://docs.astral.sh/uv/).

```bash
cp .env.example .env
# Set NEO4J_PASSWORD. LLM_API_KEY is needed only for generated answers.

docker compose up -d

uv venv
uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -e ".[dev,embed,serve,worker]"

vtlaw scrape verify
vtlaw parse check
vtlaw graph import --wipe
vtlaw embed run
vtlaw api serve
```

Open <http://127.0.0.1:18080/>. The UI exposes retrieval profile, legal date,
sources, route, citation verification, and latency per response.

For an NVIDIA Windows host, the optional CUDA profile and its measured memory
constraints are documented in [docs/benchmark.md](docs/benchmark.md).

## Quality gates

```bash
uv run --no-sync ruff check .
uv run --no-sync pytest -m "not integration" -q
VTLAW_TEST_WIPE=1 uv run --no-sync pytest -m integration -q
```

Integration tests use an isolated, disposable Neo4j database and require the
explicit wipe opt-in. CI also verifies the Docker image imports as a non-root
user.

## Portfolio walkthrough

- [Case study](docs/portfolio-case-study.md): problem, design decisions,
  evidence, and CV bullets.
- [Benchmark report](docs/benchmark.md): evaluation protocol, historical vs
  current metrics, and known limitations.
- [Data provenance](data/README.md): immutable snapshot, labels, licenses, and
  corpus limitations.

## Scope and boundaries

The graph models recorded amendment relationships as a ranking heuristic; it
does not synthesize consolidated legal text or determine real-world legal
validity. The project intentionally excludes live crawling, PDF/OCR ingestion,
and arbitrary LLM-generated Cypher. A verified citation proves provenance to
retrieved evidence, not legal correctness.

## Project layout

```text
src/vtlaw/
  parse/       legal-structure parser and citable UID model
  graph/       Neo4j hierarchy, amendment importer, safe query templates
  embed/       Vietnamese bi-encoder and provenance-aware indexing
  retrieve/    exact citation, hybrid search, router, temporal safeguards
  generate/    grounding, citation verification, evidence fallback
  api/         FastAPI, SSE chat UI, operational middleware
  eval/        retrieval metrics, latency, provenance artifacts
tests/         offline unit tests and isolated integration tests
data/          versioned corpus snapshot, amendment annotations, QA labels
```

## Licence and attribution

Project code is MIT. The fixed corpus, labels, and amendment annotations have
their own provenance and licence constraints; see [data/README.md](data/README.md).
