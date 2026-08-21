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

All results below are full-track, `k=8`, `git_dirty: false` artifacts from
commit `19600d3`. The supplied QA labels carry no date of fact, so this is a
**label-retrieval** benchmark: temporal filtering and amendment demotion are
disabled only while scoring labels. The API/UI still apply the selected `as_of`
date and use the temporal graph in production.

| Track / profile | Recall@8 | Precision@8 | MRR@8 | p50 latency | Decision |
|---|---:|---:|---:|---:|---|
| QA_NLP (94), hybrid baseline | 0.8121 | 0.1649 | 0.6495 | 0.102s | reference |
| QA_NLP (94), query composition | **0.9025** | **0.1769** | **0.7061** | 1.429s | default quality profile |
| QA_NLP (94), GPU cross-encoder | 0.7261 | 0.1503 | 0.4715 | 0.993s | experimental; rejected as default |
| QA Part 2–5 (200), hybrid baseline | 0.6002 | 0.1888 | 0.5185 | 0.080s | reference |
| QA Part 2–5 (200), query composition | **0.6691** | **0.2037** | **0.5655** | 1.482s | confirms default choice |

Query composition improves Recall@8 on both independent tracks; it costs about
1.3–1.4s p50 because it makes an LLM call. The real CUDA reranker loses to the
baseline on every reported QA_NLP metric, so it remains visible as an
experiment rather than being enabled for résumé theatre. Commands, artifacts,
and evaluation limits are in [docs/benchmark.md](docs/benchmark.md).

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
