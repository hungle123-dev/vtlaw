# vtlaw — Vietnamese traffic-law graph RAG

Question answering over Vietnamese road-traffic law, with citations down to the
`Điểm / Khoản / Điều` and awareness of which law was in force when.

Built on the domain groundwork of [`n3sfan/NLP-LegalQA`](https://github.com/n3sfan/NLP-LegalQA)
(MIT, © Le Truong Thinh) — used with permission as a reference and as the source
of the committed corpus. The implementation here is independent.

## Status

| Stage | | State |
|---|---|---|
| 1 | `scrape` — source API → snapshot | done, 20 tests |
| 2 | `parse` — text → provisions | done, 68 tests |
| 3 | `graph` — provisions → Neo4j | done, 25 tests |
| 4 | `embed` — vectors + indexes | done, 7,381/7,381 embedded |
| 5 | `retrieve` — hybrid search + rerank | done; rerank off by default, see below |
| 6 | `generate` — answer with citations | done, verified against a live server |
| 7 | `api` — HTTP surface | done, `/chat` `/health` `/metrics` |
| 8 | `eval` — recall@k, MRR, latency | done, 94-question dataset |

265 tests pass (252 offline, 13 needing Neo4j).

Four modules are written but wired to nothing: `QueryRouter`, `QueryDecomposer`,
`QueryRewriter`, `TextToCypher`. They have tests and no callers. The async
worker in `ingest/worker.py` does not run — it is scaffolding, not working code.
There is no CI and no frontend.

## Measured retrieval quality

94 questions, `data/evaluation/qa/QA_NLP.csv`, hybrid strategy:

| | recall@1 | recall@5 | MRR | p50 latency |
|---|---|---|---|---|
| hybrid | 0.359 | 0.598 | 0.473 | 0.24s |
| hybrid + cross-encoder rerank | 0.255 | 0.609 | 0.421 | 17.0s |

**Rerank is off by default because these numbers do not justify it.** It helps 10
questions and hurts 11. The pattern in the regressions is legible: the reranker
promotes `168/2024/NĐ-CP` above `100/2019/NĐ-CP`, which is legally correct — the
2024 decree replaced the 2019 one — but the dataset is labelled against the older
law, so the reranker is penalised for being right. Re-measure after the labels
are reconciled.

Two caveats that apply to every number above:

* `is_relevant` matches by UID prefix, so retrieving any descendant of a
  referenced Article counts as a hit. This inflates recall for Article-level
  references.
* 85 of the 94 questions have their ground truth in one document
  (`100/2019/NĐ-CP`). This measures coverage of one decree, not of the corpus.

The first-stage ceiling is the real constraint: at k=30 recall is 0.758, so
**24% of questions have no correct provision anywhere in 30 candidates**.
No amount of reranking reaches those.


## Why a graph

A legal citation *is* a path through a hierarchy, and a Point on its own is not an
answer. Take `NĐ 168/2024` Điều 6 Khoản 3 điểm a:

```
Point  a  "Điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h"
Clause 3  "Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển…"
```

The offence is in the Point; the amount is in its parent Clause. Retrieval that
returns the Point alone reads like an answer while omitting the penalty. Walking
up the hierarchy is a correctness requirement, not a convenience — which is what
the graph buys.

## Quick start

```bash
docker compose up -d                 # Neo4j on 127.0.0.1:17687, Redis on 16379
cp .env.example .env                 # set NEO4J_PASSWORD (8+ chars)
uv venv && uv pip install -e ".[dev,embed,serve]"

vtlaw scrape verify                  # hash-check the committed snapshot
vtlaw parse check                    # 12 documents -> 7,381 provisions
vtlaw graph import --wipe            # write to Neo4j
vtlaw graph import-amends            # amendment edges from data/amends
vtlaw embed run                      # embed provisions missing a vector
vtlaw graph status                   # read the counts back
```

Then ask it something:

```bash
vtlaw retrieve search "không đội mũ bảo hiểm phạt bao nhiêu"
vtlaw generate answer "không đội mũ bảo hiểm phạt bao nhiêu"   # needs LLM_API_KEY
vtlaw api serve                                                 # http://127.0.0.1:18080
vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 5
```

`vtlaw embed run` is idempotent — it skips nodes that already carry a vector, so
a re-run after an interruption resumes rather than recomputing.

Inspect a single provision with its ancestors:

```bash
vtlaw parse show "168/2024/NĐ-CP::article::6::clause::3::point::a"
```

## Layout

```
src/vtlaw/
├── scrape/     stage 1 — the only stage that touches the network
├── parse/      stage 2 — Điều / Khoản / Điểm extraction
├── graph/      stage 3 — Neo4j schema, import, AMENDS edges
├── embed/      stage 4 — vietnamese-bi-encoder vectors
├── retrieve/   stage 5 — vector + BM25 → RRF → rerank → legal heuristic
├── generate/   stage 6 — graph-walk context → LLM with citations
├── api/        stage 7 — FastAPI
├── eval/       stage 8 — recall@k, MRR, latency
└── ingest/     async worker — does not run yet

data/           committed corpus — see data/README.md
tests/          one directory per stage
```

Each stage reads what the previous one wrote and is runnable on its own. Only
`scrape` needs the network, so everything after it replays offline from the
committed snapshot.

## Testing

```bash
pytest -m "not integration"    # 252 tests, no services needed
pytest -m integration          # 13 tests, needs Neo4j running
```

Integration tests wipe the database. Do not point them at data you care about.


## Design notes

**The snapshot is the boundary.** The source API is a public government service
with no SLA, and it went down mid-development. Committing a hash-verified snapshot
means every stage after acquisition is reproducible without it. See
[`data/README.md`](data/README.md) for provenance and measured defects.

**HTTP 200 is not success.** The source answers `200` with a non-null `error`
field when its backend fails. `raise_for_status()` passes that, so the client
validates the envelope — otherwise an upstream failure is recorded as "no more
documents".

**A refusal is not a fault.** On `403` the client stops rather than retrying or
working around it.

**Counts are read back, not trusted.** `vtlaw graph import` compares the parse
count, the import count, and a count queried from the database. Agreement is the
check; a write that reports success without landing fails it.

**Duplicate numbering does not merge.** When a document numbers two distinct
provisions identically, both are kept under distinct UIDs. Measured: this recovers
2 provisions in `118/2025/QH15` that a naive UID scheme loses.

**A silent fallback hid a dead reranker.** `rerank` catches
`(ImportError, ValueError, OSError)` and returns the unranked hits. That is the
right behaviour — retrieval should degrade, not fail — but the reranker raised
`ValueError: Unrecognized processing class` on every call for want of
`sentencepiece`, logged one WARNING, and the pipeline reported success while
ranking nothing. The fallback is still there; the dependency is now declared.

## Licence

MIT.
