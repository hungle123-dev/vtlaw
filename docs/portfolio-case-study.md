# vtlaw — Portfolio case study

## One line

An amendment-aware Vietnamese legal GraphRAG that retrieves 30 candidates per
leg, fuses a 30-item candidate pool, and returns 8 contexts. Its measured
quality profile uses query composition; its AMENDS behavior remains explicitly
a ranking heuristic, not consolidated-law reconstruction.
## Problem

Legal retrieval is not semantic search with extra steps. Three things break a
naive RAG on Vietnamese traffic law:

1. **A provision alone is not an answer.** A Point states the offence; the
   penalty amount lives in its parent Clause. Return the Point and you have
   something that reads like an answer with the number missing.
2. **A citation is not a query.** "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP" has
   exactly one correct answer. Approximate search on it is a bug.
3. **Law has dates.** A document not yet in effect, or already expired, must
   not be presented as applicable — and an amendment recorded in 2025 says
   nothing about a question asked as of 2024.

## What I built

- Parsed 12 fixed-corpus documents into 7,381 stable `Document → Article →
  Clause → Point` nodes in Neo4j, handling the cases a line-matching parser
  gets wrong: quoted amendment text is content, not structure (2,123 lines
  across the corpus, where a bare `Điều 5.` names another law rather than
  opening a section here), letter-suffixed numbering is real legal text
  (`Điều 18a`, 18 such headers — all of them inside quoted blocks on this
  corpus, so the branch exists for the next document, not this one), and
  duplicate numbering must not merge (2 clauses).
- Imported 440 resolved amendment edges and date-bounded their effect on
  ranking by the amending document's own effective date.
- Added exact lookup for a full `Điều / Khoản / Điểm + document` citation;
  ambiguous questions fall through to vector + BM25 weighted-RRF retrieval.
- Weighted the fusion legs by measured quality (vector 3 : BM25 1). Equal
  weights let BM25's ordering drag the fused list *below* plain vector search.
- Measured query composition on both supplied QA tracks and made it the quality
  profile only after it improved Recall@8 and MRR@8 on both; its LLM latency is
  exposed to the user rather than hidden.
- Measured the cross-encoder reranker on an RTX 3050 CUDA run, then **rejected
  it as default** because it regressed Recall@8, Precision@8, and MRR@8 against
  the same baseline. It remains an explicit experimental profile in the UI.
- Added the upstream four-way intent contract plus multi-turn rewriting. The
  graph-query feature safely selects among three parameterized, server-owned,
  read-only templates; it never executes arbitrary Text-to-Cypher.
- Constrained generation to retrieved evidence: verify each cited provision
  against the hits, attempt one evidence-only repair, then return a cited
  fallback rather than an unverifiable legal claim.
- Made vector provenance explicit: an embedding fingerprint over
  model@revision + segmenter + text version forces a refresh when any input
  changes, so an index can never silently disagree with its weights.
- Exposed it through FastAPI with real dependency health checks, Redis caching,
  request IDs, sliding-window rate limiting, optional API-key auth, and a
  non-root Docker image.
- Served a dependency-free responsive chat UI showing evidence, legal date,
  profile, router decision, citation status, and per-stage timings.

## Evidence

| Check | Result |
|---|---|
| Corpus / graph | 12 documents, 7,381 provisions, 440 `AMENDS` edges |
| Embedding coverage | 566 Articles, 2,763 Clauses, 4,052 Points — 100% |
| QA_NLP label retrieval, baseline → composition (94) | Recall@8 0.8121 → **0.9025**; MRR 0.6495 → **0.7061** |
| QA Part 2–5 label retrieval, baseline → composition (200) | Recall@8 0.6002 → **0.6691**; MRR 0.5185 → **0.5655** |
| Reproducibility | five full-track artifacts at commit `19600d3`, `git_dirty: false`, zero skipped rows |
| CUDA reranker QA_NLP k=8 | Recall@8 0.7261 / Precision@8 0.1503 / MRR@8 0.4715 / p50 0.993s |
| Reranker decision | rejected as default: it loses to baseline on every reported QA_NLP metric |
| Temporal safety | API/UI retain `as_of` filtering and AMENDS demotion; labels without a date are never used as a temporal-quality score |
| Citation-currency investigation (historical) | 16/31 → 8/31 → 2/31 |
| Latency | baseline 0.080–0.102s p50; composition 1.429–1.482s p50 |
| Quality gates | offline tests, isolated Neo4j integration suite, Ruff, Docker build/import/non-root check |

The two QA tracks are reported separately because `QA_Part2`–`QA_Part5`
concatenate exactly into `QA_Part2345`. Full conditions and limits:
[benchmark.md](benchmark.md).

## Engineering decisions worth defending

| Decision | Why |
|---|---|
| Query composition as the quality profile | On clean full-track ablations it improved Recall@8, Precision@8, and MRR@8 on both supplied QA tracks; latency is the declared cost. |
| Rejected the reranker despite it being the obvious "add a reranker" move | The current GPU run regressed every reported QA_NLP metric, so enabling it by default would be a checklist item, not an improvement. |
| Safe read-only graph-template selection instead of Text-to-Cypher | An LLM writing Cypher against a live database is an arbitrary-query primitive. Three bounded, server-owned templates cover the questions users actually ask. |
| Citation verification with one repair, then fallback | An unsupported legal citation is worse than no answer. The fallback cites sources and says it could not verify. |
| Score-span-relative amendment demotion | An absolute penalty of −5.0 against an RRF list (span ~0.3) does not demote a hit, it replaces the ranking. |
| Demote before output truncation | A current candidate at rank 9–30 cannot be promoted if top-8 is cut first; the amendment-aware heuristic sees the full 30-candidate pool. |
| Report only cutoffs the run measured | A `--top-k 5` run reporting recall@10 shows a plateau that is really truncation. |
| Built a new metric rather than trusting Recall@k | Two decrees both in effect, both retrieved: Recall@5 scores it a hit while the answer cites the superseded one. A defect no existing metric can see needs its own measurement, not a hunch. |

## 90-second demo

![Hybrid retrieval, evidence panel, and the citation-provenance verdict](images/ui-hybrid-answer.png)

```powershell
docker compose up -d
vtlaw graph status
vtlaw embed status
vtlaw retrieve search "Điểm a Khoản 3 Điều 6 168/2024/NĐ-CP"
vtlaw api serve
```

In a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:18080/health
Start-Process http://127.0.0.1:18080/
```

The UI ships four example questions, one per path. Point out that the exact
citation resolves to one provision with no approximate search, while a
plain-language question follows hybrid retrieval. Then ask
`Nghị định 168/2024/NĐ-CP có bao nhiêu điều?` for the safe graph-template
route, and a follow-up to show the rewritten standalone question. Every reply
carries its citation-provenance verdict and per-stage timings.

Screenshots of all four paths are in the [README](../README.md#the-four-retrieval-paths-as-they-render);
regenerate them with `scripts/capture_screenshots.py` against a running API.

## Honest boundary

Taking my own screenshots found a bug the retrieval metrics rate as a success.
Two penalty decrees, `100/2019/NĐ-CP` and `168/2024/NĐ-CP`, are both in effect
with no expiry date; retrieval returns provisions from each, Recall@5 counts the
hit, and the model cited the 2020 one. The date filter cannot separate two live
documents, amendment demotion only covers annotated edges, and citation
verification checks provenance rather than currency — so no existing layer was
even looking.

I measured it instead of guessing. On 2026-08-20 the three artifacts record
**16/31 → 8/31 → 2/31**:
`2026-08-20-citation-currency-before.json`,
`2026-08-20-citation-currency-after.json`, and
`2026-08-20-citation-currency-hierarchy-fixed.json`. Naming the newest decree
explicitly in the prompt took 16 to 8; chasing those 8 found the real cause. The whole
ancestor hierarchy was missing from the prompt: `nodes(path)` read through the
driver's `Result.data()` loses the node labels, so every label test fell through
and the parent Clause holding the penalty amount never arrived. The model had an
offence with no amount attached and borrowed a plausible-looking number from a
neighbouring clause. Fixing the projection took the rate to 0.065.

The unit tests passed the whole time, because their fixtures used a node shape
Neo4j never sends. The remaining 2 questions are behaviours the newer decree does
not clearly cover, and the corpus holds no consolidated text to arbitrate. That
number is recorded, not rounded away.

This is a production-minded portfolio system, not a legal-advice product. It uses
only the fixed, versioned corpus artifacts. Amendment instructions are modeled as graph
evidence, but the project does not fabricate consolidated legal text or claim
that retrieval labels establish legal applicability without a date-of-fact.
Citation verification checks provenance, not legal correctness. The graph API
intentionally does not execute arbitrary LLM-generated Cypher. Upstream's
published numbers are quoted from its committed `eval_results/`, not re-measured
here.

## CV bullets

- Built an amendment-aware Vietnamese legal GraphRAG heuristic over 7,381 citable provisions in Neo4j —
  exact-citation resolution, weighted-RRF hybrid retrieval, date-scoped
  temporal filtering, multi-turn resolution, and parameterized read-only graph
  safe read-only template selection instead of arbitrary Text-to-Cypher.
- Improved Recall@8 from 0.6002 to 0.6691 and MRR@8 from 0.5185 to 0.5655 on a
  200-question Vietnamese traffic-law track through measured query composition;
  added a date-aware evaluation contract so historical reference UIDs are not
  mistaken for a current-law score.
- Designed an evaluation harness recording corpus SHA-256, pinned embedding
  revision, fusion weights, `as_of` date, dropped rows, Git dirty state, and
  superseded-label counts; separated deterministic current artifacts from
  historical LLM experiments rather than quoting their best run.
- Productionized the service with FastAPI, Redis caching keyed on retrieval
  provenance, Prometheus metrics, request-scoped tracing, rate limiting,
  citation-provenance guard with repair-then-fallback, non-root Docker, and CI
  running lint, offline tests, isolated Neo4j/Redis integration tests, and an
  image import check.
