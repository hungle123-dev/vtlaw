# vtlaw — Portfolio case study

## One line

An amendment-aware Vietnamese legal GraphRAG whose current contract fetches 30
candidates per retrieval leg, fuses a 30-item candidate pool, and returns 8
contexts. Its AMENDS behavior is explicitly a ranking heuristic, while
historical k=5 measurements remain labelled as such.
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

- Parsed 12 NLP-LegalQA documents into 7,381 stable `Document → Article →
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
- Retained NLP-LegalQA-inspired query decomposition as an explicit experimental
  profile. Its pre-temporal-pool measurements improved legacy-label recall but
  are kept historical because the current legal-date benchmark has a different
  relevance condition.
- Measured the cross-encoder reranker on both the historical CPU tracks and a
  current RTX 3050 QA_NLP k=8 run, then **rejected it as default**: the current
  run raises Precision@8 slightly but regresses Recall@8 and MRR@8. It stays an
  opt-in experiment, labelled as such in the UI.
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
| Historical 2026-08-20 temporal baseline QA_NLP | Raw-label Recall@5 0.2606 / MRR 0.2261; 65/94 rows reference only superseded provisions |
| Historical 2026-08-20 temporal baseline QA_Part2345 | Recall@5 0.6012 / MRR 0.5885; zero returned top-1 superseded provisions |
| Decomposition QA_NLP / QA_Part2345 | historical pre-temporal-pool experiment; Recall@5 0.8599 / 0.6192 |
| Baseline reproducibility | deterministic profile; historical artifacts record `git_dirty: true` and require a post-commit rerun for a commit-level claim |
| Decomposition reproducibility | ±0.014 Recall@5 run-to-run; reported as a range, not a point |
| vs upstream basic RAG (same 200 rows) | 0.6012 vs 0.4892 Recall@5, zero LLM calls either side |
| Current GPU reranker QA_NLP k=8 | Recall@8 0.2606 / Precision@8 0.0691 / MRR@8 0.1838 / p50 1.362s |
| Reranker decision | rejected as default: baseline Recall@8 0.2766 / MRR@8 0.2289; precision gain does not offset the regression |
| Citation-currency stages (2026-08-20) | 16/31 → 8/31 → 2/31 |
| Latency | 0.348 s p50 current temporal baseline; 1.362 s GPU reranker; historical decomposition 1.5–1.7 s |
| Quality gates | offline tests, isolated Neo4j integration suite, Ruff, Docker build/import/non-root check |

The two QA tracks are reported separately because `QA_Part2`–`QA_Part5`
concatenate exactly into `QA_Part2345`. Full conditions and limits:
[benchmark.md](benchmark.md).

## Engineering decisions worth defending

| Decision | Why |
|---|---|
| Deterministic baseline, LLM as an opt-in profile | A reproducible number needs a pipeline with no sampling in it. Running decomposition twice confirmed it: ±0.014 Recall@5 at `temperature=0`, while the deterministic path repeated to four decimals. |
| Rejected the reranker despite it being the obvious "add a reranker" move | The current GPU run improved Precision@8 but regressed Recall@8 and MRR@8; adding it as default would be a checklist item, not an improvement. |
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
only the fixed NLP-LegalQA artifacts. Amendment instructions are modeled as graph
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
- Raised Recall@5 from 0.4892 (upstream basic RAG) to 0.6012 on the same
  200-question reporting track with zero LLM calls; added a date-aware label
  audit so historical reference UIDs are not mistaken for current-law accuracy.
  Kept decomposition and the cross-encoder as measured experiments rather than
  claiming them as a default improvement.
- Designed an evaluation harness recording corpus SHA-256, pinned embedding
  revision, fusion weights, `as_of` date, dropped rows, Git dirty state, and
  superseded-label counts; separated deterministic current artifacts from
  historical LLM experiments rather than quoting their best run.
- Productionized the service with FastAPI, Redis caching keyed on retrieval
  provenance, Prometheus metrics, request-scoped tracing, rate limiting,
  citation-provenance guard with repair-then-fallback, non-root Docker, and CI
  running lint, offline tests, isolated Neo4j/Redis integration tests, and an
  image import check.
