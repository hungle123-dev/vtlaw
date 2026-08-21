# Retrieval benchmark

## What this measures

The fixed QA labels identify provision UIDs but provide no date of fact. A
benchmark without `--as-of` therefore measures whether retrieval finds those
labels: it keeps the graph hierarchy, exact-citation routing, weighted-RRF
fusion, and optional query composition, but disables document-date filtering
and AMENDS demotion. It is not a current-law applicability score.

The deployed API is different by design: the UI supplies a local current date
and callers may provide `as_of`; that path retains temporal filtering and
amendment-aware demotion. Use a dated QA set before reporting a temporal
Recall@k score.

## Clean, reproducible ablation — 2026-08-21

All five artifacts below were generated from commit `19600d3` with
`git_dirty: false`, `k=8`, the 12-document snapshot (7,381 provisions, 440
`AMENDS` edges), `bkai-foundation-models/vietnamese-bi-encoder`, and weighted
RRF (vector : BM25 = 3 : 1). No row was skipped.

| Track / configuration | Recall@8 | Precision@8 | MRR@8 | p50 / p95 |
|---|---:|---:|---:|---:|
| QA_NLP (94), hybrid | 0.8121 | 0.1649 | 0.6495 | 0.102s / 0.256s |
| QA_NLP (94), hybrid + query composition | **0.9025** | **0.1769** | **0.7061** | 1.429s / 2.461s |
| QA_NLP (94), hybrid + CUDA cross-encoder | 0.7261 | 0.1503 | 0.4715 | 0.993s / 1.437s |
| QA Part 2–5 (200), hybrid | 0.6002 | 0.1888 | 0.5185 | 0.080s / 0.123s |
| QA Part 2–5 (200), hybrid + query composition | **0.6691** | **0.2037** | **0.5655** | 1.482s / 2.828s |

Query composition is the `quality` profile because it improves recall and MRR
on both tracks. It uses `gpt-4o-mini` at temperature 0 to produce additional
standalone queries, preserves the original query, and fuses all legs using the
same retriever. It is intentionally a quality/latency trade-off, not a hidden
optimization.

The cross-encoder was a real RTX 3050 CUDA run with
`AITeamVN/Vietnamese_Reranker`; it regressed every reported QA_NLP metric, so
the `rerank` profile remains opt-in. It is not combined with the default
quality profile.

### Evidence artifacts

- [QA_NLP baseline](../data/evaluation/results/2026-08-21-graphrag-baseline-qa-nlp-k8.json)
- [QA_NLP query composition](../data/evaluation/results/2026-08-21-graphrag-composition-qa-nlp-k8.json)
- [QA_NLP CUDA reranker](../data/evaluation/results/2026-08-21-graphrag-rerank-gpu-qa-nlp-k8.json)
- [QA Part 2–5 baseline](../data/evaluation/results/2026-08-21-graphrag-baseline-qa-part2345-k8.json)
- [QA Part 2–5 query composition](../data/evaluation/results/2026-08-21-graphrag-composition-qa-part2345-k8.json)

## Re-run commands

```bash
# Label-retrieval protocol: no --as-of because these QA files have no date of fact.
vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 8
vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 8 --decompose
vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 8 --rerank

vtlaw eval run data/evaluation/qa/QA_Part2345.csv --top-k 8
vtlaw eval run data/evaluation/qa/QA_Part2345.csv --top-k 8 --decompose

# Temporal safety audit: use only a QA set whose facts have the stated date.
vtlaw eval run path/to/dated-qa.csv --top-k 8 --as-of 2026-08-21
```

The CUDA reranker requires a compatible GPU build of PyTorch. On constrained
hosts, `RERANK_MIN_AVAILABLE_MEMORY_MB=256` was used only to permit this
measured GPU experiment; it is not a claim that 256 MB is a safe production
capacity.

## Limits

- Recall@k, Precision@k, and MRR rate retrieval against historical labels;
  they do not prove the legal conclusion is correct.
- Query composition calls an external LLM. Two of 200 composition requests
  returned `[]`; the decomposer kept the original query and the run completed
  with zero skipped rows. Future work should use a dated, adjudicated test set
  and report run-to-run variance.
- The amendment graph is a ranking safeguard over recorded relationships. It
  does not reconstruct consolidated legal text or determine legal validity.
- Citation verification proves that a cited provision was in rendered evidence;
  it does not fact-check the legal reasoning.
