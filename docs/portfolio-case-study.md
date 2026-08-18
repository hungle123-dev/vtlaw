# vtlaw — Portfolio case study

## One-line summary

Built a reproducible Vietnamese legal Graph RAG on the fixed NLP-LegalQA
corpus: citable legal hierarchy, amendment relations, date-scoped retrieval,
exact citation lookup, hybrid retrieval, and an operational API.

## Problem

Legal retrieval is not only semantic search. A useful answer must preserve the
source provision, distinguish a specific legal citation from a topical query,
and avoid presenting a future or expired document as applicable on the selected
date.

## What I built

- Parsed 12 NLP-LegalQA documents into 7,381 stable `Document → Article →
  Clause → Point` graph nodes in Neo4j.
- Imported 440 resolved amendment edges and date-bounded their use in temporal
  retrieval.
- Added exact lookup for a full `Điều / Khoản / Điểm + document` citation;
  ambiguous questions use vector + BM25 weighted-RRF retrieval instead.
- Made vector provenance explicit: pinned model revisions and an embedding
  fingerprint force refresh when source text or embedding inputs change.
- Exposed the system through FastAPI with health checks, caching, request IDs,
  rate limiting, optional API-key protection, and a non-root Docker image.
- Added reproducible evaluation outputs containing corpus, model, retrieval
  configuration, dataset hash, and `as_of` date.

## Evidence

| Check | Result |
|---|---|
| Corpus / graph | 12 documents, 7,381 provisions, 440 `AMENDS` edges |
| Embedding coverage | 566 Articles, 2,763 Clauses, 4,052 Points |
| QA_NLP | Recall@5 0.7713; MRR 0.6449 |
| QA_Part2345 | Recall@5 0.5620; MRR 0.5106 |
| Quality gates | offline tests, 15 isolated Neo4j integration tests, Ruff, Docker build/import/non-root check |

The two QA tracks are reported separately because the Part datasets overlap.
The full evaluation conditions and limitations are in
[benchmark.md](benchmark.md).

## 90-second demo

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
```

Point out that the exact citation resolves to one provision without approximate
search, while a plain-language question follows hybrid retrieval. Show
`docs/benchmark.md` rather than claiming an unrecorded accuracy improvement.

## Honest boundary

This is a production-minded portfolio system, not a legal-advice product. It
uses only the fixed NLP-LegalQA artifacts. Amendment instructions are modeled
as graph evidence, but the project does not fabricate consolidated legal text
or claim that retrieval labels establish legal applicability without a
date-of-fact.

## CV bullets

- Built a Vietnamese legal Graph RAG over 7.4K citable provisions, combining
  Neo4j hierarchy/amendment relations with exact-citation and hybrid retrieval.
- Designed reproducible retrieval evaluation with pinned embedding provenance,
  temporal `as_of` filtering, and Recall@k/MRR reporting.
- Productionized a FastAPI RAG service with Redis caching, observability,
  destructive-test safeguards, Docker non-root execution, and CI validation.
