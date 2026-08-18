# Retrieval benchmark — 2026-08-18

This is the reproducible retrieval baseline for the current worktree. It is not
a claim of legal validity and it is not a comparison to prior JSON files: the
embedding fingerprint and evaluation contract changed, so older numbers are not
an apples-to-apples baseline.

## Conditions

| Item | Value |
|---|---|
| Corpus | 12 NLP-LegalQA snapshot documents; 7,381 provisions; 440 AMENDS edges |
| Snapshot manifest SHA-256 | `1ec811da809d03036d4923ebaa28bc720fde8983e6dd5b566a000584c8d77672` |
| Retrieval | vector + BM25 + weighted RRF; `rrf_k=10`, vector/BM25 weight `3:1` |
| Candidate/result depth | 30 candidates per leg, report top 5 |
| Legal date | `2026-08-18` |
| Embedder | `bkai-foundation-models/vietnamese-bi-encoder` at `84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0` |
| Embedding fingerprint | `…|pyvi-ViTokenizer|v2` |
| Reranker / decomposition | disabled |

Commands:

```bash
vtlaw eval run data/evaluation/qa/QA_NLP.csv \
  --top-k 5 --as-of 2026-08-18 \
  --output data/evaluation/results/2026-08-18-qa-nlp.json

vtlaw eval run data/evaluation/qa/QA_Part2345.csv \
  --top-k 5 --as-of 2026-08-18 \
  --output data/evaluation/results/2026-08-18-qa-part2345.json
```

## Results

| NLP-LegalQA reporting track | Rows | Recall@1 | Recall@3 | Recall@5 | MRR | p50 / p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| `QA_NLP.csv` | 94 | 0.4973 | 0.7420 | 0.7713 | 0.6449 | 0.181s / 0.265s |
| `QA_Part2345.csv` | 200 | 0.3728 | 0.5110 | 0.5620 | 0.5106 | 0.217s / 0.308s |

`QA_NLP` p99 was 6.075s because the first retrieval loaded the local embedding
model; the remaining latency statistics are the steady-state request path.

## Interpretation and limits

- The tracks are reported separately. `QA_Part2`–`QA_Part5` overlap with the
  larger Part files, so they are not averaged or presented as separate tests.
- Labels name historical provision UIDs but do not provide a date-of-fact. A
  higher label-retrieval score can disagree with the most current rule after an
  amendment; `as_of` and the AMENDS annotations handle only the evidence the
  corpus actually contains.
- These are retrieval metrics, not grounded-answer faithfulness metrics. LLM
  generation, reranking, and query decomposition need their own separately
  recorded experiment before any claim that they improve the baseline.
- Result JSON files are intentionally ignored by Git. Re-run the commands after
  any corpus, model, index, or retrieval-configuration change and update this
  report only from those generated artifacts.
