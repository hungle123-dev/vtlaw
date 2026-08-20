# Retrieval benchmark — 2026-08-20

The reproducible retrieval baseline for the current worktree. Not a claim of
legal validity, and not a comparison to older JSON files: the embedding
fingerprint and evaluation contract have changed since, so earlier numbers are
not an apples-to-apples baseline.

## Conditions

| Item | Value |
|---|---|
| Corpus | 12 NLP-LegalQA snapshot documents; 7,381 provisions; 440 AMENDS edges |
| Snapshot manifest SHA-256 | `1ec811da809d03036d4923ebaa28bc720fde8983e6dd5b566a000584c8d77672` |
| Retrieval | vector + BM25 + weighted RRF; `rrf_k=10`, vector/BM25 weight `3:1` |
| Candidate/result depth | 30 candidates per leg, 4× overfetch before the date filter, report top 5 |
| Legal date | `2026-08-20` |
| Embedder | `bkai-foundation-models/vietnamese-bi-encoder` at `84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0` |
| Embedding fingerprint | `…\|pyvi-ViTokenizer\|v2` |
| Decomposition LLM | `gpt-4o-mini` via `api.openai.com/v1`, `temperature=0` |
| Reranker | disabled (see below) |
| Rows dropped by a retrieval error | 0 on every run |

Commands:

```bash
vtlaw eval run data/evaluation/qa/QA_NLP.csv \
  --top-k 5 --as-of 2026-08-20 \
  --output data/evaluation/results/2026-08-20-qa-nlp-baseline.json

vtlaw eval run data/evaluation/qa/QA_Part2345.csv \
  --top-k 5 --as-of 2026-08-20 \
  --output data/evaluation/results/2026-08-20-qa-part2345-baseline.json
```

Add `--decompose` for the decomposition profile.

## Deterministic baseline

| NLP-LegalQA reporting track | Rows | Recall@1 | Recall@3 | Recall@5 | P@1 | P@3 | MRR | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `QA_NLP.csv` | 94 | 0.4973 | 0.7420 | 0.7713 | 0.5106 | 0.2730 | 0.6449 | 0.189s / 0.500s |
| `QA_Part2345.csv` | 200 | 0.3728 | 0.5110 | 0.5637 | 0.4100 | 0.2083 | 0.5127 | 0.180s / 0.276s |

This path is bit-exactly reproducible: every figure above matches the
2026-08-18 run to four decimal places, on a graph re-read from the same
snapshot. That is the point of keeping the default free of LLM calls.

`QA_NLP` p99 was 7.466s because the first retrieval loads the local embedding
model; the p50/p95 columns are the steady-state request path.

## Selected NLP-LegalQA techniques — measured experiments

The upstream project contains more techniques than should be enabled by
default. Each was evaluated against the same snapshot and legal date instead of
being added as a portfolio checklist item.

### LLM query decomposition

`gpt-4o-mini` produces extra standalone phrasings; the original query is kept as
its own leg and all result lists are fused through the existing weighted RRF.
Full-track E2E runs, not a five-question smoke test.

| Track | Rows | Recall@1 | Recall@3 | Recall@5 | MRR | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| `QA_NLP.csv` hybrid + decomposition | 94 | 0.4849 | 0.7757 | 0.8599 | 0.6757 | 1.652s / 2.663s |
| `QA_Part2345.csv` hybrid + decomposition | 200 | 0.3782 | 0.5630 | 0.6192 | 0.5637 | 1.542s / 2.725s |

Recall@5 improves on both tracks (+0.0886 and +0.0555) for roughly 1.4s of
added p50. Recall@1 moves in opposite directions on the two tracks — the extra
phrasings surface more of the labelled provisions overall while sometimes
displacing the single best hit.

**This profile is not reproducible to the digit.** Even at `temperature=0` the
provider is not deterministic. Two runs of the identical command on identical
data:

| Track | 2026-08-18 Recall@5 | 2026-08-20 Recall@5 | Spread |
|---|---:|---:|---:|
| `QA_NLP.csv` | 0.8741 | 0.8599 | 0.0142 |
| `QA_Part2345.csv` | 0.6211 | 0.6192 | 0.0019 |

So the honest claim is "Recall@5 ≈ 0.86–0.87 on QA_NLP, ±0.015 run to run",
and a single decomposition number should never be quoted as a fixed
capability. This is exactly why the API keeps a deterministic default and
exposes decomposition as an explicit, separately-measured profile.

### Cross-encoder reranker

`AITeamVN/Vietnamese_Reranker` was tested with identical 20-candidate inputs
(2026-08-18 run; not re-measured on 2026-08-20 because the decision stands).

| Track | Baseline Recall@5 / MRR / p50 | Reranked Recall@5 / MRR / p50 | Decision |
|---|---|---|---|
| `QA_NLP.csv` (94) | 0.7713 / 0.6271 / 0.149s | 0.6605 / 0.4839 / 7.556s | reject as default |
| `QA_Part2345.csv` (200) | 0.5628 / 0.5163 / 0.134s | 0.5887 / 0.5539 / 8.753s | retain as experiment |

The effect changes sign by track and costs ~50× the latency. The UI/API label
`Compose + rerank` is therefore marked experimental: it has a live E2E contract
test, but no full-track claim that it beats decomposition alone.

### Citation currency — a generation defect Recall@k cannot see

The screenshots in the README exposed a failure the retrieval metrics rate as a
success. Both `100/2019/NĐ-CP` and `168/2024/NĐ-CP` are in effect on 2026-08-20
with no expiry date, so the temporal filter keeps both, and retrieval returns
provisions from each. Recall@5 counts that as a hit — the labelled provision was
found. The model then cited the 2020 decree.

Nothing existing catches it. The date filter cannot separate two documents that
are both in effect. Amendment demotion only demotes what the annotations cover,
and `168/2024` has 199 `AMENDS` edges into `100/2019` but none onto the clause
that was cited. Citation verification says `verified`, correctly: the citation
*is* in the evidence. It checks provenance, not currency.

So it needed its own measurement. `scripts/measure_citation_currency.py` scores
the questions where the defect is possible at all — evidence spanning two or
more penalty decrees — and asks whether the answer cited only a superseded one:

| Change | Contested questions | Cited only a superseded decree | Rate |
|---|---:|---:|---:|
| Starting point | 31 | 15 | 0.484 |
| Newest decree named explicitly before the evidence | 31 | 7 | 0.226 |
| Ancestor hierarchy actually reaching the prompt | 31 | **2** | **0.065** |

Two separate causes, found in that order.

**The evidence list is ordered by retrieval score.** The effective date was
already in every provision header, but the model anchored on hit #1 and cited
whatever decree ranked first. Stating the ordering as a fact read from the graph,
rather than leaving it to be inferred, took 0.484 to 0.226.

**The ancestor hierarchy was silently missing.** `fetch_hierarchy` walks UP from
a hit to build `Điều → Khoản → Điểm`, and the parent Clause is where a penalty
amount lives. Its Cypher returned `nodes(path)`, but read through
`Result.data()` a Node arrives as a bare property dict with **no labels at all** —
so every `label == "Article"` / `"Clause"` test fell through and the whole path
was dropped. Each provision reached the prompt as its own one-line content and
nothing else. The model had an offence with no amount attached, so it borrowed a
number from whichever nearby clause looked plausible; that is how
`10.000.000 – 14.000.000 đồng`, a Khoản 9 aggregate covering other violations,
attached itself to a red-light question. Projecting
`{labels: labels(n), props: properties(n)}` server-side took 0.226 to 0.065.

The unit tests passed throughout, because their fixtures flattened labels
alongside the properties — a shape Neo4j never sends. That is the second test in
this project found to pass for the wrong reason.

A `Luật` and a `Nghị định` are not compared: they are different instruments, not
two versions of one rule. A question that names a decree ("Nghị định
100/2019/NĐ-CP quy định gì?") is asking about that document, so citing it back
is correct rather than stale — scoring it as stale was a false positive in the
first version of this measurement.

The remaining 0.065 is two questions about behaviours the newer decree does not
clearly cover, where the older text may even be the right answer, and the corpus
holds no consolidated version to arbitrate. Recorded, not claimed as fixed.
Artifacts: `2026-08-20-citation-currency-{before,after,hierarchy-fixed}.json`.

## Interpretation and limits

- The tracks are reported separately. `QA_Part2`–`QA_Part5` concatenate exactly
  into `QA_Part2345` — same 200 questions, same order, 50 rows each — so they
  are not averaged or presented as separate tests. `QA_NLP` shares no question
  with them.
- Cutoffs beyond the requested `--top-k` are not reported. Earlier artifacts
  carried `recall@7` and `recall@10` for a `--top-k 5` run; those equalled
  `recall@5` and read as a plateau when they were only list truncation.
- Rows dropped by a retrieval error are recorded in `skipped_rows` and excluded
  from every average, so a score always states its own denominator. All four
  runs above dropped nothing.
- Comparison to the upstream project (README) uses the same 200 rows and the
  same embedding model, but the two relevance predicates differ: upstream tests
  `retrieved.startswith(reference)`, this harness additionally requires the
  match to land on a `::` UID boundary. Re-scored on these artifacts the two
  agree to four decimal places, so the comparison is not an artifact of the
  predicate — but they are not the same test. Upstream's numbers are quoted from
  its committed `eval_results/`; they were not re-measured here.
- Labels name historical provision UIDs but do not provide a date-of-fact. A
  higher label-retrieval score can disagree with the most current rule after an
  amendment; `as_of` and the AMENDS annotations handle only the evidence the
  corpus actually contains.
- These are retrieval metrics, not a measure of legal correctness or
  grounded-answer faithfulness. The API performs citation-provenance checking,
  but that only verifies that an answer's citation appears in retrieved
  evidence; it does not validate the substantive legal conclusion.
- Result JSON files are intentionally ignored by Git. Re-run the commands after
  any corpus, model, index, or retrieval-configuration change and update this
  report only from those generated artifacts.
