# Historical retrieval benchmark — 2026-08-20

## Artifact status

The historical deterministic baseline was rerun after the Precision@k correction, the
evaluator's alignment with the served AMENDS heuristic, and the fix that keeps
the candidate pool until temporal demotion completes. Its two historical artifacts
are `2026-08-20-qa-nlp-baseline-current.json` and
`2026-08-20-qa-part2345-baseline-current.json`; both contain zero skipped rows.
They record `git_dirty: true`, so they describe the current working tree rather
than a commit-only release. Commit and rerun before claiming commit-level
reproducibility.

The decomposition, reranker, and citation-currency experiments below remain
historical evidence. These are retrieval measurements, not a claim of legal
validity.

## Current served/evaluated contract — 2026-08-21

Serving and evaluation return/score 8 contexts. `fetch_k=30` fetches 30
candidates per retrieval leg; results are fused into a 30-item `candidate_k`
fused candidate pool. If enabled, the cross-encoder receives that pool as
`rerank_top=30`, then `context_k = top_k = 8` is returned/scored. Reports name
Recall@8, Precision@8, and MRR@8.

Release evidence is `2026-08-21-release-evidence-k8.json`, generated with:

```bash
vtlaw eval run data/evaluation/qa/QA_NLP.csv --top-k 8 --as-of 2026-08-21 \
  --output data/evaluation/results/2026-08-21-release-evidence-k8.json
```

It covers 94 rows with zero retrieval errors and records Recall@8 0.2766,
Precision@8 0.0638, and MRR@8 0.2289. Its dataset and snapshot-manifest hashes
are recorded in the artifact; `git_dirty: true` means it is working-tree
evidence, not a commit-reproducibility claim.

## Current GPU reranker experiment — 2026-08-21

`2026-08-21-qa-nlp-rerank-gpu-k8.json` reruns the same 94 rows, corpus,
`as_of=2026-08-21`, and `k=8` contract with `AITeamVN/Vietnamese_Reranker`.
It used an RTX 3050 through PyTorch CUDA; it is a real cross-encoder run, not
a mocked or CPU-fallback result.

| Configuration | Recall@8 | Precision@8 | MRR@8 | p50 latency |
|---|---:|---:|---:|---:|
| Current hybrid baseline | 0.2766 | 0.0638 | 0.2289 | 0.348s |
| Hybrid + GPU reranker | 0.2606 | 0.0691 | 0.1838 | 1.362s |

The reranker increases Precision@8 by 0.0053, but decreases Recall@8 by
0.0160 and MRR@8 by 0.0451 while adding roughly 1.0s to p50 latency. It is
therefore retained as an explicit experimental profile, not the served default.
This is an observed result on the supplied QA track, not a claim of
universal improvement or legal-answer correctness.

## Historical baseline conditions

| Item | Value |
|---|---|
| Corpus | 12 versioned snapshot documents; 7,381 provisions; 440 AMENDS edges |
| Snapshot manifest SHA-256 | `1ec811da809d03036d4923ebaa28bc720fde8983e6dd5b566a000584c8d77672` |
| Retrieval | vector + BM25 + weighted RRF; `rrf_k=10`, vector/BM25 weight `3:1` |
| Candidate/result depth | 30 fetched per retrieval leg, then fused into a 30-item candidate pool; AMENDS reranks that pool, then reports top 5 |
| Legal date | `2026-08-20` |
| Embedder | `bkai-foundation-models/vietnamese-bi-encoder` at `84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0` |
| Embedding fingerprint | `…\|pyvi-ViTokenizer\|v2` |
| Decomposition LLM | disabled; historical experiment used `gpt-4o-mini`, `temperature=0` |
| Reranker | disabled (see below) |
| Rows dropped by a retrieval error | 0 on both current baseline runs |
| Source state | `00289f2426dbb43bf0830ff15a961b33d4204ec9`, `git_dirty: true` |

Commands:

```bash
vtlaw eval run data/evaluation/qa/QA_NLP.csv \
  --top-k 5 --as-of 2026-08-20 \
  --output data/evaluation/results/2026-08-20-qa-nlp-baseline-current.json

vtlaw eval run data/evaluation/qa/QA_Part2345.csv \
  --top-k 5 --as-of 2026-08-20 \
  --output data/evaluation/results/2026-08-20-qa-part2345-baseline-current.json
```

Add `--decompose` for the decomposition profile.

Each output records `provenance` (snapshot-manifest SHA-256, Git commit, and
dirty state), `selection` (limit, method, loaded rows, and selected rows), and
`label_currency` (how many reference UIDs are superseded at `as_of`). Its
configuration records the applied AMENDS heuristic and effective rerank depth
alongside the existing retrieval settings.

## Historical deterministic baseline

The values below use corrected Precision@k and the same AMENDS heuristic as the
served baseline. Latency is local-machine evidence, not a production SLO.

| Legacy QA reporting track | Rows | Recall@1 | Recall@3 | Recall@5 | P@1 | P@3 | MRR | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `QA_NLP.csv` | 94 | 0.1862 | 0.2500 | 0.2606 | 0.1915 | 0.1064 | 0.2261 | 0.350s / 0.435s |
| `QA_Part2345.csv` | 200 | 0.4163 | 0.5438 | 0.6012 | 0.4950 | 0.3233 | 0.5885 | 0.354s / 0.427s |

`label_currency` changes the interpretation of the tracks, not their raw
metric formula:

| Track | Superseded reference UIDs | Rows whose every reference is superseded | Returned top-1 superseded / all returned hits superseded |
|---|---:|---:|---:|
| `QA_NLP.csv` | 84 / 113 | 65 / 94 | 0 / 0 |
| `QA_Part2345.csv` | 4 / 281 | 4 / 200 | 0 / 0 |

The QA_NLP raw-label score is therefore not a current-law quality comparison at
this `as_of`: most of its labels deliberately name provisions the graph now
demotes. The post-run read-only audit above verifies that neither track returns
a superseded top-1 or a top-5 made entirely of superseded provisions. This is
annotation-based currency, not consolidated-law reconstruction.

The 2026-08-20 artifacts are auditable but not commit-reproducible until the dirty
working tree is committed and the same commands are rerun. Older baseline JSON
files remain historical because their P@3 values used the prior evaluator.

## Historical technique experiments

The upstream project contains more techniques than should be enabled by
default. Each was evaluated against the same snapshot and legal date instead of
being added as a portfolio checklist item. These experiments predate the
candidate-pool temporal fix, so their raw-label scores are historical and must
not be compared directly to the current baseline above.

### LLM query decomposition

`gpt-4o-mini` produces extra standalone phrasings; the original query is kept as
its own leg and all result lists are fused through the existing weighted RRF.
These full-track E2E artifacts are historical, not a current five-question
smoke-test substitute or a replacement benchmark.

| Track | Rows | Recall@1 | Recall@3 | Recall@5 | MRR | p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| `QA_NLP.csv` hybrid + decomposition | 94 | 0.4849 | 0.7757 | 0.8599 | 0.6757 | 1.652s / 2.663s |
| `QA_Part2345.csv` hybrid + decomposition | 200 | 0.3782 | 0.5630 | 0.6192 | 0.5637 | 1.542s / 2.725s |

Recall@5 improves on both tracks (+0.0886 and +0.0555) for roughly 1.4s of
added p50. Recall@1 moves in opposite directions on the two tracks — the extra
phrasings surface more of the labelled provisions overall while sometimes
displacing the single best hit.

**The two historical runs did not match to the digit.** Even at
`temperature=0` the provider is not deterministic. Two runs of the identical
command on identical data:

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

The investigation and its artifacts predate the evaluator/served-heuristic
alignment, so its rates are historical. Current retrieval demotes targets of
effective `bãi bỏ`, `thay thế`, `sửa đổi`, and `sửa đổi, bổ sung` annotations
across the candidate pool before top-k is cut; `bổ sung` alone does not trigger
demotion. This remains annotation-based score demotion, not legal
consolidation, and the generation rates below have not been rerun with it.

The screenshots in the README exposed a failure the retrieval metrics rate as a
success. Both `100/2019/NĐ-CP` and `168/2024/NĐ-CP` are in effect on 2026-08-20
with no expiry date, so the temporal filter keeps both, and retrieval returns
provisions from each. Recall@5 counts that as a hit — the labelled provision was
found. The model then cited the 2020 decree.

The historical evaluation did not catch it. The date filter cannot separate two
documents that are both in effect, and annotation-based demotion can only use
the recorded edges. `168/2024` has 199 `AMENDS` edges into `100/2019` but none
onto the clause that was cited. Citation verification says `verified`,
correctly: the citation *is* in the evidence. It checks provenance, not
currency or legal correctness.

So it needed its own measurement. `scripts/measure_citation_currency.py` scores
the questions where the defect is possible at all — evidence spanning two or
more penalty decrees — and asks whether the answer cited only a superseded one:

| Change | Contested questions | Cited only a superseded decree | Rate |
|---|---:|---:|---:|
| Starting point | 31 | 16 | 0.516 |
| Newest decree named explicitly before the evidence | 31 | 8 | 0.258 |
| Ancestor hierarchy actually reaching the prompt | 31 | **2** | **0.065** |

Two separate causes, found in that order.

**The evidence list is ordered by retrieval score.** The effective date was
already in every provision header, but the model anchored on hit #1 and cited
whatever decree ranked first. Stating the ordering as a fact read from the graph,
rather than leaving it to be inferred, took 0.516 to 0.258.

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
`{labels: labels(n), props: properties(n)}` server-side took 0.258 to 0.065.

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
Measured sequence on 2026-08-20: **16/31 → 8/31 → 2/31**. Artifacts:
`2026-08-20-citation-currency-before.json`,
`2026-08-20-citation-currency-after.json`, and
`2026-08-20-citation-currency-hierarchy-fixed.json`.

## Interpretation and limits

- The tracks are reported separately. `QA_Part2`–`QA_Part5` concatenate exactly
  into `QA_Part2345` — same 200 questions, same order, 50 rows each — so they
  are not averaged or presented as separate tests. `QA_NLP` shares no question
  with them.
- Cutoffs beyond the requested `--top-k` are not reported. Earlier artifacts
  carried `recall@7` and `recall@10` for a `--top-k 5` run; those equalled
  `recall@5` and read as a plateau when they were only list truncation.
- Rows dropped by a retrieval error are recorded in `skipped_rows` and excluded
  from every average, so a score always states its own denominator. The
  historical runs above dropped nothing.
- Comparison to the upstream project (README) uses the same 200 rows and the
  same embedding model, but the two relevance predicates differ: upstream tests
  `retrieved.startswith(reference)`, this harness additionally requires the
  match to land on a `::` UID boundary. Re-scored on these artifacts the two
  agree to four decimal places, so the comparison is not an artifact of the
  predicate — but they are not the same test. Upstream's numbers are quoted from
  its committed `eval_results/`; they were not re-measured here.
- Labels name historical provision UIDs but do not provide a date-of-fact. The
  current artifacts expose this with `label_currency` (84 of 113 QA_NLP UIDs,
  and 65 of 94 full rows, are superseded at the chosen date). A higher
  label-retrieval score can disagree with the most current rule after an
  amendment; `as_of` filtering and AMENDS annotation demotion handle only the
  evidence the corpus actually contains. They do not establish legal
  applicability or construct consolidated text.
- These are retrieval metrics, not a measure of legal correctness or
  grounded-answer faithfulness. The API performs citation-provenance checking,
  but that only verifies that an answer's citation appears in retrieved
  evidence; it does not validate the substantive legal conclusion.
- Result JSON files are ignored by Git except `data/evaluation/results/2026-08-21-release-evidence-k8.json`. Re-run the commands after any corpus, model, index, evaluator, or served-ranking change and update this report only from newly generated artifacts.
