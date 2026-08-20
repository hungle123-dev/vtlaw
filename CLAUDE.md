# CLAUDE.md — vtlaw

Vietnamese traffic-law Graph RAG over a fixed NLP-LegalQA corpus. Portfolio
system: correctness and honest measurement matter more than feature count.

## Environment

The package is **not installed in `.venv`**. Every command needs both:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest -m "not integration" -q
PYTHONPATH=src .venv/Scripts/python.exe -m vtlaw.cli graph status
```

Omitting `PYTHONPATH=src` produces ~16 fake collection errors that look like
real breakage. `uv` is on PATH; `.venv` has no `pip`, so use
`uv run --no-project --with <pkg>` for one-off dev tools rather than installing
into the venv.

Neo4j and Redis run via `docker compose up -d` on non-default ports (Bolt
17687, Redis 16379) so they cannot collide with another local instance.

## Rules that came from real bugs

**Never quote a number that is not in a generated artifact.** Every score in
`README.md` and `docs/` must be traceable to a `data/evaluation/results/*.json`
file. A stale JSON once had us quoting recall measured against a graph with
zero embeddings. Result files are gitignored on purpose — re-run
`vtlaw eval run` and update the report from its output.

**The default retrieval path stays deterministic and LLM-free.** A reproducible
number needs a pipeline with no sampling in it. Measured: the baseline repeats
to four decimal places across runs days apart; the decomposition profile varies
±0.014 Recall@5 at `temperature=0`. Report LLM-assisted profiles as a range,
never as a point, and never make one the default.

**Reject a technique that does not measure better, even the obvious one.** The
cross-encoder reranker regressed QA_NLP (0.7713 → 0.6605) and cost 7–9 s p50.
It stays an opt-in experiment, labelled experimental in the UI. Adding it would
have been a checklist item, not an improvement.

**`QA_Part2`–`QA_Part5` concatenate exactly into `QA_Part2345`** — same 200
questions, same order, 50 rows each. Never average them as independent
datasets. `QA_NLP` shares no question with them, so the two tracks are always
reported separately.

**Only report cutoffs the run measured.** A `--top-k 5` run has no recall@10;
reporting it shows a plateau that is really truncation. `aggregate_metrics`
averages each cutoff over the rows that measured it, not over every row.

**Every setting that changes results belongs in three places**: the eval
artifact `configuration` block, the Redis cache key, and `.env.example`.
`overfetch_factor` was missing from all three while silently being read by
retrieval.

**No LLM-generated Cypher.** The router may select the `cypher_query` intent,
but user text only fills parameters in fixed read-only templates
(`graph/query_templates.py`). Every template applies the same
effective/expiry date predicate as retrieval — a count for a date the document
does not apply on is a wrong answer, not a harmless one.

**Retrieval and generation must be asked the same question.** When a follow-up
is rewritten, the rewrite goes to *both*. Passing the raw turn to generation
makes the LLM answer a question the evidence was never gathered for.

**Quote conversation history as data, not as replayed chat turns.** Replayed,
the model sees a dialogue ending on a user question and completes it as the
assistant: a follow-up came back as an invented penalty figure, which then
became the retrieval query. The system prompt loses to the message structure.

**A citation regex needs both boundaries.** Without a trailing `(?!\w)`,
`168/2024/NĐ-CPx` verifies against `168/2024/NĐ-CP` — a hallucinated document
identity passes the provenance check. Document-identity patterns live once, in
`parse/patterns.py`.

**The static UI ships no runtime dependency.** Single file, inline CSS and
vanilla JS, no CDN, no framework. It must render offline. All dynamic content
goes through `textContent` / `createTextNode` — never `innerHTML`, since answer
text is LLM output and snippets come from Neo4j.

## Testing

Non-trivial logic leaves one runnable check behind. Prefer a test that fails
for the specific reason the bug existed, with the measured evidence in the
docstring, over a generic assertion.

Watch for tests that pass for the wrong reason. Three real cases here: a fixture
that `return`ed inside a `with patch(...)` block, so the patches were gone
before the test body ran; `assert "http://" not in page`, which allows
`//cdn.example/x.js` and uppercase schemes; and a hierarchy fixture that
flattened Neo4j labels alongside node properties — a shape the driver never
sends, which hid the whole ancestor path being dropped from every prompt.

**A Neo4j Node read through `Result.data()` has no labels.** `nodes(path)` comes
back as bare property dicts, so `label == "Clause"` silently never matches.
Project what you need server-side: `{labels: labels(n), props: properties(n)}`.
This dropped the parent Clause — where the penalty amount lives — from every
generated answer while all tests passed.

Integration tests **wipe Neo4j**. They require `VTLAW_TEST_WIPE=1` and should
only run against CI's throwaway service container or an expendable local
database. After running them locally, re-run `vtlaw graph import --wipe` and
`vtlaw embed run`.

## Scope boundary

Out of scope by decision, not omission: live crawling, PDF/OCR ingestion,
arbitrary LLM Cypher, and legal text consolidation. The corpus contains
amendment *instructions*, not verified consolidated text, so the system cites
retrieved sources and never synthesizes a replacement rule. Citation
verification checks provenance, not legal correctness — say so wherever the
claim could be misread.
