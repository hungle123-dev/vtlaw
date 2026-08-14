# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-15  
**Status:** Draft v8 (post-Codex round-7 revision — every contract made executable: static temporal filter compiler, regime resolution, independent pending-amendment lookup, metadata-baseline effectivity, coverage manifest, concrete corpus scope, collision-safe identity, BM25 query plan, canonical cache keys, reproducibility freeze/replay, repo-truth delivery)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (target: vendored as git submodule at pinned SHA, read-only reference — see §19 repo-truth acceptance; NOT yet wired). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

**This draft is a design target. It is NOT "production-ready" until every acceptance test in §18 passes and every delivery acceptance criterion in §19 is met.**

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Production-shape code and infrastructure.
- Right-sized for a bounded corpus. No fake scale claims.
- Legally correct validity model with **executable** temporal semantics: static compiled filters (no dynamic expressions), regime resolution before retrieval, independent pending-amendment detection, metadata-baseline effectivity with ledger precedence, coverage manifest.
- **True snapshot isolation**: every serving request pins exactly one immutable release descriptor; no half-visible state during ingest. Active + previous releases coexist safely.
- **Legal history preserved**: tombstone closes validity intervals only; never physical-deletes history.
- **Publish integrity**: `publish_fence` (strictly monotonic) + fingerprint CAS; no stale publishes; rollback allocates new fence.
- **Executable correctness**: every contract below has schema, pipeline, failure behavior, and acceptance test.

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters.
- Fine-tuning pipeline. Eval harness can evaluate fine-tuned models but training is not shipped.
- Public deployment or multi-tenant auth in v1. Single-user demo.
- OCR. Source API returns structured HTML/JSON.
- LLM-based legal validation of amendments. Pydantic validates structure, not legal meaning.
- Free-form Cypher from LLM in v1. Text2Cypher uses server-owned parameterized templates only.
- Free-text condition evaluation. Fact-dependent rules use finite reviewed enum only.
- Per-document/per-regime custom temporal evaluator beyond the supported RuleKind enum → return `temporal_rule_unsupported` / clarification; never silently fall back to query_time.

---

## 2. Decisions Log

Decisions D1–D54 are carried from prior drafts (see changelog history). New/updated decisions in v8:

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D13 | TLS verification | Default ON. `INSECURE_TLS=true` causes production startup FAIL; only allowed in dev profile. Legacy `verify=False` replaced. | Security by default |
| D55 | Canonical factual inputs | occurred_at canonical; event_date deprecated alias (both-different → 422); ongoing requires detected_date (NO fallback); ended_at for completed-lasting | Unambiguous factual contract |
| D56 | EffectivityLedgerEntry structured | entry_id, scope_locator, rule_kind enum, effective_from/to, precedence, regime_id, evidence_locator, validity_basis, review_status | Executable materialization |
| D78 | Static temporal filter compiler | Eligibility predicate compiled to 3 static nested-filter branches (query_time/conduct_time/detection_time), each with concrete date substituted. No dynamic `governing_dates[kind]` expression. Qdrant and Neo4j use identical compiled logic. | Qdrant cannot execute dynamic expressions |
| D79 | Regime resolution before retrieval | Resolve selected_regime_ids from facts before retrieval. conduct_time/detection_time/query_time map to valid regime(s). selected_regime_ids in request context, cache key, predicate, citation. regime_id checked inside same nested interval. | NĐ100/NĐ168 disambiguation |
| D80 | Independent pending-amendment lookup | PendingAmendmentRecord index checked by legal locator/provision_uid BEFORE or PARALLEL to retrieval, not gated on top-k. Hit → legal_status_review_pending outcome + warning even if target not in top-k. | review_pending never missed |
| D81 | Metadata-baseline effectivity | effectDate (metadata) = document baseline; expireDate = outer validity cap; ledger exception/transition override with higher precedence. Metadata/text conflict → quarantine with evidence. | Integrates repo metadata fields |
| D82 | Reviewed-overrides artifact | Machine-readable reviewed-overrides artifact materializes intervals; promotion workflow: review log decision → reviewed-overrides artifact → materialize at release build. Review log alone insufficient. | Review decisions become executable |
| D83 | Coverage manifest | verified_from, verified_to, gaps[], per-regime/document coverage, latest_source_watermark. historical_coverage_unsupported based on coverage manifest, not single date. | Honest coverage |
| D84 | Concrete corpus scope | Allow-list, source IDs/queries, dependency closure, exclusion evidence committed in §6 (not Open Items). Serving requires populated allow-list. Reverse-amendment discovery mechanism + stated limitation. | Core serving behavior, not TBD |
| D85 | Delta scan | content_hash + metadata/updDateTime capability probe + overlap/watermark + full reconciliation. NOT issueDate/effectDate as cursor. | Correct change detection |
| D86 | Parse fail carry-forward | Parse fail on one document carries forward last-good document version in release; full projection preserved; document never lost. | No data loss |
| D87 | legal_document_key collision map | Stable audited collision map. docIdentity collision → distinct legal_document_keys suffixed by source_document_id. Resolver rejects ambiguity; ingestion never loses documents. | Collision-safe identity |
| D88 | provision_version_id full hash | provision_version_id = {provision_uid}::rev::{full SHA256 provision_text_hash}. No 12-char truncation. Collision registry test. | No truncation collision |
| D89 | Metadata/ExternalDocument immutability | Metadata + ExternalDocument nodes immutable per metadata_revision (content-addressed). Release manifest references specific metadata revisions. N-1 cannot mutate metadata when N updates. Reconcile checks referenced metadata revision. | Snapshot-safe global refs |
| D90 | BM25 query plan | Exact Neo4j fulltext query plan: fulltext yield → WHERE release_id + validity predicate → ORDER BY score → LIMIT. release_id + validity filtered BEFORE LIMIT. max_fulltext_scan bound. | N-1 cannot crowd out N |
| D91 | Graph expansion bounds | Depth limit, row limit, timeout, token budget enforced BEFORE materializing context. Final cap ≤8 guaranteed after expansion. | Bounded expansion |
| D92 | TemplateCall complete v1 enum | All v1 templates enumerated explicitly (§12). No "... extensible" in running contract. | Executable contract |
| D93 | Canonical cache keys | All cache keys = SHA256(canonical JSON object). Sorted keys, no whitespace. No delimiter-less string concatenation. | Deterministic keys |
| D94 | Cypher branch cache | Cypher branch uses answer cache (lookup before execute, write after done success) consistent with retrieve branch. | No dangling cache write |
| D95 | Streaming semantics | Emit token immediately; tee to bounded accumulator; write answer cache ONLY after terminal done success. No cache on pre-token failure/mid-stream error/disconnect. Buffered/non-cancellable provider holds slot to timeout; no false "released" claim. | Correct streaming + caching |
| D96 | Feedback ownership | Single-user deployment. Feedback validates trace_id exists in this deployment's trace store + idempotency_key. No per-user session ownership (single-user). | Honest ownership model |
| D97 | Concrete limits | MAX_REQUEST_BODY_BYTES=262144, MAX_HISTORY_MESSAGES=40, MAX_HISTORY_TOKENS=8000, MAX_QUERY_CHARS=2000, MAX_FILTER_VALUES_PER_FIELD=10, MAX_DECOMPOSED_QUERIES=6, MAX_GRAPH_EXPANSION_DEPTH=2, MAX_GRAPH_EXPANSION_ROWS=50, MAX_EXECUTOR_QUEUE=100, MAX_FULLTEXT_SCAN=500, serving context_k=8, token_budget=4096 | Bounded everywhere |
| D98 | Reproducibility freeze/replay | Retrieval eval freezes/replays router/rewrite/decomposition outputs; pins provider/model/params/prompt version. Commits baseline artifact, split membership, run fingerprint, promotion rule. | Reproducible eval |
| D99 | Successor-aware relevance | Eval relevance uses successor/AMENDS-aware matching (resolved_provision_version_ids + successor sets), not prefix-match UID. | Correct relevance |
| D100 | eval_k sync | eval_k=8; primary metric Recall@8. Legacy top_k=30/no-Recall@8 scripts deprecated, not called validation. | Consistent validation |
| D101 | Import-time test | No-network/no-LLM import-time test; no infinite retry loop (bounded retries with deadline). | Safe tests |
| D102 | Repo-truth delivery acceptance | Mandatory: commit .gitmodules + gitlink at pinned SHA, recursive checkout, SHA assertion in CI. Current state (untracked NLP-LegalQA/, no .gitmodules) documented as NOT yet met. | Honest delivery |
| D103 | Secret scan | CI secret scan (gitleaks/trufflehog). Never copy credentials from reference repo. Reference repo contains real-looking creds (nguyenhoangquan.com/Neoneo4j in CLAUDE.md) — flagged, not propagated. | Credential safety |
| D104 | PendingAmendmentRecord schema | target_locator, target_provision_uid (nullable), planned_effective_date (nullable), governing_date_kind, effective_rule_id (nullable), unresolved_rule_id (nullable), evidence_locator, review_status (auto/quarantined) | Executable pending detection |
| D105 | AMENDS state relationship | auto = resolved, annotation-only; quarantined = unresolved/invalid; review_pending = serve-time state derived from auto/quarantined records relevant to query; reviewed = materializes eligibility. Raw pending candidate does NOT require reviewed effective_rule_id; only reviewed edges materialize eligibility. | Clear state machine |

(D1–D12, D14–D54 carried from prior drafts; full list retained in spec body where referenced.)

---

## 3. Architecture Overview

### Service Map (Two Application Containers)
```
┌─────────────────────────── docker-compose ───────────────────────────┐
│  FRONTEND                                                            │
│  ├─ nextjs          : React chat UI, SSE via fetch() ReadableStream  │
│                                                                      │
│  APP CONTAINERS                                                      │
│  ├─ api             : FastAPI (single worker process)                │
│  │                   validate+normalize→trace→facts→regime-resolve   │
│  │                   →pin release→route→rewrite→cache(1)            │
│  │                   →decompose→cache(2)                             │
│  │                   →[pending lookup ∥ retrieval]                   │
│  │                   →compile eligibility filter                     │
│  │                   →rerank(GPU, in-process semaphore)              │
│  │                   →gen(prompt wrapper, per-context basis)         │
│  │                   Endpoints: POST /v1/chat, POST /v1/feedback,    │
│  │                   GET /livez, GET /readyz, GET /metrics           │
│  ├─ arq-worker      : ingestion offline                              │
│  │                   (scrape→parse→effectivity→amends→embed          │
│  │                    →build→reconcile→publish-CAS)                  │
│  │                   Uses write/schema/publish Neo4j principal       │
│                                                                      │
│  DATA STORES                                                         │
│  ├─ neo4j           : graph hierarchy + AMENDS + Lucene BM25         │
│  │                   Nodes keyed by snapshot_node_id                │
│  │                   Filtered by release_id per snapshot            │
│  │                   Metadata/ExternalDoc = immutable per revision   │
│  │                   API read-only principal                        │
│  ├─ qdrant          : Per-release collections (legal_v{N})          │
│  │                   HNSW dense + payload (intervals w/             │
│  │                   governing_date_kind + regime_id) + context     │
│  └─ redis           : arq broker (arq:*)                            │
│                       cache (cache:*)                               │
│                       descriptor authority (descriptor:active Hash)  │
│                       publish_fence counter (descriptor:fence_ctr)   │
│                       pending-amendment index (descriptor:pending)   │
│                       Config: AOF ON, noeviction                    │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : self-host compose with closure, volumes,       │
│  │                    backup, resource profile (§19)                 │
│  ├─ prometheus      : infra metrics, scrapes GET /metrics           │
│  └─ grafana         : dashboard                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### INVAR 1 (Snapshot Isolation)
Every serving request pins exactly one release_descriptor at ingress. All reads use that pinned descriptor. No read crosses releases. Active + previous releases coexist safely.

### Request Path (Hot Path)
```
POST /v1/chat {query, chat_history?, as_of_date?, occurred_at?,
               ended_at?, detected_date?, violation_state?, filters?}
  │
  ▼ VALIDATE + NORMALIZE (§13 limits):
     • Body ≤ MAX_REQUEST_BODY_BYTES; query ≤ MAX_QUERY_CHARS;
       history ≤ MAX_HISTORY_MESSAGES / MAX_HISTORY_TOKENS;
       filters ≤ MAX_FILTER_VALUES_PER_FIELD
     • event_date alias → occurred_at; both-different → 422
     • After normalization only canonical fields downstream
  ▼ create fresh trace_id (every request, including cache hits)
  ▼ PIN RELEASE DESCRIPTOR (Redis Hash authority; fallback local file)
     If unavailable + no fallback: 503
  ▼ DERIVE GOVERNING DATES:
     conduct_time = ended_at ?? occurred_at (completed)
     detection_time = detected_date (ongoing; REQUIRED; NO fallback)
     query_time = as_of_date (default today)
     governing_dates = {conduct_time?, detection_time?, query_time}
  ▼ REGIME RESOLUTION (D79):
     selected_regime_ids = resolve_regimes(governing_dates, transition rules)
     (e.g., conduct_time in NĐ100 era → traffic_sanctions_v1;
      conduct_time in NĐ168 era → traffic_sanctions_v2)
  ▼ requires_clarification check:
     • sanction/transitional + missing facts → clarify (one token + done, pre-retrieval)
     • governing_dates outside coverage manifest (§5/D83) → historical_coverage_unsupported
  ▼ route (LLM, temp=0, max_tokens=64)
     intent ∈ {greeting, cypher_query, retrieve, reject, clarify}
  │
  ├─ reject → refusal, done
  ├─ greeting → deterministic, done
  ├─ clarify → one deterministic token + done (outcome=clarified)
  ├─ cypher_query → rewrite → ANSWER CACHE LOOKUP (D94)
  │   → TemplateCall (complete v1 enum, extra="forbid", resolve via legal_document_key)
  │   → execute (pinned release + validity predicate + LIMIT + timeout)
  │   → gen → tee tokens + bounded accumulator
  │   → on done SUCCESS: write answer cache + trace → stream
  │
  └─ retrieve:
       ▼ rewrite (multi-turn; skip if no history)
       ▼ ANSWER CACHE LOOKUP (after route+rewrite) (§14):
          key = SHA256(canonical JSON {rewritten_query, intent, filters,
              governing_dates, selected_regime_ids, violation_state,
              release_fingerprint, embedding_version, reranker_version,
              model_name, prompt_wrapper_version,
              retrieval_policy_fingerprint, runtime_policy_fingerprint})
          HIT → new trace (cache_origin_trace_id), skip below
       ▼ decompose (max_subqueries=MAX_DECOMPOSED_QUERIES + dedup)
       ▼ RETRIEVAL CACHE LOOKUP (after decompose, before search):
          key = SHA256(canonical JSON {decomposed_queries, filters,
              governing_dates, selected_regime_ids, violation_state,
              release_fingerprint, embedding_version, reranker_version,
              retrieval_policy_fingerprint})
          HIT → skip search+rerank
       ▼ PARALLEL:
          (a) PENDING-AMENDMENT LOOKUP (D80/D104): independent by
              legal locator/provision_uid, NOT gated on top-k
          (b) multi_search (compiled eligibility filter, D78):
              Qdrant: compiled 3-branch nested filter (regime_id inside)
              Neo4j BM25: same compiled predicate, release_id BEFORE LIMIT (D90)
       ▼ fuse RRF per sub-query → aggregate
       ▼ fetch context (Qdrant payload primary; Neo4j fallback)
       ▼ expand context (D91 bounds: depth ≤ MAX_GRAPH_EXPANSION_DEPTH,
          rows ≤ MAX_GRAPH_EXPANSION_ROWS, timeout, token budget;
          final-select ≤8 contexts AND token_budget)
       ▼ per-request per-context applicability + legal_time_basis
       ▼ rerank (GPU, FP16) → heuristic rerank (post-eligibility, eval-gated)
       ▼ final-select top serving contexts (≤8 AND token_budget)
       ▼ build context_str (per-context applicability + review_pending flags)
       ▼ WRITE RETRIEVAL CACHE (after retrieval+rerank success, even if gen fails)
       ▼ gen (versioned prompt wrapper: governing_dates, per-context basis,
          selected intervals, citations, eligible contexts, review_pending flags;
          answer ONLY from supplied eligible contexts; caveated if review_pending)
          Retry ONLY before first token. No retry mid-stream.
       ▼ tee tokens to client + bounded accumulator
       ▼ on terminal done SUCCESS: write ANSWER CACHE
          NOT on pre-token failure / mid-stream error / disconnect
       ▼ SSE: meta → sources → token* → done|error
          outcome includes legal_status_review_pending if pending lookup hit
       ▼ trace Langfuse + metrics Prometheus
```

### Ingestion Path (Offline)
```
cron/manual → arq enqueue
  ▼ ALLOCATE publish_fence = INCR descriptor:fence_ctr
     Record base_fingerprint + base_fence
  ▼ scrape (allow-list §6, dependency closure, delta D85, TLS verify ON,
    throttle, future-effective D45). Raw immutable provenance (§7).
  ▼ parse (src/legal_rag/ingestion owner, D53; alphanumeric clause;
    appendix default-preserve; NFC; idempotent by source revision identity)
     Parse fail on one doc → CARRY-FORWARD last-good version (D86)
  ▼ parse-quality gate (raw length, anchors, hierarchy coverage;
    partial → keep last-good, no tombstone)
  ▼ effectivity_extract (D81/D82):
     effectDate = document baseline; expireDate = outer cap;
     ledger exception/transition override with higher precedence.
     Metadata/text conflict → quarantine with evidence.
     Reviewed entries → reviewed-overrides artifact → materialize intervals
     (governing_date_kind + regime_id per interval; static general truncated
      at exception dates D70; regime-aware transitions D71).
     Unsupported evaluator → temporal_rule_unsupported annotation.
  ▼ amends_extract (D104/D105):
     Discriminated union; EvidenceLocator + effective_rule_id (validated for
     reviewed edges; raw pending candidates do NOT require reviewed rule_id).
     Edge ID includes evidence identity. Failed → quarantine.
     auto/quarantined with planned_effective_date → PendingAmendmentRecord
     written to pending-amendment index.
  ▼ embed (reuse ONLY when embedding_input_fingerprint matches D64)
  ▼ BUILD release N (staging):
     Qdrant legal_v{N} full projection (carry-forward unchanged; embed changed;
     all provision versions within temporal_support_window; payload intervals
     carry governing_date_kind + regime_id).
     Neo4j release_id=N (snapshot_node_id; multi-version; same-release rels;
     metadata/ExternalDocument immutable per metadata_revision D89).
     Materialize validity from REVIEWED AMENDS (D105).
     Static general truncated at exceptions (D70).
     Fulltext indexes scoped to release_id.
     Pending-amendment index built.
     Coverage manifest built (§5/D83).
  ▼ RECONCILE (D63/D89): node tuple set, edge digest+endpoints
     (incl. RELATED_TO external), derived_state_hash per point,
     fulltext readiness, Qdrant point/config fingerprint,
     metadata revision referenced by release.
     Mismatch → DO NOT publish.
  ▼ PUBLISH (CAS with fencing D61):
     Lua CAS on Redis Hash: check fingerprint AND base_fence AND
     new_fence > current_fence. Bootstrap via exclusive init lock.
     On success → write active_release.json, notify API.
     Stale → discard/rebase.
  ▼ GC (after grace > max request deadline; never delete temporal support window)
  ▼ write ingestion_manifest + reviewed-overrides + review_log, emit metrics
```

### INVAR 2 (Full Projection)
Each release is a FULL serving projection. Parse failure carries forward last-good (D86). Delta only optimizes compute/vector reuse.

### INVAR 3 (Single Pointer with CAS Authority)
Redis Hash `descriptor:active` = CANONICAL AUTHORITY. Local `active_release.json` = DURABLE CACHE written AFTER successful CAS. Rollback allocates new publish_fence, audited.

---

## 4. Identity Model

| Identifier | Format | Stability | Purpose |
|---|---|---|---|
| `source_document_id` | docGUId (UUID) | Immutable | Downloaded document file key |
| `raw_blob_hash` | SHA256(raw HTML) | Per blob | Blob dedup |
| `source_metadata_revision_hash` | SHA256(metadata JSON) | Per metadata revision | Metadata-only updates |
| `normalized_text_hash` | SHA256(normalized text) | Per normalized content | Parse idempotency; evidence binding |
| `provision_text_hash` | SHA256(single provision text) | Per provision content | Provision version granularity |
| `legal_document_key` | See §4.1 collision map | Stable per legal act | Internal non-ambiguous key |
| `provision_uid` | `{legal_document_key}::article::{n}::clause::{n}::point::{letter}` | Stable logical lineage | Citations, AMENDS, QA labels |
| `provision_version_id` | `{provision_uid}::rev::{provision_text_hash}` (FULL SHA256, D88) | Per content version | Canonical content ID; Qdrant point ID input |
| `snapshot_node_id` | UUIDv5(RELEASE_NAMESPACE, `{release_id}:{provision_version_id}`) | Per (release, version) | Neo4j physical node key |

### 4.1 legal_document_key Collision Map (D87)
docIdentity can collide. Maintain a stable, audited collision map:
- Primary: `legal_document_key = normalized docIdentity` when unambiguous.
- Collision: if multiple source_document_ids share a docIdentity and are distinct legal documents, assign distinct keys: `{docIdentity}#{source_document_id[:8]}`. Recorded in `collision_map` artifact (audited, immutable entries, versioned).
- Resolver: when a legal_locator maps to multiple legal_document_keys → reject ambiguity (quarantine the citation), do NOT auto-pick.
- Ingestion: NEVER loses documents on collision. Every source_document_id gets a legal_document_key (possibly suffixed). Collision does not block ingest; it routes to the collision map for review.

### 4.2 Source Revision Identity (D66)
```
source_revision_identity = (source_document_id, raw_blob_hash,
    source_metadata_revision_hash, normalized_text_hash,
    normalizer_version, parser_version)
source_revision_identity_hash = SHA256(canonical JSON of above)
```
Manifest path: `manifests/{source_revision_identity_hash}/manifest.json`. Each transform revision = separate immutable manifest. Parse idempotency key = full identity.

### 4.3 provision_version_id Full Hash (D88)
provision_version_id = `{provision_uid}::rev::{provision_text_hash}` using FULL 64-char SHA256. NO truncation. Collision registry test (§18) asserts distinct fixture texts produce distinct provision_version_ids.

### Mapping Round-Trip
```
Qdrant legal_v{N}: point_id = UUIDv5(POINT_NAMESPACE, provision_version_id)
Neo4j release_id=N: node.snapshot_node_id = UUIDv5(RELEASE_NAMESPACE, "{N}:{provision_version_id}")
Cross-store: Qdrant payload.provision_version_id ←→ Neo4j node.provision_version_id
Resolution: legal_locator → legal_document_key (ambiguity-rejecting) → provision_uid → provision_version_id
```

### INVAR 5 (History Preserved)
Tombstone = close validity interval. NEVER physical-delete text/evidence due to reparse or retention. Physical deletion only after temporal support window + grace.

---

## 5. Temporal Model

### Factual Inputs (D55)
```typescript
{
  as_of_date?: string, occurred_at?: string, ended_at?: string,
  detected_date?: string, violation_state?: "completed"|"ongoing"|"unknown"
}
```
event_date = deprecated alias → normalized to occurred_at; both-different → 422. Ongoing requires detected_date (NO fallback). After normalization only canonical fields.

Governing dates:
- completed → conduct_time = ended_at ?? occurred_at
- ongoing → detection_time = detected_date (REQUIRED; missing → clarify)
- query_time = as_of_date (default today)

### 5.1 Static Temporal Filter Compiler (D78)
Qdrant/Neo4j cannot execute `governing_dates[interval.governing_date_kind]` dynamically. The **eligibility compiler** produces a disjunction of static nested-filter branches, one per governing date present in the request:

```python
def compile_eligibility(governing_dates, selected_regime_ids) -> Filter:
    branches = []
    for kind, date_val in [
        ("query_time", governing_dates.query_time),
        ("conduct_time", governing_dates.conduct_time),
        ("detection_time", governing_dates.detection_time),
    ]:
        if date_val is None:
            continue
        branches.append(AND(
            Match(interval.governing_date_kind == kind),
            Range(interval.from <= date_val),
            OR(IsNull(interval.to), Range(interval.to > date_val)),
            Match(interval.state == "in_force"),
            Match(interval.regime_id IN selected_regime_ids),
        ))
    return OR(*branches)  # nested: all conditions on SAME interval element
```

- Qdrant: compiled to `Filter(should=[nested Match/Range per branch])` using nested payload conditions so all fields are evaluated on the same interval element.
- Neo4j: compiled to identical Cypher WHERE using a list-comprehension over `node.validity_intervals` with the same branch structure.
- regime_id is checked INSIDE each nested branch (D79), not just stored in payload.
- Branches with absent governing date are omitted. If no branches (all dates absent) → not eligible.

### 5.2 Regime Resolution (D79)
Before retrieval, resolve `selected_regime_ids` from governing_dates + reviewed transition rules:
```
resolve_regimes(governing_dates, transition_rules) -> set[regime_id]
  reference = governing_dates.conduct_time ?? detection_time ?? query_time
  return {r.regime_id for r in reviewed_transition_rules
          if r.applies_at(reference)}
```
Example: conduct_time=2023-06-01 → NĐ100 era → `selected_regime_ids={traffic_sanctions_v1}`. conduct_time=2025-06-01 → NĐ168 era → `{traffic_sanctions_v2}`. selected_regime_ids added to request context, cache key, compiled predicate, citation.

### 5.3 Effectivity with Metadata Baseline (D81)
Precedence (highest → lowest):
1. Reviewed AMENDS-materialized intervals.
2. Reviewed ledger exception/transition rules.
3. Reviewed ledger static general.
4. Metadata effectDate (document baseline).

`expireDate` (metadata) = OUTER VALIDITY CAP on all intervals (interval.to ≤ expireDate).

Metadata/text conflict (e.g., effectDate ≠ text "Hiệu lực thi hành") → quarantine with evidence for review; do not silently pick.

Materialization at release build uses reviewed-overrides artifact (D82). Payload/manifest carry: effectDate, expireDate, ledger entries, materialized intervals.

### 5.4 Reviewed-Overrides Artifact + Promotion (D82)
- `review_log`: records reviewer decisions (edge/entry id, reviewer, timestamp, decision, rationale). Annotation only.
- `reviewed-overrides` artifact: machine-readable mapping from reviewed decisions to concrete materialized intervals/effectivity entries. Generated from review_log via promotion workflow.
- Promotion workflow: review_log decision → promotion step validates → writes reviewed-overrides artifact → release build materializes intervals from reviewed-overrides. Review log alone does NOT materialize.

### 5.5 Coverage Manifest (D83)
```python
@dataclass(frozen=True)
class CoverageManifest:
    verified_from: date
    verified_to: date
    gaps: list[CoverageGap]        # [{from, to, regime_id, reason}]
    per_regime: dict[str, RegimeCoverage]
    per_document: dict[str, DocumentCoverage]
    latest_source_watermark: str   # updDateTime watermark
```
historical_coverage_unsupported is determined by the coverage manifest: if governing/reference date falls in a `gap` or outside `[verified_from, verified_to]` → unsupported (distinct from "not found"). Rule for dates within gaps → `historical_coverage_unsupported` with gap reason.

### 5.6 EffectivityLedgerEntry (D56)
```python
class RuleKind(str, Enum):
    STATIC_GENERAL="static_general"; STATIC_EXCEPTION="static_exception"
    COMPLETED_CONDUCT_TIME="completed_conduct_time"
    ONGOING_DETECTION_TIME="ongoing_detection_time"
    REVIEW_PENDING="review_pending"

@dataclass(frozen=True)
class EffectivityLedgerEntry:
    entry_id: str; scope_locator: str; rule_kind: RuleKind
    effective_from: Optional[date]; effective_to: Optional[date]
    precedence: int; regime_id: str
    evidence_locator: EvidenceLocator; validity_basis: str
    review_status: str
```
No free-text condition. Rules outside enum → REVIEW_PENDING. Unsupported evaluator → temporal_rule_unsupported/clarification (D72).

### 5.7 Validity Interval Payload (Executable)
```json
{
  "from": "2025-01-01", "to": null, "state": "in_force",
  "governing_date_kind": "query_time",
  "effective_rule_id": "ledger_123", "basis": "static_general",
  "regime_id": "traffic_sanctions_v2", "precedence": 10
}
```

### 5.8 Eligibility Ordering (D58)
Reviewed AMENDS > fact-dependent transition rules > static effectivity. No overlapping applicable in_force intervals per provision/regime without materialized precedence (D70).

### INVAR 40 (Conditional Materialization)
conditional/unknown eligible ONLY after reviewer materializes in_force interval (via reviewed-overrides artifact). Predicate matches state=in_force.

### Future-Effective Ingestion (D45)
Promulgated-but-not-yet-effective ingested immediately; future intervals self-handle boundary.

---

## 6. Corpus Scope and Allow-List (D84 — Concrete)

This section is committed core serving behavior, NOT an Open Item.

### Versioned Allow-List (`allowlist_v1.yaml`, committed)
```yaml
version: 1
source: phapluat.gov.vn
doc_group_ids: [<populated at bootstrap>]   # concrete IDs, committed before serving
field_ids: [<populated at bootstrap>]        # giao thông đường bộ field IDs
search_queries:
  - "giao thông đường bộ"
  - "trật tự an toàn giao thông"
  - "xử phạt vi phạm hành chính giao thông"
local_matching:
  doc_name_pattern: "giao thông|trật tự.*giao thông|xử phạt.*giao thông|đường bộ"
  exclusion_patterns: [<administrative-only patterns, each with exclusion evidence>]
discovery:
  pagination: {rowAmount: 100, iterate: pageIndex}
  delta_strategy: content_hash + updDateTime_capability_probe + overlap_watermark + full_reconciliation
dependency_closure: true
```
- **Bootstrap requirement**: allow-list doc_group_ids/field_ids MUST be populated (via facet discovery) and committed BEFORE serving starts. Serving refuses to start with empty allow-list (`/readyz` → 503 `allowlist_not_populated`).
- **Exclusion evidence**: each exclusion_pattern carries a justification + test fixture proving excluded docs are non-normative.

### Dependency Closure (D38)
Any document amending/repealing an in-scope provision MUST be fully ingested regardless of keyword/field/title match.

### Reverse-Amendment Discovery (D84)
Mechanism: after ingesting a document, extract its AMENDS targets. If a target is in-scope but the amending document is not yet ingested, enqueue the amending document (forward closure). Also traverse `docListOther` references for candidate amending docs.

**Stated limitation**: reverse discovery only finds amendments reachable from already-ingested documents (via their outgoing AMENDS or docListOther). A document that amends an in-scope provision but is itself out-of-scope and never referenced by any ingested document will NOT be discovered automatically. Coverage manifest records this as a known gap class.

### Delta Scan (D85)
Delta detection uses content_hash comparison + metadata/updDateTime capability probe + overlap/watermark + periodic full reconciliation. NOT issueDate/effectDate as cursor.

### Effect Status Not Scope Gate (D37)
Allow-list does NOT filter by document effect_status (NĐ 168 = "Hết Hiệu lực một phần" must be included).

---

## 7. Raw Provenance (D49, D66)

### Store Layout
```
raw/
  blobs/{raw_blob_hash}/original.html
  metadata/{source_metadata_revision_hash}/metadata.json
  normalized/{normalized_text_hash}/{normalizer_version}/normalized.txt
  manifests/{source_revision_identity_hash}/manifest.json
```
Each transform revision = separate immutable manifest. Blobs dedupe by hash. Evidence offsets bind to specific `(normalized_text_hash, normalizer_version)` within the manifest identified by `source_revision_identity_hash`. Manifest immutable per revision; new revision = new entry, never overwrite.

### Owner of Serving Corpus (D53)
`src/legal_rag/ingestion` owns serving corpus guarantees. Submodule NLP-LegalQA is read-only reference only.

---

## 8. Release and Snapshot Isolation

### Release Descriptor
```python
@dataclass(frozen=True)
class ReleaseDescriptor:
    corpus_version: int
    publish_fence: int            # strictly monotonic, separate from corpus_version
    qdrant_collection: str
    neo4j_release_id: int
    embedding_version: str
    schema_version: str
    manifest_hash: str
    coverage_manifest_hash: str   # D83
    published_at: datetime
    temporal_support_window_from: date
    fingerprint: str
```
fingerprint = SHA256(canonical JSON of all fields except fingerprint).

### Publish CAS (D61)
Redis Hash authority. Lua CAS:
```lua
-- KEYS[1]=descriptor:active, KEYS[2]=descriptor:init_lock
-- ARGV[1]=base_fence, ARGV[2]=base_fp, ARGV[3]=new_fence, ARGV[4..]=field/value pairs
local cur_fence = tonumber(redis.call('HGET', KEYS[1], 'publish_fence'))
local cur_fp = redis.call('HGET', KEYS[1], 'fingerprint')
if cur_fence == nil then
  if redis.call('EXISTS', KEYS[2]) == 1 then
    for i=4,#ARGV,2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1]) end
    return 1
  else
    return -1  -- bootstrap contention
  end
end
if cur_fp == ARGV[2] and cur_fence == tonumber(ARGV[1])
   and tonumber(ARGV[3]) > cur_fence then
  for i=4,#ARGV,2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1]) end
  return 1
else
  return 0  -- stale
end
```
Uses HGET/HSET (Redis Hash), not json_extract. Bootstrap via SETNX init_lock. Rollback allocates new publish_fence (INCR) + new fingerprint → stale builds cannot re-publish. Audited (who/when/from/to fence + corpus_version).

### Reconcile (D63, D89)
Publish requires match on: node tuple set `(snapshot_node_id, release_id, provision_version_id)`; edge digest + endpoints (incl. RELATED_TO external); derived_state_hash per point; fulltext-index readiness; Qdrant point/config fingerprint; **metadata revision referenced by release** (D89).

### GC (D32)
Delete old releases only after grace > max request/SSE deadline. Never delete temporal support window history. Metadata/ExternalDocument nodes GC'd only when no release references their metadata_revision (D89).

### Redis Outage, /readyz, Backup
Serve from in-memory + local active_release.json fallback. Fresh volume → 503 corpus_not_ready. /readyz checks descriptor + Qdrant collection + Neo4j snapshot + fulltext readiness + metadata revisions + init lock absent + allow-list populated. One common backup_manifest; cache/arq ephemeral.

---

## 9. Ingestion Pipeline (Pillar 1)

Stages (see §3 flow): allocate_fencing_token → scrape → parse (carry-forward last-good on fail, D86) → parse-quality gate → effectivity_extract (D81/D82) → amends_extract (D104/D105) → embed (D64 reuse) → build_release (full projection, coverage manifest) → reconcile → publish CAS → GC → report.

### Error Handling (Ingestion)
| Failure | Behavior |
|---|---|
| Scrape HTTP 200 + non-null error | Upstream error, retry (bounded). NOT end-of-catalog. |
| Scrape docs[] empty + error==null | End-of-catalog, stop pagination. |
| Parse fail one doc | Carry-forward last-good version (D86). Continue batch. |
| Parse-quality gate fails | Keep last-good. No tombstone. |
| Metadata/text effectivity conflict | Quarantine with evidence (D81). |
| AMENDS invalid schema / effective_rule_id invalid | Quarantine edge. Ingest continues. |
| AMENDS auto/quarantined w/ planned_effective_date | Write PendingAmendmentRecord (D104). |
| Embed/import fail | Bounded retry; after N → mark failed, alert. No infinite loop (D101). |
| Reconcile mismatch | DO NOT publish. Alert. |
| Stale publish (CAS 0) | Discard/rebase. |
| Bootstrap contention (CAS -1) | Wait, retry. |
| Interrupted publish | Restart from Redis authority. |
| INSECURE_TLS=true production | Startup FAIL (D13/D76). |

All retries bounded with deadline (D101). No infinite retry loops.

---

## 10. AMENDS Edge Model

### 10.1 AMENDS State Machine (D105)
| State | Meaning | Eligibility Impact |
|---|---|---|
| `auto` | Locator resolved, schema valid, not reviewed | Annotation only; source of pending candidates |
| `quarantined` | Unresolved/invalid/validation-failed | Logged; source of pending candidates |
| `review_pending` | Serve-time state derived from auto/quarantined records relevant to query | Warning + non-confident answer; does NOT materialize eligibility |
| `reviewed` | Human/corroborated verified | Materializes eligibility (via reviewed-overrides artifact) |

**Key**: raw pending candidates do NOT require reviewed effective_rule_id. Only reviewed edges materialize eligibility. review_pending is a serve-time derivation, not an ingest state.

### 10.2 PendingAmendmentRecord (D104, D80)
```python
@dataclass(frozen=True)
class PendingAmendmentRecord:
    record_id: str
    target_locator: LegalLocator
    target_provision_uid: Optional[str]      # None if unresolved
    planned_effective_date: Optional[date]    # None if unresolved
    governing_date_kind: GoverningDateKind
    effective_rule_id: Optional[str]          # None for raw pending
    unresolved_rule_id: Optional[str]
    evidence_locator: EvidenceLocator
    review_status: Literal["auto", "quarantined"]
```
Index stored in Redis `descriptor:pending` (per release) + materialized in release. Built during ingestion from auto/quarantined AMENDS edges with planned_effective_date.

### 10.3 Independent Pending Lookup (D80)
Pending lookup runs INDEPENDENTLY of retrieval, keyed by legal locator/provision_uid, BEFORE or PARALLEL to ANN search. NOT gated on whether target made top-k. If the query's legal locators / candidate provisions intersect a PendingAmendmentRecord whose planned_effective_date ≤ relevant governing_date → set outcome = `legal_status_review_pending`, add warning, and generator produces caveated (non-confident) answer — even if target is not in top-k.

### 10.4 Discriminated Union Ops (D22, D59)
AddProvisionOp / ReplaceTextOp / RepealOp / AmendOp, each requiring EvidenceLocator + effective_rule_id (validated: exists, correct source/release, reviewed, scope covers target — for reviewed edges). Pydantic model_validator enforces one-of/at-least-one. PatchSpec requires expected_successor_hash OR expected_successor_text (validator). Patch materializes only if deterministic on exact base hash.

### 10.5 AMENDS Edge Identity (D74)
edge_id = SHA256(operation + amending_snapshot_node_id + target_snapshot_node_id + effective_rule_id + evidence_locator.excerpt_hash). Two amendments cannot MERGE.

### EvidenceLocator (defined IDs only)
```python
@dataclass(frozen=True)
class EvidenceLocator:
    source_document_id: str; source_metadata_revision_hash: str
    normalized_text_hash: str; normalizer_version: str
    start_offset: int; end_offset: int; excerpt_hash: str
```

---

## 11. Embeddings + Indexing (Pillar 2)

### Embedding Job
Model vietnamese-bi-encoder (pinned revision); pyvi tokenize; batch 32–64; CPU default. Reuse vector ONLY when embedding_input_fingerprint matches (D64): SHA256(model_revision + segmenter_version + tokenizer_version + preprocessing_config + dimension + provision_text). Pipeline change → re-embed.

### Qdrant Per-Release Collection
point_id = UUIDv5(POINT_NAMESPACE, provision_version_id). Payload includes provision_uid, provision_version_id, source_document_id, legal_document_key, doc_identity, doc_type, validity_intervals[] (with governing_date_kind + regime_id per interval), effectDate, expireDate, field[], organ[], label/number/parents, corpus_version, embedding_version, context_text, embedding_input_fingerprint, citation projection (title, official_url, human_citation, provision_path, snippet). NO static applicability_basis.

### Eligibility Filter
Compiled via §5.1 compiler (3 static branches, regime_id inside). Applied to Qdrant + Neo4j BM25 BEFORE final LIMIT.

---

## 12. Retrieval + Generation (Pillar 3)

### Neo4j BM25 Query Plan (D90)
Fulltext index per label (Article/Clause/Point) on content/title. Exact query plan:
```cypher
CALL db.index.fulltext.queryNodes($index_name, $text) YIELD node, score
WHERE node.release_id = $release_id
WITH node, score
WHERE <compiled validity predicate on node.validity_intervals>
RETURN node.provision_version_id AS uid, score
ORDER BY score DESC
LIMIT $k
```
- `release_id` filtered immediately after fulltext yield, BEFORE validity and LIMIT.
- Validity predicate BEFORE final LIMIT.
- `MAX_FULLTEXT_SCAN` bound on candidates to cap scan.
- Because release_id + validity precede LIMIT, release N-1 cannot crowd out N.

### Graph Expansion Bounds (D91)
Expansion (sibling Points, Article/Clause children) enforces BEFORE materializing context: depth ≤ MAX_GRAPH_EXPANSION_DEPTH (2), rows ≤ MAX_GRAPH_EXPANSION_ROWS (50), timeout, token budget. Same eligibility predicate applied to expanded nodes. Expand + dedupe, then final-select ≤8 contexts AND token_budget (4096). Final cap guaranteed.

### Dual Cap
Max 8 serving contexts AND token_budget=4096. eval_k separate from serving_k.

### Generator Prompt Wrapper (D47)
Versioned. Receives governing_dates, per-context legal_time_basis, selected intervals/basis, immutable citations, eligible contexts, review_pending flags. Answer ONLY from supplied eligible contexts. Caveated if review_pending. NOT legacy prompt verbatim (legacy biases "Ngày hiện tại").

### Text2Cypher (D18, D68, D92)
Complete v1 TemplateCall enum (no "... extensible"):
```python
class TemplateKind(str, Enum):
    COUNT_ARTICLES_BY_DOC = "count_articles_by_doc"
    LIST_SIGNERS_OF_DOC = "list_signers_of_doc"
    LIST_DOCS_BY_YEAR = "list_docs_by_year"
    COUNT_DOCS_BY_FIELD = "count_docs_by_field"
    GET_DOC_METADATA = "get_doc_metadata"
    LIST_DOCS_BY_ORGAN = "list_docs_by_organ"

class CountArticlesByDocRequest(BaseModel):
    kind: Literal["count_articles_by_doc"]
    legal_document_key: str
    class Config: extra = "forbid"
# ... one typed model per TemplateKind member (all six enumerated)

TemplateCall = Annotated[Union[<all six models>], Field(discriminator="kind")]
```
Server-owned parameterized templates. Each: fixed server Cypher (never model-generated), hard LIMIT, timeout, result schema, output provision_version_id + CitationContext, validity predicate built-in, scoped to pinned release_id, resolve via legal_document_key (ambiguity-rejecting). Invalid kind/params → route to retrieval. No free-form Cypher.

### Execution Model (D51)
Bounded thread pool executor with BOUNDED admission queue (capacity MAX_EXECUTOR_QUEUE=100). GPU semaphore within single API worker. Per-stage timeout/deadline. Saturation → 429/503 + Retry-After. OOM → graceful CPU degradation. Cancellation propagated. Streaming accumulator bounded.

---

## 13. API + SSE Contract

### POST /v1/chat
```typescript
{
  query: string,                  // ≤ MAX_QUERY_CHARS (2000)
  chat_history?: Array<{role:"user"|"assistant", content:string}>,  // ≤ MAX_HISTORY_MESSAGES (40) / MAX_HISTORY_TOKENS (8000)
  as_of_date?: string, occurred_at?: string, ended_at?: string,
  detected_date?: string, violation_state?: "completed"|"ongoing"|"unknown",
  filters?: {doc_type?: string[], field?: string[], organ?: string[]}  // ≤ MAX_FILTER_VALUES_PER_FIELD (10) each
}
```
Total body ≤ MAX_REQUEST_BODY_BYTES (262144). event_date alias → occurred_at; both-different → 422. Validation errors → HTTP problem response BEFORE opening SSE stream.

Response: SSE via fetch() + ReadableStream.

### SSE Event Sequence
`meta` → `sources`? → `token*` → exactly one terminal `done`|`error`.
```
event: meta
data: {"state":"started","trace_id":"...","corpus_version":N,
       "release_fingerprint":"...","publish_fence":N,
       "as_of_date":"...","occurred_at":"...","ended_at":"...",
       "detected_date":"...","violation_state":"...",
       "governing_dates":{...},"selected_regime_ids":[...],
       "streaming_mode":"streaming|buffered"}

event: sources
data: [{"citation":"...","title":"...","official_url":"...",
        "provision_path":[...],
        "selected_interval":{"from":"...","to":null,"state":"in_force"},
        "governing_date_kind":"query_time",
        "applicability_label":"applicable","review_pending":false,
        "snippet":"..."}]

event: token
data: {"text":"..."}

event: done
data: {"trace_id":"...",
       "outcome":"full|degraded|cached|clarified|coverage_unsupported|legal_status_review_pending",
       "degraded_components":[],"warnings":[]}

event: error
data: {"code":"...","message":"...","trace_id":"..."}
```
- meta: immutable request/release fields + state:"started". Canonical factual fields only. Includes selected_regime_ids.
- sources: per-context selected_interval + governing_date_kind + review_pending.
- done: outcome includes legal_status_review_pending.
- Client disconnect: cancel + log; do NOT emit done.
- buffered mode: exactly one token event with full text. Buffered/non-cancellable provider: hold slot to timeout; do NOT claim resources released immediately (D95).

### Streaming + Cache Write Semantics (D95)
Emit token immediately; tee into bounded accumulator. Write answer cache ONLY after terminal done success. Do NOT cache on pre-token failure, mid-stream error, or disconnect.

### POST /v1/feedback (D96)
```typescript
{ trace_id: string, rating: "helpful"|"not_helpful",
  comment?: string, idempotency_key: string }
```
Single-user deployment. Ownership = trace_id must exist in this deployment's trace store (trace existence validation) + idempotency_key dedup. No per-user session ownership (single-user). Returns 200 / 409 (dup idempotency) / 404 (unknown trace).

### Health Endpoints
/livez (no deps), /readyz (descriptor + Qdrant + Neo4j + fulltext readiness + metadata revisions + init lock absent + allow-list populated), /metrics. NO remote LLM call.

---

## 14. Caching (Pillar 4)

### Canonical Cache Keys (D93)
ALL cache keys = `SHA256(canonical JSON object)`. Canonical JSON = sorted keys, separators=(",", ":"), no whitespace. NO delimiter-less string concatenation.

### Cache Key Definitions
- **Answer cache**: `cache:ans:{SHA256(canonical({rewritten_query, intent, filters, governing_dates, selected_regime_ids, violation_state, release_fingerprint, embedding_version, reranker_version, model_name, prompt_wrapper_version, retrieval_policy_fingerprint, runtime_policy_fingerprint}))}`
- **Retrieval cache**: `cache:ret:{SHA256(canonical({decomposed_queries, filters, governing_dates, selected_regime_ids, violation_state, release_fingerprint, embedding_version, reranker_version, retrieval_policy_fingerprint}))}`
- **Context cache**: `cache:ctx:{provision_version_id}:{release_fingerprint}` (raw text)
- **Amends cache**: `cache:amends:{provision_uid}:{release_fingerprint}:{SHA256(canonical(governing_dates))}` (date-resolved result)

retrieval_policy_fingerprint = SHA256(canonical({aggregate, bm25_enabled, rerank_model_version, fetch_k, rerank_top, top_k, heuristic_enabled, code_schema_version})).
runtime_policy_fingerprint = SHA256(canonical({prompt_wrapper_version, generator_config_version})).

### Cache Lookup Ordering + Cypher Branch (D94)
1. Answer cache: after route+rewrite. Applies to BOTH retrieve branch AND cypher_query branch (lookup before execute, write after done success). No dangling cache write.
2. Retrieval cache: after decompose, before search/rerank (retrieve branch only).

### Write Timing (D65, D95)
| Layer | Written When | Contains | TTL |
|---|---|---|---|
| Answer | After terminal done SUCCESS only | answer + sources + intent + policy fingerprints | 24h |
| Retrieval | After retrieval+rerank success (even if gen fails) | provision_version_id refs + score | 6h |
| Context | After context build success | raw text per (provision_version_id, release) | 6h |
| Amends | After amends resolve (date-resolved) | per (provision_uid, release, governing_dates) | 6h |

Invalidation via release_fingerprint in keys. Redis role separation (cache fail-open; descriptor fallback; broker stops ingestion).

---

## 15. Graph Schema + Validity

### Node Scoping (D62, D89)
- **Release-scoped**: Content hierarchy + AMENDS nodes (snapshot_node_id). GC'd with retention.
- **Immutable per metadata_revision**: Metadata nodes + ExternalDocument. Content-addressed by metadata_revision. NOT mutated across releases. Release manifest references specific metadata revisions. N-1 cannot change metadata when N updates. Reconcile checks referenced metadata revision. GC only when unreferenced.

### Relationships
Hierarchy (HAS_*), Metadata (BELONGS_TO_GROUP etc.), RELATED_TO (ExternalDocument global-reference exception), AMENDS (operation, evidence_locator, effective_rule_id, edge_id, review_status, validity_basis). All content relationships within same release.

### Deterministic Edge ID (D63, D74)
Canonical edge tuples per type; edge digest = SHA256(sorted canonical edge tuples). AMENDS edge_id includes evidence identity. Reconcile checks node tuples + edge digest + endpoints.

### Constraints
snapshot_node_id unique (release-scoped); metadata_revision unique (global). MERGE idempotent.

---

## 16. Monitoring + Evaluation (Pillar 5)

### Evaluation Dataset (D67, D98)
```json
{
  "question":"...", "legal_locators":["..."],
  "as_of_date":"...", "occurred_at":"...", "ended_at":null,
  "detected_date":null, "violation_state":"completed",
  "governing_dates":{"conduct_time":"...","query_time":"..."},
  "rule_kind":"completed_conduct_time",
  "resolved_provision_version_ids":["..."],
  "successor_provision_version_ids":["..."],
  "release_fingerprint":"...", "reference_answer":"...", "notes":"..."
}
```
Labels include resolved_provision_version_ids + successor sets + release fingerprint + canonical facts + governing dates/rule kind. Dedup + group-split by legal_document_key BEFORE 80/20.

### Reproducibility (D98)
Retrieval eval FREEZES/REPLAYS router, rewrite, decomposition outputs (golden fixtures). Pins provider/model/params/prompt version. Commits: baseline artifact, split membership, run fingerprint, promotion rule. Resume must not mix outputs from different fingerprints.

### Relevance (D99)
Successor/AMENDS-aware: a retrieved provision_version_id is relevant if it equals OR is a successor (via reviewed AMENDS) of a labeled resolved_provision_version_id. NOT prefix-match UID.

### Metrics + Quality Gate (D100)
eval_k=8; primary metric Recall@8. Also Precision@8, MRR, nDCG@8. Token-budget assertion. Legacy scripts (top_k=30, no Recall@8) deprecated — NOT called validation. Quality gate: Recall@8 drop ≤2% absolute, nDCG@8 drop ≤3% absolute vs pinned baseline. Gates promotion/deploy, not every PR.

### Import-Time Test (D101)
No-network/no-LLM test runs at import time. Bounded retries with deadline; no infinite retry loop.

### Online Observability
Langfuse (self-host, D75) + Prometheus + Grafana. Trace every request. NO high-cardinality labels.

---

## 17. Error Handling

| Component failure | Behavior |
|---|---|
| Release descriptor unavailable (Redis + fallback) | 503 |
| Router LLM | Fallback intent=retrieve |
| Rewriter | Original query |
| Decomposer | `[{query: original}]` |
| Qdrant unavailable | Degraded error (no ANN claim) |
| Neo4j unavailable | Qdrant payload context + citation projection; degraded but functional |
| Both stores | Friendly unavailable |
| Reranker | ANN ordering |
| Generator LLM pre-token | Retry once (bounded); then friendly error |
| Generator LLM mid-stream | NO retry; emit error event |
| Redis cache | Fail-open |
| Redis descriptor | Durable local fallback |
| Langfuse/Prometheus | Fail-open |
| Missing facts | Clarify (pre-retrieval) |
| Date in coverage gap / outside window | historical_coverage_unsupported (coverage manifest) |
| Admission queue saturated | 429/503 + Retry-After |
| event_date/occurred_at conflict | 422 |
| Pending-amendment hit | legal_status_review_pending + warning |
| Unsupported temporal evaluator | temporal_rule_unsupported / clarification |
| INSECURE_TLS=true production | Startup FAIL |
| Empty allow-list | /readyz 503 allowlist_not_populated |
| Metadata/text effectivity conflict | Quarantine with evidence |
| docIdentity collision | Collision map; resolver rejects ambiguity; ingest continues |

All retries bounded with deadline (D101).

---

## 18. Testing Strategy

### Unit Tests (No Network, No Services)
- Parser hierarchy: alphanumeric clause (2a, 18a); footer split; quote-block; NFC; fixtures.
- Identity: legal_document_key collision map; provision_uid building; provision_version_id FULL hash (no truncation); collision registry test (distinct texts → distinct version IDs); snapshot_node_id; UUIDv5; round-trip; resolver ambiguity rejection.
- Source revision identity: parse idempotency changes with raw/parser/normalizer; manifest path from full identity; raw-body-changed-metadata-same → new revision.
- Temporal filter compiler: 3 static branches; absent date omitted; regime_id inside branch; same-element evaluation; Qdrant + Neo4j compilation equivalence.
- Regime resolution: NĐ100 vs NĐ168 by conduct_time; selected_regime_ids correctness.
- Effectivity: effectDate baseline; expireDate outer cap; ledger override precedence; metadata/text conflict → quarantine; reviewed-overrides promotion.
- Validity intervals: governing_date_kind per interval; no overlapping applicable intervals; static general truncated at exception.
- AMENDS discriminated union: schema validation (one-of/at-least-one); effective_rule_id validation; PatchSpec expected_successor required; edge_id uniqueness.
- PendingAmendmentRecord: schema; auto/quarantined source; independent lookup (not gated on top-k); review_pending derivation.
- AMENDS state machine: auto/quarantined/review_pending/reviewed transitions.
- Cache keys: canonical JSON; all dependency dimensions; version bump → key change; cypher branch cache.
- Eligibility predicate: boundary cases; half-open [from,to); conditional/unknown excluded.
- Per-context applicability + legal_time_basis.
- Coverage manifest: gap detection; verified_from/to; historical_coverage_unsupported.
- RRF aggregation; heuristic rerank; query normalization (no stopwords); canonical JSON filters.
- Text2Cypher: complete v1 enum; extra="forbid"; resolve via legal_document_key; output provision_version_id + CitationContext; invalid → retrieval.
- Subquery dedup + max_subqueries; token budget; SSE serialization; requires_clarification; prompt wrapper; ReleaseDescriptor.fingerprint; publish_fence separate from corpus_version; bounded admission; event_date normalization/conflict; embedding_input_fingerprint; BM25 query plan (release_id before LIMIT); graph expansion bounds; import-time test no-network/no-LLM; no infinite retry.

### Integration Tests (Docker-Compose Test Profile)
- Import → node/rel counts + constraints.
- Embed → Qdrant points + filter + context + citation + governing_date_kind + regime_id.
- E2E retrieval → expected top_k provision_version_ids.
- Reconcile: node tuples + edge digest + fulltext readiness + metadata revision; partial failure → no publish.
- Tombstone: interval closed, evidence preserved.
- Cache invalidation on new release.
- Atomic publish CAS.
- Parse fail → carry-forward last-good (full projection preserved).
- AMENDS cascade; eligibility before/after amendment.
- Staging isolation; unchanged carry-forward.
- Rollback with new fence; stale-after-rollback prevented; concurrent build; stale build; equal-base fence; interrupted CAS; Redis restart.
- Same-day boundary; multi-interval same-element; NĐ168 general/exception/conditional/completed-vs-ongoing; static exception no overlap; predecessor/successor regime.
- ADD/REPLACE/REPEAL; metadata-only revision; normalizer revision; raw-body-changed-metadata-same.
- Degraded Qdrant citations; Redis descriptor fallback; fresh-volume bootstrap (exclusive init); two-container GPU (single worker).
- BM25 release isolation (N-1 not crowd N); fulltext readiness; expansion bounds + final cap.
- Generator historical governing_dates + per-context basis; future-effective ingest.
- Historical cache no collision; completed vs ongoing; interval before/on/after repeal; failed/disconnected stream not cached; relationship reconcile; clarify path; coverage unsupported (gap).
- Pending lookup: auto amendment after effective date → legal_status_review_pending + warning even when target not in top-k.
- Conditional reviewed materializes in_force; AMENDS effective_rule validation; per-context governing_date_kind; amends cache date-resolved; temporal_rule_unsupported; INSECURE_TLS production fail.
- Collision map: docIdentity collision → distinct legal_document_keys; resolver ambiguity reject; ingest continues.
- provision_version_id full hash: no truncation collision in integration fixtures.
- Metadata immutability: N update doesn't mutate N-1 metadata; reconcile checks referenced revision.
- Eval freeze/replay: golden fixtures for router/rewrite/decompose; run fingerprint; promotion rule; Recall@8 computed.

### Acceptance Tests (Concrete Scenarios)
| Scenario | Expected |
|---|---|
| Before/after amendment | occurred_at before/after → different applicable provision |
| Event_date transition | NĐ100 vs NĐ168 by occurred_at |
| Amendment adds 2a | ADD_PROVISION numbering 2a retrievable + cited |
| Repeal cascades | REVIEWED REPEAL → children intervals closed |
| Partial-ingest not exposed | Truncated fetch → last-good kept, no tombstone |
| Parse fail carry-forward | Doc not lost from full projection |
| Stale payload after amendment | Hash recomputed; vector reused if fingerprint matches |
| Truncated parser no tombstone | Last-good retained |
| SSE disconnect | Cancel, no done, NOT cached |
| Cache-hit fresh trace | New trace_id; cache_origin_trace_id |
| Fresh-volume bootstrap | Exclusive init; consistent serving |
| Staging isolation | Ingest N, serve N-1 |
| Rollback no cross-release | New fence; stale can't re-publish |
| Same-day boundary | [from,to) selects new provision |
| Multi-interval same-element | Each interval evaluated independently |
| NĐ168 conditional | Excluded + warning |
| NĐ168 exception 2026 | General truncated; no overlap |
| NĐ168 completed vs ongoing | conduct_time vs detection_time |
| Predecessor/successor regime | Traced via regime_id |
| Regime resolution | conduct_time selects valid regime only |
| Static temporal filter | 3 branches compiled; Qdrant+Neo4j equivalent |
| Degraded citations | Neo4j down → Qdrant payload + governing_date_kind |
| Redis descriptor outage | Local fallback |
| Metadata-only revision | New manifest |
| Normalizer revision | New normalized artifact; offsets bound |
| Raw-body-changed-metadata-same | New revision + manifest |
| Two-container GPU | Single worker; no contention |
| N+N+1 coexist | Independent queries |
| Historical cache no collision | Distinct keys |
| Future-effective ingest | Excluded until effective |
| Clarify path | One token + done, pre-retrieval |
| Coverage gap | historical_coverage_unsupported |
| BM25 release isolation | N-1 not crowd N |
| Expansion bounds | Depth/rows/timeout/token enforced; ≤8 final |
| Generator historical basis | Historically-grounded |
| Failed stream not cached | Mid-stream error → no answer cache |
| Concurrent build | One CAS succeeds |
| Stale publish | CAS returns 0 |
| Stale-after-rollback | Fence mismatch → CAS 0 |
| Equal-base fingerprint | Higher fence wins |
| Interrupted CAS | Restart from Redis authority |
| Redis restart | Descriptor survives |
| Manual rollback | New fence + audit log |
| Model revision change | Re-embed |
| Pending-amendment warning | legal_status_review_pending even when target not in top-k |
| Metadata/external scope | Immutable per revision; not GC'd while referenced |
| Edge-digest reconcile | Corrupted edge → publish blocked |
| Answer cache policy fingerprint | Includes retrieval + runtime |
| Retrieval cache on gen failure | Written despite gen failure |
| Cypher branch cache | Lookup + write consistent |
| Parser/normalizer revision | New source revision identity |
| event_date conflict | 422 |
| Conditional reviewed materializes | in_force interval created |
| AMENDS effective_rule validation | Exists/reviewed/scope |
| Per-context governing_date_kind | SSE sources per-item |
| Amends cache date-resolved | Key includes governing_dates |
| Temporal rule unsupported | Clarification |
| INSECURE_TLS production | Startup fail |
| Collision map | Distinct keys; resolver reject; ingest continues |
| provision_version_id full hash | No truncation |
| docIdentity collision no data loss | All docs ingested |
| Metadata immutability | N-1 unchanged by N |
| Allow-list empty | /readyz 503 |
| Reverse-amendment discovery | Forward closure works; limitation documented |
| Eval Recall@8 | Primary metric; legacy top_k=30 deprecated |
| Eval freeze/replay | Golden fixtures reproducible |
| Import-time test | No network/LLM; bounded retries |

### CI
pytest unit per PR (no network); integration on parser/importer/retrieval/security/identity/release changes; acceptance on release candidates; eval on retrieval-change PRs (LLM-judge nightly/manual); security tests (template injection, credential leaks, secret scan) per PR; quality gate on promotion.

---

## 19. Delivery, Security, Repo Truth

### Repo-Truth Delivery Acceptance (D102)
**Current state (as of this draft):** `NLP-LegalQA/` exists as an untracked nested directory; `.gitmodules` and gitlink are NOT committed. The spec's submodule claim is a TARGET, not yet met.

**Mandatory acceptance criteria (must pass before claiming delivery):**
1. Commit `.gitmodules` with submodule entry for `NLP-LegalQA` at pinned SHA.
2. Commit gitlink (tree entry) referencing that SHA.
3. CI checks out with `--recurse-submodules`.
4. CI asserts pinned SHA matches expected.
5. Until all four pass, spec MUST NOT claim "production-ready" or submodule-wired.

### Security (D13, D103)
- **Secret scan**: CI runs gitleaks/trufflehog on every PR. Fail on detected secrets.
- **Reference credentials**: NLP-LegalQA's CLAUDE.md contains real-looking credentials (`nguyenhoangquan.com:7687`, password `Neoneo4j` at lines ~217–219). These are in the READ-ONLY reference and MUST NOT be copied into this project, tests, docs, or compose. Our `.env.example` uses `<placeholder>` only. Flag for the reference repo owner to rotate; this project never propagates them.
- **TLS**: Default verify ON. `INSECURE_TLS=true` fails production startup; dev profile only. Legacy `verify=False` from reference NOT adopted.
- **Neo4j principals**: API read-only; worker write/schema/publish. Separate credentials.
- **Secrets policy**: env vars only; `.env` gitignored; test placeholders; docs `<placeholder>`.

### Pinning + Langfuse + GPU + Volumes
Everything pinned (Python, Node, base image digests, service tags, HF model revisions, submodule SHA). Langfuse self-host: compose closure, persistent volumes (postgres/clickhouse/minio), backup/restore, resource profile (min 4GB RAM/2 CPU/20GB disk). GPU: single API worker process (in-process semaphore suffices); multi-worker requires distributed lease (out of v1 scope, documented). Volumes/backup per §8/§16.

### Open Items (Environment/Data Dependent Only)
1. Langfuse resource budget vs host machine (profile documented; allocation host-dependent).
2. HuggingFace model first-time download (cached in volume).
3. UpdDateTime capability probe result (fallback if unsupported).

**Note**: exact allow-list IDs are NOT an Open Item — they are a bootstrap requirement (§6/D84) that must complete and be committed before serving. Multi-worker GPU coordination is explicitly out of v1 scope.

---

## 20. Resolved / Remaining / Explicitly Out of Scope

### Resolved (contract + test exist in this draft)
| Item | Contract § | Test § |
|---|---|---|
| Qdrant static temporal filter (no dynamic expr) | §5.1/D78 | unit: temporal filter compiler |
| Regime resolution before retrieval | §5.2/D79 | unit+integration: regime resolution |
| Independent pending-amendment lookup | §10.2/10.3/D80/D104 | unit+acceptance: pending warning not top-k-gated |
| Metadata-baseline effectivity + expireDate cap | §5.3/D81 | unit: effectivity precedence |
| Reviewed-overrides promotion | §5.4/D82 | unit: promotion workflow |
| Coverage manifest + gaps | §5.5/D83 | unit+acceptance: coverage gap |
| Concrete corpus scope + reverse-amendment discovery | §6/D84 | acceptance: reverse discovery + limitation |
| Delta scan (content hash + watermark, not date cursor) | §6/D85 | integration: delta |
| Parse fail carry-forward | §3/D86 | integration: carry-forward |
| legal_document_key collision map | §4.1/D87 | unit+acceptance: collision no data loss |
| provision_version_id full hash | §4.3/D88 | unit: collision registry |
| Metadata/ExternalDocument immutable per revision | §15/D89 | integration: metadata immutability |
| BM25 release_id+validity before LIMIT | §12/D90 | integration: BM25 release isolation |
| Graph expansion bounds | §12/D91 | integration: expansion bounds |
| TemplateCall complete v1 enum | §12/D92 | unit: complete enum |
| Canonical cache keys | §14/D93 | unit: canonical JSON keys |
| Cypher branch cache consistency | §13/14/D94 | unit: cypher cache |
| Streaming + cache write semantics | §13/D95 | integration: failed stream not cached |
| Feedback ownership (single-user) | §13/D96 | unit: feedback validation |
| Concrete limits | §13/D97 | unit: limit enforcement |
| Reproducibility freeze/replay | §16/D98 | integration: golden fixtures |
| Successor-aware relevance | §16/D99 | integration: relevance |
| eval_k=8 Recall@8 | §16/D100 | integration: Recall@8 |
| Import-time no-network test | §16/D101 | unit: import-time |
| Repo-truth delivery acceptance | §19/D102 | CI acceptance |
| Secret scan + credential non-propagation | §19/D103 | CI secret scan |
| publish_fence CAS | §8/D61 | integration: concurrent/stale/rollback |

### Remaining (must be done during implementation; not design gaps)
| Item | Owner | Gate |
|---|---|---|
| Populate allow-list doc_group_ids/field_ids via facet discovery | Implementation bootstrap | /readyz allowlist_not_populated until done |
| Pin exact HF model revision hashes | Implementation | Manifest fingerprint |
| Generate Langfuse secrets + resource allocation on host | Deployment | Langfuse compose up |
| Wire .gitmodules + gitlink (currently untracked) | Implementation | CI SHA assertion |

### Explicitly Out of Scope (v1)
| Item | Reason |
|---|---|
| Multi-worker GPU coordination (distributed lease) | Single API worker suffices for v1 |
| Fine-tuning training pipeline | Eval-only |
| Free-form Cypher generation | Template-only for safety |
| Per-document custom temporal evaluator | RuleKind enum only |
| Multi-tenant auth | Single-user |
| OCR | Structured source |
| Reverse-amendment discovery beyond reachable docs | Stated coverage limitation |
| Distributed ANN / sharding | Bounded corpus |

---

## Changelog: v7 → v8

| Area | v7 Issue | v8 Fix |
|---|---|---|
| Temporal filter | Dynamic `governing_dates[interval.governing_date_kind]` not executable in Qdrant | Static compiler emits 3 nested-filter branches (query_time/conduct_time/detection_time) with concrete dates; Qdrant+Neo4j identical; regime_id inside branch (D78) |
| Regime resolution | Absent | Resolve selected_regime_ids before retrieval; added to context/cache key/predicate/citation (D79) |
| review_pending | Lookup after candidates → misses targets filtered from top-k | PendingAmendmentRecord schema; independent lookup by locator/provision_uid before/parallel to retrieval; warning even if target not in top-k (D80/D104) |
| AMENDS states | Relationship unclear | Explicit state machine auto/quarantined/review_pending/reviewed; raw pending doesn't require reviewed rule_id; only reviewed materializes (D105) |
| Effectivity | Only ledger/text; ignored metadata effectDate/expireDate | effectDate=baseline, expireDate=outer cap, ledger overrides higher precedence; conflict→quarantine; reviewed-overrides artifact + promotion workflow (D81/D82) |
| Coverage | temporal_support_window_from only | CoverageManifest: verified_from/to, gaps, per-regime/document, watermark; coverage-based historical_coverage_unsupported (D83) |
| Corpus scope | "See v6 §6 (unchanged)"; allow-list IDs in Open Items | §6 fully written: allow-list schema, bootstrap requirement, dependency closure, reverse-discovery + limitation, exclusion evidence (D84) |
| Delta scan | issueDate/effectDate cursor risk | content_hash + updDateTime probe + overlap/watermark + full reconciliation (D85) |
| Parse fail | Could lose doc from projection | Carry-forward last-good; full projection preserved (D86) |
| legal_document_key | "validated unique" only | Stable audited collision map; suffixed keys on collision; resolver rejects ambiguity; ingestion never loses docs (D87) |
| provision_version_id | 12-char truncation | Full SHA256; collision registry test (D88) |
| Metadata/ExternalDocument | Global mutable risk across releases | Immutable per metadata_revision; release references revisions; reconcile checks; GC only when unreferenced (D89) |
| BM25 | No exact query plan | Exact Cypher: release_id + validity BEFORE LIMIT; max_fulltext_scan; N-1 can't crowd N (D90) |
| Graph expansion | No bounds | Depth/row/timeout/token bounds before materialize; final ≤8 guaranteed (D91) |
| TemplateCall | "... extensible" in running contract | Complete v1 enum (6 templates), all typed models enumerated (D92) |
| Cache keys | Delimiter-less string concat | SHA256(canonical JSON object) everywhere (D93) |
| Cypher cache | Write without lookup | Cypher branch lookup+write consistent with retrieve (D94) |
| Streaming | Ambiguous cache write | Emit token immediately; tee bounded accumulator; cache only after done success; buffered holds slot to timeout (D95) |
| Feedback ownership | "validate ownership" without session | Single-user: trace existence + idempotency; no per-user session (D96) |
| Limits | Not concrete | All limits set: body 262144B, history 40msg/8000tok, query 2000ch, filter 10/field, decompose 6, expansion depth 2/rows 50, queue 100, fulltext scan 500, k=8, budget 4096 (D97) |
| Eval reproducibility | Not frozen | Freeze/replay router/rewrite/decompose; pin provider/model/params/prompt; baseline artifact, split, fingerprint, promotion rule (D98) |
| Relevance | Prefix-match UID risk | Successor/AMENDS-aware via resolved + successor version IDs (D99) |
| eval_k | top_k=30 legacy | eval_k=8, Recall@8 primary; legacy deprecated (D100) |
| Import-time test | Missing | No-network/no-LLM import-time test; bounded retries, no infinite loop (D101) |
| Repo truth | Claimed submodule not wired | Current untracked state documented; mandatory acceptance (.gitmodules+gitlink+recursive+SHA assert); no production-ready claim until met (D102) |
| Security | Reference creds; verify=False | Secret scan; reference creds flagged not propagated; TLS default ON (D103/D13) |
| Structure | No resolved/remaining table | §20 Resolved/Remaining/Out-of-scope table added |
