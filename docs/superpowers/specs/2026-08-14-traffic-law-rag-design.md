# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-15  
**Status:** Draft v9 (post-Codex round-8 revision — atomic release reservation, per-kind regime mapping, immutable per-release artifacts, full schemas for every contract, real corpus/eval numbers, repo-truth honesty)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (target: vendored as git submodule at pinned SHA, read-only reference — NOT yet wired, see §19).

**THIS IS A DESIGN TARGET, NOT A PRODUCTION-READY SYSTEM.** Verified repo state at v9 authoring time: `git ls-files` returns exactly **1 tracked file** (this spec). `NLP-LegalQA/` is **untracked**. There is no `src/`, no `frontend/`, no `docker-compose.yml`, no `tests/`, no `eval/`, no `.gitmodules`. No claim of production readiness is valid until §19 acceptance gates pass.

---

## 1. Goals and Non-Goals

### Goals
- Production-grade graph-RAG for Vietnamese traffic law covering five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Amendment-aware, temporally-correct retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`.
- Right-sized for a bounded corpus. No fake scale claims.
- **Atomic release lifecycle**: release_id reserved atomically; staging isolated per build_id; all release-referenced artifacts immutable and hashed before reconcile; promote only after reconcile; CAS with ownership-token bootstrap.
- **Executable temporal semantics**: per-governing-date-kind regime mapping; deterministic interval split algorithm; static compiled filters; explicit inclusive/exclusive date conversion.
- **Legal safety**: pending amendments never silently ignored; unresolved effective dates surface as explicit outcomes; no confident penalty numbers when a pending amendment could change them.
- **Every contract has schema, serialization, failure behavior, and acceptance test.**

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters.
- Fine-tuning training pipeline (eval can score fine-tuned models; training not shipped).
- Multi-tenant auth. Single-user deployment in v1.
- OCR (source API returns structured HTML/JSON).
- LLM-based legal validation. Pydantic validates structure, not legal meaning.
- Free-form LLM-generated Cypher.
- Free-text temporal condition evaluation.
- Reverse-amendment discovery beyond reachable documents (explicit coverage limitation, surfaced to users).

---

## 2. Decisions Log

Decisions D1–D105 carry forward from drafts v1–v8 unless superseded below. New/superseding decisions in v9:

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D106 | Atomic release_id reservation | `release_id = INCR descriptor:release_ctr` (atomic). Staging writes go to build-scoped names: Qdrant `staging_{build_id}`, Neo4j `build_id` property. Promote (rename/alias + relabel) only AFTER reconcile passes. Two workers can never write the same staging namespace. | publish_fence alone does not prevent staging corruption |
| D107 | build_id | `build_id = UUIDv4` per build attempt. Distinct from release_id. Staging Qdrant collection = `staging_{build_id}`; Neo4j staging nodes carry `build_id` + `pending_release_id`. | Isolates concurrent builds |
| D108 | Immutable release artifacts | release_manifest, coverage_manifest, pending_index, reviewed_overrides, review_log_snapshot, metadata_revision_set — each written immutably (content-addressed) and hashed BEFORE reconcile/CAS. Descriptor carries all hashes. No artifact write after publish. | Same release always yields same results |
| D109 | Pending index immutable per release | pending_index is a per-release immutable artifact with hash in manifest + descriptor. Redis `descriptor:pending` is a CACHE ONLY; a mutable Redis value must never change a published release's behavior. | Deterministic pending detection |
| D110 | Backup/restore ordering | Restore order: raw provenance → metadata revisions → reviewed_overrides → review_log → coverage_manifest → pending_index → Neo4j → Qdrant → release_manifest → **then** activate pointer. Pointer activation last. | No partial-visible restore |
| D111 | CAS bootstrap ownership token | Bootstrap uses `SET descriptor:init_lock <owner_token> NX EX <ttl>`. Lua CAS verifies `GET init_lock == owner_token`, not merely EXISTS. | Prevents foreign-lock publish |
| D112 | Per-kind regime mapping | `regime_map: {query_time: [regime_ids], conduct_time: [regime_ids], detection_time: [regime_ids]}`. Each compiled nested filter branch uses the regime set for THAT kind. No single collapsed reference date. | Prevents mixing current law with law-at-conduct |
| D113 | Empty regime set behavior | If a governing_date_kind has an empty regime set (no reviewed transition rule covers it), that branch is omitted. If ALL branches are omitted → `temporal_rule_unsupported` outcome + clarification. Never silent empty retrieval. | Honest failure |
| D114 | Intent→governing-kind binding | Explicit table (§5.4): sanction/penalty intents bind to conduct_time (completed) or detection_time (ongoing); "what is the current rule" intents bind to query_time. A single request never mixes kinds across contexts unless the user explicitly asks for a comparison, which is served as separate labeled branches. | No accidental law mixing |
| D115 | Date semantics | Source `effectDate` is INCLUSIVE start → `interval.from = effectDate` (date-only, Asia/Ho_Chi_Minh). Source `expireDate` is INCLUSIVE last day of validity → `interval.to = expireDate + 1 day` (half-open). Null effectDate → `unknown` state, not eligible. Null expireDate → `to = None` (open-ended). `expireDate < effectDate` → quarantine with evidence. Metadata/text conflict → quarantine. | Exact `[from,to)` conversion |
| D116 | Interval split algorithm | Deterministic algorithm (§5.6) handles static general + N static exceptions: build boundary set, sort, emit non-overlapping segments, assign most-specific rule per segment by (scope specificity, precedence, entry_id) total order. | No hand-wavy "truncate" |
| D117 | Full temporal schemas | EffectivityLedgerEntry, CoverageGap, RegimeCoverage, DocumentCoverage, ValidityInterval, GoverningDates, RegimeMap — all with typed enums and canonical JSON serialization. | Executable |
| D118 | Unresolved pending effective date | `planned_effective_date = None` MUST NOT be skipped. Match rule: a pending record matches if `planned_effective_date is None` (unknown → always potentially relevant) OR `planned_effective_date <= governing_date[record.governing_date_kind]`. Unknown-date match → `temporal_rule_unsupported` if the answer would depend on it, else `legal_status_review_pending`. | Never silently ignore |
| D119 | Pending lookup from free text | Pending lookup resolves candidate targets from (a) explicit legal_locators parsed from the query, (b) legal-locator regex extraction, (c) a lexical/keyword index over pending records' target citation text + amending excerpt, (d) provision_uids from retrieval candidates. Union of all four. | Works without known locator |
| D120 | Sanction-safety rule | For penalty/sanction intents where a matched pending amendment could change the outcome, the generator MUST NOT emit specific amounts or a confident conclusion. Response is warning/clarification/review-pending WITH citations to the current provision and the pending amendment evidence. | Legal safety |
| D121 | Pending record schema | Canonical target, source amendment, evidence, governing_date_kind, unresolved_reason enum, release_fingerprint (§10.2). | Executable |
| D122 | Pending state machine | raw_candidate → auto \| quarantined (persisted edge states) → review_pending (SERVE-TIME derivation only, never a persisted edge state) → reviewed \| rejected (persisted). | Clear lifecycle |
| D123 | Full AMENDS schemas | Complete Pydantic for AddProvisionOp, ReplaceTextOp, RepealOp, AmendOp, PatchSpec, LegalLocator, plus successor_provision_version requirement. | No name-only ops |
| D124 | EvidenceLocator offsets | Includes `source_revision_identity_hash` (manifest id), parser_version, normalizer_version, `offset_unit: Literal["unicode_code_points"]`, start/end offsets, `excerpt_encoding: "utf-8"`, excerpt_sha256. Offsets validated against the referenced normalized artifact before review/apply. | Unambiguous evidence |
| D125 | reviewed_overrides schema | promotion_id, reviewer, decided_at, decision enum, target_scope, effective_interval, evidence, source_review_log_entry_id, artifact hash (§10.6). | Machine-readable promotion |
| D126 | Allow-list bootstrap artifact | Allow-list IDs are a REQUIRED bootstrap artifact (`allowlist_bootstrap.json`, content-addressed, version + hash recorded in release descriptor). No `<placeholder>` in a committed allow-list. Serving refuses to start without it. | Honest concreteness |
| D127 | corpus_coverage_incomplete | When dependency/corpus closure is unproven for the answer's scope, outcome = `corpus_coverage_incomplete`, gap is injected into the prompt and surfaced in citations/warnings. Never answer as if corpus were complete. | Honest coverage |
| D128 | Coverage gap taxonomy | `CoverageGapKind = {temporal, corpus_dependency, parse_quality, unresolved_amendment}`. Coverage checked per (governing_date_kind, regime_id, document), not a single global window. | Precise coverage |
| D129 | Delta scan contract | `latest_source_watermark` = max observed `updDateTime` in UTC ISO-8601, stored per allow-list scope with an overlap window (default 7 days), cursor persisted in the release manifest, bounded retries, plus periodic full reconciliation (default weekly). | Executable delta |
| D130 | Parse-fail full carry-forward | Carry forward document node, all provision versions, metadata revision reference, pending edges, and coverage entries from last-good release. Not text only. | No partial loss |
| D131 | Metadata node key | `metadata_node_key = SHA256(canonical({source_document_id, source_metadata_revision_hash}))`. Metadata JSON hash alone can collide across documents. | Collision-safe |
| D132 | ExternalDocument identity/version | Key = `SHA256(canonical({source_document_id}))` when docGUID known, else `SHA256(canonical({normalized_citation_text}))` with `identity_kind` enum recorded. Version = `external_revision` monotonic per observed change; stubs immutable per revision. | Stable stubs |
| D133 | Raw manifest full JSON schema | §7 defines complete schema: source URL, fetch time, hashes, parser/normalizer versions, evidence offsets, content length, parse quality metrics, provenance chain. | Executable provenance |
| D134 | GC reference guard | GC never deletes raw/evidence artifacts referenced by: any retained release manifest, review_log, reviewed_overrides, eval label, or emitted citation registry. | No dangling evidence |
| D135 | QA label migration map | Migration from comma-separated `reference` strings (real format, verified) to structured labels with source_document_id + legal_document_key + provision_version_id per release. Undated or repealed-target questions marked `historical` or `ambiguous`, or excluded from held-out. | Real dataset fix |
| D136 | Qdrant filter fixture test | Compiled filters expressed in exact qdrant-client models (`Filter/Nested/FieldCondition/MatchValue/Range/IsNullCondition`), with a fixture test executed against a real Qdrant instance in the integration profile. | Syntax proven |
| D137 | BM25 executable bounds | Overfetch limit, query timeout, row cap, and budget-exceeded behavior defined (§12.1). Enforced via Cypher `LIMIT` on the fulltext yield plus transaction timeout, not prose. | Bounded scan |
| D138 | DB-level expansion bounds | Expansion depth/rows/timeout enforced INSIDE the Cypher query (bounded variable-length pattern + LIMIT + tx timeout), not post-fetch truncation. | No large fetch |
| D139 | Full degraded citation | Qdrant-only citation payload includes title, official_url, human_citation, provision_path, selected_interval, regime_id, rule_kind, effective_rule_id, governing_date + kind, applicability_basis, pending_evidence. | Neo4j-down parity |
| D140 | SSE sources carry full basis | `sources[].legal_time_basis` is the complete LegalTimeBasis object per context. | Complete contract |
| D141 | Complete API models | CitationContext, LegalLocator, ProblemDetails, Warning, FeedbackResponse, Outcome enum, ErrorCode enum — all fully defined (§13). | No placeholders |
| D142 | Outcome enum completeness | Includes `full, degraded, cached, clarified, no_eligible_context, coverage_unsupported, corpus_coverage_incomplete, temporal_rule_unsupported, legal_status_review_pending, generator_failed`. | Every path named |
| D143 | Streaming order fixed | Flow diagrams and prose state: emit tokens immediately → tee to bounded accumulator → on terminal `done` success write answer cache. No "write cache → stream". | Consistent |
| D144 | Buffered provider slot | For buffered/non-cancellable providers, the executor slot and accumulator are held until provider timeout; disconnect does NOT release resources immediately and the spec does not claim it does. | Honest |
| D145 | Local durable trace registry | Feedback validates against a LOCAL durable trace registry (SQLite file on `manifest_store` volume) written on every request, independent of Langfuse. Langfuse fail-open never loses feedback linkage. | Feedback integrity |
| D146 | SSE behavior per outcome | Explicit pre-stream vs post-stream behavior table (§13.5) for clarify, coverage_unsupported, no_eligible_context, pending-warning, generator_failed. | Deterministic protocol |
| D147 | Date validation rules | `occurred_at <= ended_at`; `detected_date >= occurred_at` when both present; date-only normalization to Asia/Ho_Chi_Minh; `violation_state=completed` forbids detected_date-only input; `violation_state=ongoing` requires detected_date; conflicts → 422 with ProblemDetails. | Executable validation |
| D148 | Canonical keys all layers | Context and amends cache keys also use `SHA256(canonical JSON object)`. No `:`-joined string keys anywhere. | Uniform |
| D149 | Canonicalization ordering | Filters, selected regimes, decomposed queries, and any logical set are sorted deterministically before hashing; lists whose order is semantically meaningful are preserved and marked. | Deterministic keys |
| D150 | Eval freeze/replay fixtures | Golden fixtures for router/rewrite/decompose keyed by `SHA256(canonical({query, history, prompt_version, model_id, temperature, max_tokens}))`; CI replays fixtures with no live LLM. | Reproducible |
| D151 | Real eval dataset | `eval_dataset_v1` built from verified real data: `QA_NLP.csv` = **94 rows / 94 unique questions**; combined with `QA_Part2345.csv` = **294 rows / 293 unique (1 duplicate)**; **100 rows** have comma-separated multi-references; referenced doc identities = **100/2019/NĐ-CP, 123/2021/NĐ-CP, 15/2012/QH13, 168/2024/NĐ-CP, 35/2024/QH15, 36/2024/QH15, 44/2019/QH14, 67/2020/QH14**. NĐ 100 labels marked historical. | Real numbers, not invented |
| D152 | Legacy evaluator removed from CI | Prefix-match-UID / top_k=30 evaluators are DELETED from the CI path (not merely labeled deprecated). CI fails if they are invoked. | Enforced |
| D153 | Release-aware relevance | Successor/AMENDS relevance computed per (release, applicable interval, governing_dates), not a static successor graph. | Correct |
| D154 | Langfuse full closure | Service map lists langfuse-web + langfuse-worker + postgres + clickhouse + valkey/redis + minio, with healthchecks, pinned images, named volumes, secrets, internal network, backup/restore. | Real deployment |
| D155 | Exposure policy | Only `nextjs` and `api` publish host ports. Neo4j, Qdrant, Redis, and all Langfuse services are internal-network only. | Least exposure |
| D156 | /readyz full checks | Active descriptor + coverage hash + pending hash + metadata revision set + fulltext smoke query + Qdrant nested-filter smoke query + allow-list artifact + release reconciliation status. | Real readiness |
| D157 | Repo-truth statement | Spec states verified `git ls-files` = 1 file, `NLP-LegalQA/` untracked, no src/frontend/compose/tests/eval/.gitmodules. No production-ready claim until acceptance passes. | Honesty |
| D158 | Workspace-wide secret scan | Secret scan covers the whole workspace including the reference submodule path and docs. Reference credentials (verified present in `NLP-LegalQA/CLAUDE.md` lines ~217–219: host `nguyenhoangquan.com:7687`, password `Neoneo4j`) are flagged for rotation by the reference owner and never copied. Serving path never uses `verify=False`. | Security |
| D159 | Root CI requirements | CI includes a compose test profile job; unit tests import no DB/LLM clients at import time; all retries bounded with deadlines. | Safe CI |

---
## 3. Architecture Overview

### 3.1 Service Map (full closure, D154/D155)
```
┌──────────────────────── docker-compose (single machine) ────────────────────────┐
│ EXPOSED TO HOST (only these two)                                               │
│  ├─ nextjs        :3000  React chat UI, SSE via fetch() ReadableStream          │
│  └─ api           :8000  FastAPI, single uvicorn worker                         │
│                                                                                │
│ INTERNAL NETWORK ONLY (no host ports)                                          │
│  APP                                                                            │
│  └─ arq-worker           ingestion; Neo4j write/schema/publish principal        │
│  STORES                                                                         │
│  ├─ neo4j         5.x    graph + Lucene fulltext; healthcheck: cypher-shell ping│
│  ├─ qdrant        1.x    per-release collections; healthcheck: /readyz          │
│  └─ redis         7.x    arq broker | cache | descriptor Hash | counters        │
│                          AOF on, maxmemory-policy noeviction                    │
│  LANGFUSE STACK (self-host closure)                                            │
│  ├─ langfuse-web         UI + ingestion API; healthcheck: /api/public/health    │
│  ├─ langfuse-worker      async event processing                                 │
│  ├─ langfuse-postgres    metadata store; volume langfuse_postgres               │
│  ├─ langfuse-clickhouse  analytics store; volume langfuse_clickhouse            │
│  ├─ langfuse-valkey      queue/cache for langfuse; volume langfuse_valkey       │
│  └─ langfuse-minio       blob storage; volume langfuse_minio                    │
│  OBSERVABILITY                                                                  │
│  ├─ prometheus           scrapes api:8000/metrics; volume prometheus_data       │
│  └─ grafana              dashboards; volume grafana_data                        │
└────────────────────────────────────────────────────────────────────────────────┘
All images pinned by digest. All secrets via env file (gitignored). Every service
has a healthcheck; api depends_on healthy neo4j/qdrant/redis.
```

### 3.2 INVAR-SNAP (Snapshot Isolation)
Every serving request pins exactly one `ReleaseDescriptor` at ingress. All reads — Qdrant collection, Neo4j `release_id`, BM25, expansion, templates, pending index, cache keys, citations, SSE meta — use that pinned descriptor. No read crosses releases.

### 3.3 Request Path
```
POST /v1/chat
  ▼ VALIDATE (§13.1 limits; §13.6 date rules) — failures → HTTP ProblemDetails, no stream
  ▼ NORMALIZE: event_date alias → occurred_at (conflict → 422)
  ▼ create trace_id; write LOCAL TRACE REGISTRY row (D145)
  ▼ PIN ReleaseDescriptor (Redis Hash authority → local file fallback → 503)
  ▼ DERIVE GoverningDates (§5.2)
  ▼ BUILD RegimeMap per kind (§5.3); empty-all → temporal_rule_unsupported (D113)
  ▼ BIND intent→governing kind (§5.4)
  ▼ COVERAGE CHECK per (kind, regime, document) (§5.8)
        → coverage_unsupported | corpus_coverage_incomplete when applicable
  ▼ route (LLM temp=0, max_tokens=64) → {greeting, cypher_query, retrieve, reject, clarify}
  ├─ reject   → refusal, done(outcome=full)
  ├─ greeting → deterministic text, done(outcome=full)
  ├─ clarify  → ONE deterministic token, done(outcome=clarified)   [no retrieval]
  ├─ cypher_query:
  │     rewrite → ANSWER CACHE LOOKUP → TemplateCall (§12.4)
  │     → execute parameterized template (pinned release, validity predicate, LIMIT, tx timeout)
  │     → emit tokens immediately, tee to bounded accumulator
  │     → on terminal done SUCCESS: write answer cache
  └─ retrieve:
        rewrite → ANSWER CACHE LOOKUP (§14.2)
        decompose (≤ MAX_DECOMPOSED_QUERIES, dedup) → RETRIEVAL CACHE LOOKUP
        PARALLEL:
          (a) PENDING LOOKUP (independent; 4 resolution sources, D119)
          (b) SEARCH: Qdrant compiled filter (§5.5) ∥ Neo4j BM25 (§12.1)
        RRF fuse → context fetch (Qdrant payload primary; Neo4j fallback)
        DB-BOUNDED expansion (§12.2) → dedupe → per-context applicability + LegalTimeBasis
        rerank (GPU) → heuristic (post-eligibility) → final-select ≤8 AND token budget
        SANCTION-SAFETY GATE (D120): pending match + penalty intent
              → suppress amounts/conclusion; force warning/review-pending answer
        WRITE RETRIEVAL CACHE (on retrieval+rerank success, even if generator later fails)
        gen → EMIT TOKENS IMMEDIATELY → tee bounded accumulator
        on terminal done SUCCESS → WRITE ANSWER CACHE   (D143)
        SSE: meta → sources → token* → done|error
```

### 3.4 Ingestion Path
```
trigger → arq enqueue
  ▼ RESERVE: release_id = INCR descriptor:release_ctr    (D106, atomic)
             build_id   = uuid4()                        (D107)
  ▼ scrape (allow-list §6; delta §6.4; TLS verify ON; throttle; future-effective)
  ▼ parse (src/legal_rag/ingestion owner); fail → FULL carry-forward (D130)
  ▼ parse-quality gate → partial → keep last-good, no tombstone
  ▼ effectivity_extract → EffectivityLedgerEntry[] (§5.7) + interval split (§5.6)
  ▼ amends_extract → typed ops (§10.4); auto/quarantined → PendingAmendmentRecord[]
  ▼ embed (reuse iff embedding_input_fingerprint matches)
  ▼ STAGING WRITE (build-scoped, never shared):
        Qdrant collection  = staging_{build_id}
        Neo4j nodes        = {build_id, pending_release_id: release_id}
  ▼ WRITE IMMUTABLE ARTIFACTS + HASHES (D108, before reconcile):
        release_manifest, coverage_manifest, pending_index,
        reviewed_overrides, review_log_snapshot, metadata_revision_set,
        allowlist_bootstrap reference
  ▼ RECONCILE (§8.4) against staging namespace only
  ▼ PROMOTE (only after reconcile passes):
        Qdrant: staging_{build_id} → legal_v{release_id}
        Neo4j : set release_id, drop build_id staging marker
  ▼ PUBLISH CAS (§8.3): fence check + fingerprint check + ownership-token bootstrap
  ▼ GC with reference guard (D134)
  ▼ report metrics.  NO artifact writes after publish (D108).
```

### 3.5 INVAR-PROJ (Full Projection)
Each release is a complete serving projection: every in-scope document, all provision versions inside the temporal support window, plus carried-forward last-good documents. Delta detection only avoids recomputation.

### 3.6 INVAR-PTR (Single Pointer)
`descriptor:active` (Redis Hash) is the authority. `active_release.json` on the `manifest_store` volume is a durable cache written only after a successful CAS. Rollback allocates a new `publish_fence` and is audited.

---

## 4. Identity Model

| Identifier | Definition | Stability |
|---|---|---|
| `source_document_id` | `docGUId` from source API | immutable per artifact |
| `raw_blob_hash` | SHA256(raw HTML bytes) | per blob |
| `source_metadata_revision_hash` | SHA256(canonical metadata JSON) | per metadata revision |
| `normalized_text_hash` | SHA256(normalized UTF-8 text) | per normalized content |
| `provision_text_hash` | SHA256(provision UTF-8 text) | per provision content |
| `legal_document_key` | §4.1 collision map | stable per legal act |
| `provision_uid` | `{legal_document_key}::article::{n}[::clause::{n}[::point::{l}]]` | stable lineage |
| `provision_version_id` | `{provision_uid}::rev::{provision_text_hash}` (FULL 64-hex) | per content version |
| `snapshot_node_id` | `UUIDv5(RELEASE_NS, "{release_id}:{provision_version_id}")` | per (release, version) |
| `metadata_node_key` | `SHA256(canonical({source_document_id, source_metadata_revision_hash}))` (D131) | per (doc, metadata revision) |
| `external_document_key` | D132 | per external stub revision |
| `source_revision_identity_hash` | SHA256(canonical(6-tuple in §4.2)) | per transform revision |
| `build_id` | UUIDv4 per build attempt | per build |
| `release_id` | `INCR descriptor:release_ctr` | monotonic |
| `publish_fence` | `INCR descriptor:fence_ctr` | monotonic, independent of release_id |

### 4.1 legal_document_key Collision Map (D87)
```python
class CollisionMapEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    doc_identity_normalized: str
    source_document_id: str
    legal_document_key: str          # doc_identity, or f"{doc_identity}#{source_document_id[:8]}"
    reason: Literal["unique", "collision_suffixed"]
    decided_at: datetime
    decided_by: Literal["auto", "reviewer"]
```
Rules: unique docIdentity → key = normalized docIdentity. Collision across distinct `source_document_id`s → each gets a suffixed key; the entry is appended immutably to `collision_map` (versioned artifact, hash in release manifest). Resolver: a citation resolving to >1 key → **reject as ambiguous** (quarantine the citation). Ingestion never drops a document because of collision.

### 4.2 Source Revision Identity
```
source_revision_identity = (source_document_id, raw_blob_hash,
    source_metadata_revision_hash, normalized_text_hash,
    normalizer_version, parser_version)
source_revision_identity_hash = SHA256(canonical_json(source_revision_identity))
```
Manifest path: `raw/manifests/{source_revision_identity_hash}/manifest.json` (immutable).

### 4.3 provision_version_id
Full 64-hex SHA256, never truncated (D88). Collision-registry unit test asserts distinct fixture texts → distinct ids.

### 4.4 Metadata + ExternalDocument Immutability (D89/D131/D132)
- Metadata nodes keyed by `metadata_node_key`, immutable once written.
- A release manifest lists the exact `metadata_revision_set` it references.
- Release N updating a document's metadata creates a NEW metadata node; release N-1 continues to reference its own revision. Title, URL, effectivity, status, and citation text for N-1 are therefore unchanged by N.
- ExternalDocument stub:
```python
class ExternalDocumentStub(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    external_document_key: str
    identity_kind: Literal["source_document_id", "normalized_citation_text"]
    source_document_id: Optional[str]
    normalized_citation_text: Optional[str]
    display_name: str
    external_revision: int            # monotonic per observed change
    first_seen_release_id: int
```
Stubs are immutable per `(external_document_key, external_revision)`; not retrievable, not validity-bearing.

---

## 5. Temporal Model

### 5.1 Types
```python
class GoverningDateKind(str, Enum):
    QUERY_TIME = "query_time"
    CONDUCT_TIME = "conduct_time"
    DETECTION_TIME = "detection_time"

class ValidityState(str, Enum):
    IN_FORCE = "in_force"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"

class RuleKind(str, Enum):
    STATIC_GENERAL = "static_general"
    STATIC_EXCEPTION = "static_exception"
    COMPLETED_CONDUCT_TIME = "completed_conduct_time"
    ONGOING_DETECTION_TIME = "ongoing_detection_time"
    REVIEW_PENDING = "review_pending"

class GoverningDates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_time: date                       # always present (as_of_date, default today)
    conduct_time: Optional[date] = None
    detection_time: Optional[date] = None

class RegimeMap(BaseModel):                # D112
    model_config = ConfigDict(extra="forbid", frozen=True)
    query_time: tuple[str, ...] = ()
    conduct_time: tuple[str, ...] = ()
    detection_time: tuple[str, ...] = ()

class ValidityInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    from_: date = Field(alias="from")       # inclusive
    to: Optional[date] = None               # exclusive; None = open-ended
    state: ValidityState
    governing_date_kind: GoverningDateKind
    effective_rule_id: str
    basis: str
    regime_id: str
    precedence: int
```

### 5.2 Factual Inputs → GoverningDates
Canonical inputs: `as_of_date`, `occurred_at`, `ended_at`, `detected_date`, `violation_state`. `event_date` is a deprecated alias normalized to `occurred_at` (both present and different → 422).

```
query_time     = as_of_date (default: today in Asia/Ho_Chi_Minh)
conduct_time   = ended_at if ended_at else occurred_at      # violation_state=completed
detection_time = detected_date                              # violation_state=ongoing (REQUIRED, no fallback)
```
Missing required facts for a sanction/transitional intent → `clarify` before retrieval.

### 5.3 Per-Kind Regime Resolution (D112/D113)
```python
def build_regime_map(gd: GoverningDates, rules: Sequence[TransitionRule]) -> RegimeMap:
    def regimes_at(d: Optional[date]) -> tuple[str, ...]:
        if d is None:
            return ()
        return tuple(sorted({r.regime_id for r in rules
                             if r.review_status == "reviewed" and r.covers(d)}))
    return RegimeMap(
        query_time=regimes_at(gd.query_time),
        conduct_time=regimes_at(gd.conduct_time),
        detection_time=regimes_at(gd.detection_time),
    )
```
Each compiled filter branch uses the regime set **of its own kind**. A kind with an empty regime set contributes **no branch**. If every branch is empty → outcome `temporal_rule_unsupported` with clarification (never a silent empty result).

Worked example: `occurred_at = 2023-06-01`, `as_of_date = 2026-08-15` →
`regime_map.conduct_time = ("traffic_sanctions_v1",)` (NĐ 100 era), `regime_map.query_time = ("traffic_sanctions_v2",)` (NĐ 168 era). The conduct branch cannot pull NĐ 168 provisions, and the query branch cannot pull NĐ 100 provisions.

### 5.4 Intent → Governing Kind Binding (D114)
| Intent class | Bound kind(s) | Note |
|---|---|---|
| penalty/sanction, completed conduct | `conduct_time` only | law at time of conduct |
| penalty/sanction, ongoing conduct | `detection_time` only | law at detection |
| "current rule / what does the law say now" | `query_time` only | current law |
| transitional question | `conduct_time` + `detection_time` as applicable | per transitional rule |
| explicit comparison ("before vs after") | multiple kinds, **served as separate labeled branches**, each with its own retrieval + citations | never blended into one unlabeled answer |

A single answer never blends kinds implicitly. Comparison requests produce labeled sections.

### 5.5 Static Filter Compiler (D78/D136)
```python
def compile_eligibility(gd: GoverningDates, rm: RegimeMap) -> models.Filter:
    should: list[models.Filter] = []
    for kind, d, regimes in (
        (GoverningDateKind.QUERY_TIME,     gd.query_time,     rm.query_time),
        (GoverningDateKind.CONDUCT_TIME,   gd.conduct_time,   rm.conduct_time),
        (GoverningDateKind.DETECTION_TIME, gd.detection_time, rm.detection_time),
    ):
        if d is None or not regimes:
            continue
        di = to_epoch_days(d)
        should.append(models.Filter(must=[models.NestedCondition(nested=models.Nested(
            key="validity_intervals",
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="governing_date_kind",
                                          match=models.MatchValue(value=kind.value)),
                    models.FieldCondition(key="state",
                                          match=models.MatchValue(value="in_force")),
                    models.FieldCondition(key="regime_id",
                                          match=models.MatchAny(any=list(regimes))),
                    models.FieldCondition(key="from_days",
                                          range=models.Range(lte=di)),
                ],
                should=[
                    models.IsNullCondition(is_null=models.PayloadField(key="to_days")),
                    models.FieldCondition(key="to_days", range=models.Range(gt=di)),
                ],
                min_should=models.MinShould(conditions=[], min_count=1),
            ),
        ))]))
    if not should:
        raise TemporalRuleUnsupported()      # D113
    return models.Filter(should=should, min_should=models.MinShould(conditions=[], min_count=1))
```
Notes: dates are stored redundantly as integer `from_days` / `to_days` (epoch days) inside each interval object so numeric `Range` works; `to_days` absent ⇒ open-ended handled by the `IsNull` branch. All conditions sit inside one `Nested` filter so they bind to the **same** interval element. `min_should` is used rather than bare `should` so at least one branch must match. A fixture test runs this exact filter against a real Qdrant container (D136).

Neo4j equivalent (same semantics, one list comprehension per branch):
```cypher
WHERE any(iv IN n.validity_intervals WHERE
        iv.governing_date_kind = $kind
    AND iv.state = 'in_force'
    AND iv.regime_id IN $regimes
    AND iv.from_days <= $d
    AND (iv.to_days IS NULL OR iv.to_days > $d))
```

### 5.6 Deterministic Interval Split (D116)
Given one `static_general` entry and N `static_exception` entries scoped to a provision:
```python
def split_intervals(general: Entry, exceptions: Sequence[Entry]) -> list[ValidityInterval]:
    # 1. boundary set
    bounds = {general.effective_from}
    if general.effective_to: bounds.add(general.effective_to)
    for e in exceptions:
        bounds.add(e.effective_from)
        if e.effective_to: bounds.add(e.effective_to)
    ordered = sorted(bounds)                       # deterministic
    # 2. emit half-open segments
    out = []
    for i, start in enumerate(ordered):
        end = ordered[i + 1] if i + 1 < len(ordered) else None
        # 3. most-specific rule wins, total order:
        #    (scope_specificity DESC, precedence DESC, entry_id ASC)
        winner = max(
            (e for e in [general, *exceptions] if e.covers_segment(start, end)),
            key=lambda e: (e.scope_specificity, e.precedence, invert(e.entry_id)),
            default=None,
        )
        if winner is None:
            continue
        out.append(interval_from(winner, start, end))
    return merge_adjacent_identical(out)           # deterministic, no overlap
```
`scope_specificity`: point(4) > clause(3) > article(2) > document(1). Output is non-overlapping by construction. Consequence for NĐ 168: provisions named in the 2026 exception clause do **not** carry a `[2025-01-01, ∞)` segment; their general segment is bounded at `2026-01-01`, and the exception owns `[2026-01-01, ∞)`.

### 5.7 Effectivity Schemas (D117)
```python
class EffectivityLedgerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    entry_id: str
    scope_locator: str                     # provision_uid or legal_document_key
    scope_specificity: int                 # 1..4 (document..point)
    rule_kind: RuleKind
    effective_from: Optional[date]
    effective_to: Optional[date]           # exclusive after D115 conversion
    precedence: int
    regime_id: str
    governing_date_kind: GoverningDateKind
    evidence_locator: "EvidenceLocator"
    validity_basis: str
    review_status: Literal["auto", "reviewed", "quarantined"]
    source: Literal["metadata_effect_date", "metadata_expire_date", "document_text"]
```

Date semantics (D115), applied at extraction:
| Source field | Meaning | Conversion |
|---|---|---|
| `effectDate` | inclusive first day in force | `interval.from = effectDate` |
| `expireDate` | inclusive last day in force | `interval.to = expireDate + 1 day` |
| `effectDate = null` | unknown | `state = unknown`, not eligible |
| `expireDate = null` | open-ended | `to = None` |
| `expireDate < effectDate` | invalid | quarantine with evidence |
| metadata vs document text disagree | conflict | quarantine with evidence; no silent pick |

All dates are date-only, interpreted in `Asia/Ho_Chi_Minh`, stored as ISO date plus `from_days`/`to_days` epoch-day integers.

Precedence (highest → lowest): reviewed AMENDS-materialized intervals → reviewed exception/transition ledger entries → reviewed static general → metadata `effectDate` baseline. `expireDate` is an outer cap applied to every interval of the document.

### 5.8 Coverage Schemas (D117/D128)
```python
class CoverageGapKind(str, Enum):
    TEMPORAL = "temporal"
    CORPUS_DEPENDENCY = "corpus_dependency"
    PARSE_QUALITY = "parse_quality"
    UNRESOLVED_AMENDMENT = "unresolved_amendment"

class CoverageGap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: CoverageGapKind
    from_: Optional[date] = Field(default=None, alias="from")
    to: Optional[date] = None
    regime_id: Optional[str] = None
    legal_document_key: Optional[str] = None
    governing_date_kind: Optional[GoverningDateKind] = None
    reason: str
    evidence_ref: Optional[str] = None

class RegimeCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    regime_id: str
    verified_from: Optional[date]
    verified_to: Optional[date]
    document_count: int
    gaps: tuple[CoverageGap, ...] = ()

class DocumentCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    legal_document_key: str
    source_document_id: str
    metadata_revision_hash: str
    parsed_ok: bool
    carried_forward: bool
    provision_count: int
    unresolved_amendment_count: int
    gaps: tuple[CoverageGap, ...] = ()

class CoverageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    manifest_version: Literal[1] = 1
    per_regime: tuple[RegimeCoverage, ...]
    per_document: tuple[DocumentCoverage, ...]
    latest_source_watermark: str            # UTC ISO-8601 (D129)
    watermark_overlap_days: int = 7
    built_at: datetime
```
Coverage checks are performed per `(governing_date_kind, regime_id, legal_document_key)` — never a single global window. Date in a `TEMPORAL` gap or outside a regime's verified range → `coverage_unsupported`. Unproven dependency closure for the answer scope → `corpus_coverage_incomplete` (D127) with the gap injected into the prompt and surfaced in warnings/citations.

### 5.9 Conditional Materialization
`conditional`/`unknown` intervals become eligible only when a reviewer materializes an `in_force` interval through the `reviewed_overrides` artifact (§10.6). The predicate always matches `state = in_force`; there is no separate "conditional but allowed" path.

---

## 6. Corpus Scope, Allow-List, Delta

### 6.1 Allow-List Bootstrap Artifact (D126)
The allow-list is a **required bootstrap artifact**, not an inline placeholder. `allowlist_bootstrap.json` is produced by a one-time facet-discovery run against the source API, content-addressed, and referenced by hash in every release descriptor.

```python
class AllowlistBootstrap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    allowlist_version: int
    source_host: Literal["phapluat.gov.vn"]
    doc_group_ids: tuple[int, ...]         # discovered, non-empty
    field_ids: tuple[int, ...]             # discovered, non-empty
    search_queries: tuple[str, ...]        # literal API keywords, no wildcards
    doc_name_pattern: str                  # local regex, applied post-download
    exclusion_rules: tuple["ExclusionRule", ...]
    discovered_at: datetime
    artifact_hash: str                     # SHA256(canonical of all above except this)

class ExclusionRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pattern: str
    justification: str
    fixture_path: str                      # test fixture proving non-normative
```
Committed literal values (safe to fix now, verified as source-API concepts):
```
search_queries      = ("giao thông đường bộ",
                       "trật tự an toàn giao thông",
                       "xử phạt vi phạm hành chính giao thông")
doc_name_pattern    = "giao thông|trật tự.*giao thông|xử phạt.*giao thông|đường bộ"
```
`doc_group_ids` and `field_ids` are **discovered** values; they are not invented in this spec. Until `allowlist_bootstrap.json` exists with non-empty tuples and a recorded hash, `/readyz` returns `503 allowlist_not_populated` and serving refuses to start.

### 6.2 Effect Status Is Not a Scope Gate
Scope never filters on document `effect_status`. Verified: NĐ 168/2024/NĐ-CP carries `effectStatusName = "Hết Hiệu lực một phần"`; gating on status would wrongly exclude it.

### 6.3 Dependency Closure and Reverse Discovery (D127)
Forward closure: after ingesting a document, resolve its AMENDS targets; if a target is in scope but the amending document is absent, enqueue that amending document. Also traverse `docListOther` references as amending candidates.

Known limitation (surfaced, not hidden): a document that amends an in-scope provision but is itself out of scope and never referenced by any ingested document is **not** discovered. This produces a `CORPUS_DEPENDENCY` gap. When the answer's scope intersects such a gap the outcome is `corpus_coverage_incomplete`, the gap text is injected into the generator prompt, and a warning plus gap citation is returned.

### 6.4 Delta Scan Contract (D129)
```python
class DeltaCursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    allowlist_version: int
    latest_source_watermark: str      # max observed updDateTime, UTC ISO-8601
    overlap_days: int = 7
    updDateTime_supported: bool       # result of capability probe
    last_full_reconciliation_at: datetime
    full_reconciliation_interval_days: int = 7
```
Rules: candidate set = allow-list query results whose `updDateTime` ≥ (`latest_source_watermark` − `overlap_days`) when the capability probe succeeds; otherwise the full allow-list scope is scanned. Change decision is always by **content hash** comparison, never by `issueDate`/`effectDate`. Cursor is persisted in the release manifest. Retries are bounded with a deadline. A full scoped reconciliation runs at least every `full_reconciliation_interval_days`.

### 6.5 Parse-Fail Full Carry-Forward (D130)
On parse failure or a failed parse-quality gate for a document, the new release carries forward from the last-good release: the document node, every provision version, the referenced metadata revision, all pending amendment records for that document, and its `DocumentCoverage` entry with `carried_forward = true` and a `PARSE_QUALITY` gap. Nothing is tombstoned. The document never disappears from the projection.

---

## 7. Raw Provenance

### 7.1 Layout
```
raw/
  blobs/{raw_blob_hash}/original.html
  metadata/{source_metadata_revision_hash}/metadata.json
  normalized/{normalized_text_hash}/{normalizer_version}/normalized.txt
  manifests/{source_revision_identity_hash}/manifest.json
```
Blobs deduplicate by hash; manifests are immutable per transform revision.

### 7.2 Raw Manifest Schema (D133)
```python
class ParseQualityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    raw_char_length: int
    normalized_char_length: int
    article_anchor_count: int
    clause_count: int
    point_count: int
    hierarchy_coverage_ratio: float        # parsed chars / normalized chars
    passed_gate: bool
    gate_failure_reason: Optional[str] = None

class RawManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    manifest_version: Literal[1] = 1
    source_revision_identity_hash: str
    source_document_id: str
    official_url: str
    fetched_at: datetime                   # UTC
    http_status: int
    raw_blob_hash: str
    raw_content_length: int
    source_metadata_revision_hash: str
    normalized_text_hash: str
    normalizer_version: str
    parser_version: str
    offset_unit: Literal["unicode_code_points"]
    evidence_offsets: tuple["EvidenceOffsetRange", ...]
    parse_quality: ParseQualityMetrics
    provenance_chain: tuple[str, ...]      # prior source_revision_identity_hashes
    tls_verified: bool                     # must be true in serving-path ingests

class EvidenceOffsetRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str                             # e.g. "hieu_luc_thi_hanh", "chuyen_tiep"
    start_offset: int
    end_offset: int
    excerpt_sha256: str
```

### 7.3 GC Reference Guard (D134)
An artifact under `raw/` may be deleted only when no retained release manifest, `review_log`, `reviewed_overrides` entry, eval label, or emitted citation-registry row references its `source_revision_identity_hash`, `normalized_text_hash`, or `raw_blob_hash`. The guard is evaluated as a set difference before any deletion; failure to prove non-reference blocks deletion.

### 7.4 Serving-Corpus Ownership
`src/legal_rag/ingestion` owns the serving parser and its guarantees (alphanumeric clause numbers such as `2a`/`18a`, appendix preserve-by-default, effectivity extraction). The reference submodule's parser (`_RE_CLAUSE = r"^(\d+)\.\s+(.*)"`, verified) does not satisfy them and is never used for serving.

---

## 8. Release Lifecycle

### 8.1 Release Descriptor
```python
class ReleaseDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    descriptor_version: Literal[1] = 1
    release_id: int                        # from descriptor:release_ctr
    publish_fence: int                     # from descriptor:fence_ctr
    build_id: str                          # build that produced this release
    qdrant_collection: str                 # legal_v{release_id}
    neo4j_release_id: int                  # == release_id
    embedding_version: str
    schema_version: str
    release_manifest_hash: str
    coverage_manifest_hash: str
    pending_index_hash: str                # D109
    reviewed_overrides_hash: str
    review_log_snapshot_hash: str
    metadata_revision_set_hash: str
    allowlist_artifact_hash: str
    collision_map_hash: str
    temporal_support_window_from: date
    published_at: datetime
    fingerprint: str                       # SHA256(canonical of all fields except fingerprint)
```

### 8.2 Reservation and Staging (D106/D107)
1. `release_id = INCR descriptor:release_ctr` — atomic, so two workers can never select the same id.
2. `build_id = uuid4()`.
3. All staging writes are build-scoped: Qdrant collection `staging_{build_id}`; Neo4j nodes carry `build_id` and `pending_release_id`. Two concurrent builds therefore write disjoint namespaces.
4. Immutable artifacts are written and hashed (§8.5) **before** reconcile.
5. Reconcile runs against the staging namespace only.
6. Promote only after reconcile passes: Qdrant `staging_{build_id}` → `legal_v{release_id}`; Neo4j sets `release_id`, clears the staging marker.
7. Publish CAS.

### 8.3 Publish CAS with Ownership Token (D111)
```lua
-- KEYS[1]=descriptor:active (Hash)  KEYS[2]=descriptor:init_lock
-- ARGV[1]=base_fence ARGV[2]=base_fingerprint ARGV[3]=new_fence
-- ARGV[4]=owner_token ARGV[5..]=field,value pairs
local cur_fence = redis.call('HGET', KEYS[1], 'publish_fence')
local cur_fp    = redis.call('HGET', KEYS[1], 'fingerprint')

if cur_fence == false then                       -- bootstrap path
  local lock_owner = redis.call('GET', KEYS[2])
  if lock_owner ~= ARGV[4] then
    return -1                                     -- not our lock: refuse
  end
  for i = 5, #ARGV, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1]) end
  return 1
end

if cur_fp == ARGV[2]
   and tonumber(cur_fence) == tonumber(ARGV[1])
   and tonumber(ARGV[3]) > tonumber(cur_fence) then
  for i = 5, #ARGV, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1]) end
  return 1
end
return 0                                          -- stale
```
Bootstrap acquires the lock with `SET descriptor:init_lock <owner_token> NX EX 300`; the script verifies the token matches, so a worker cannot publish under another worker's lock. Return `1` = published, `0` = stale (discard, rebase on the current descriptor, rebuild), `-1` = bootstrap contention (retry).

After a successful CAS: write `active_release.json` atomically (temp file + rename) and refresh the API's in-memory copy. Rollback allocates a **new** `publish_fence` and CASes to a descriptor that points at an older `release_id`; because the fence always advances, a stale build cannot re-publish after a rollback even if its base fingerprint reappears. Rollback writes an audit row (actor, timestamp, from/to fence, from/to release_id).

### 8.4 Reconcile
Publish requires every check to pass against the staging namespace:
- Node tuple set equality: `(snapshot_node_id, pending_release_id, provision_version_id)`.
- Edge digest equality: `SHA256(sorted canonical edge tuples)` including hierarchy, metadata, AMENDS, and `RELATED_TO`→ExternalDocument.
- Endpoint existence for every edge.
- Per-point `derived_state_hash` equality.
- Fulltext index readiness plus a smoke query.
- Qdrant point-set equality and collection-config fingerprint equality.
- `metadata_revision_set` referenced by the release exists and is immutable.
- Artifact hashes recorded in the descriptor match the artifacts on disk.

Any mismatch aborts publish, leaves the active pointer untouched, and alerts.

### 8.5 Immutable Artifacts (D108/D109)
Written content-addressed and hashed before reconcile: `release_manifest`, `coverage_manifest`, `pending_index`, `reviewed_overrides`, `review_log_snapshot`, `metadata_revision_set`, `collision_map`, plus a reference to `allowlist_bootstrap`. No artifact that a published release references is ever written or mutated after publish. `descriptor:pending` in Redis is a read-through cache of the release's `pending_index`; if the cache is missing or inconsistent it is rebuilt from the immutable artifact, and it can never alter a published release's behavior.

### 8.6 Backup and Restore Ordering (D110)
Backup covers: raw provenance tree, metadata revisions, `reviewed_overrides`, `review_log`, `coverage_manifest`, `pending_index`, `collision_map`, Neo4j dump, Qdrant per-collection snapshots, release manifests, `active_release.json`, local trace registry. Cache and arq queues are ephemeral and are never restored as serving state.

Restore order is mandatory: raw provenance → metadata revisions → reviewed_overrides → review_log → coverage_manifest → pending_index → Neo4j → Qdrant → release manifests → **activate pointer last**. `/readyz` stays `503` until the pointer is activated and all reconciliation checks pass.

### 8.7 Garbage Collection
Old releases are removed only after a grace period longer than the maximum request/SSE deadline, and never inside the active release's temporal support window. Metadata and ExternalDocument nodes are removed only when no retained release references their revision. Raw artifacts obey the reference guard (§7.3).

---

## 9. Ingestion Pipeline

Stages: reserve (release_id + build_id) → scrape → parse → parse-quality gate → effectivity_extract → amends_extract → embed → staging write → immutable artifacts + hashes → reconcile → promote → publish CAS → GC → report.

### 9.1 Failure Behavior
| Failure | Behavior |
|---|---|
| HTTP 200 with non-null `error` | upstream error; bounded retry; **not** end-of-catalog |
| `docs[]` empty and `error == null` | end of catalog; stop pagination |
| Parse failure for one document | FULL carry-forward (D130); `PARSE_QUALITY` gap; continue batch |
| Parse-quality gate failure | keep last-good; no tombstone |
| `expireDate < effectDate` | quarantine with evidence |
| Metadata vs text effectivity conflict | quarantine with evidence; no silent pick |
| AMENDS schema/validation failure | quarantine edge; document ingest continues |
| AMENDS auto/quarantined with any planned date (incl. `None`) | write `PendingAmendmentRecord` (D118) |
| Embedding or store write failure | bounded retry with deadline; then mark failed and alert |
| Reconcile mismatch | abort publish; active pointer untouched; alert |
| CAS returns 0 (stale) | discard build; rebase; rebuild |
| CAS returns -1 (bootstrap contention) | wait; retry with fresh lock attempt |
| Crash between CAS and local-file write | restart reloads from Redis authority (authoritative) |
| `INSECURE_TLS=true` in production profile | startup FAILS |
| Empty allow-list artifact | `/readyz` → `503 allowlist_not_populated` |
| docIdentity collision | collision map entry; resolver rejects ambiguity; ingest continues |

All retries are bounded with an explicit deadline; no unbounded retry loops (D159).

---

## 10. AMENDS and Pending Amendments

### 10.1 Pending State Machine (D122)
```
raw_candidate                 (extraction output, not persisted as edge state)
      │
      ├─► auto            (persisted: locator resolved, schema valid, unreviewed)
      └─► quarantined     (persisted: unresolved/invalid/non-deterministic)
                │
                ▼
        review_pending    (SERVE-TIME derivation only — never a persisted edge state)
                │
        ┌───────┴────────┐
        ▼                ▼
     reviewed         rejected      (persisted decisions via reviewed_overrides)
```
Only `reviewed` decisions materialize eligibility. A raw pending candidate does **not** require a reviewed `effective_rule_id`.

### 10.2 PendingAmendmentRecord (D121)
```python
class UnresolvedReason(str, Enum):
    NONE = "none"
    TARGET_UNRESOLVED = "target_unresolved"
    EFFECTIVE_DATE_UNKNOWN = "effective_date_unknown"
    AMBIGUOUS_CITATION = "ambiguous_citation"
    NON_DETERMINISTIC_PATCH = "non_deterministic_patch"
    SCHEMA_INVALID = "schema_invalid"

class PendingAmendmentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str
    release_fingerprint: str                       # release this record belongs to
    target_locator: "LegalLocator"                 # canonical target citation
    target_provision_uid: Optional[str]            # None when unresolved
    target_legal_document_key: Optional[str]
    amending_source_document_id: str               # source amendment
    amending_locator: "LegalLocator"
    operation_hint: Optional[str]                  # ADD_PROVISION | REPLACE_TEXT | REPEAL | AMEND
    planned_effective_date: Optional[date]         # None = unknown
    governing_date_kind: GoverningDateKind
    unresolved_reason: UnresolvedReason
    edge_review_status: Literal["auto", "quarantined"]
    evidence_locator: "EvidenceLocator"
    target_citation_text: str                      # for lexical matching (D119)
    amending_excerpt: str                          # for lexical matching (D119)
```
The `pending_index` artifact is the immutable tuple of these records for a release, hashed into the descriptor (D109).

### 10.3 Independent Pending Lookup (D118/D119/D120)
Runs independently of, and in parallel with, ANN/BM25 retrieval. It is never gated on whether a target reached top-k.

Candidate target resolution uses the union of four sources:
1. Explicit `legal_locators` supplied in the request.
2. Locator regex extraction from the query text (`Điều\s+\d+[a-z]?`, `khoản\s+\d+[a-z]?`, `điểm\s+[a-zđ]`, document-number patterns).
3. A lexical/keyword index over `target_citation_text` + `amending_excerpt` of every pending record in the release.
4. `provision_uid`s of retrieval candidates (when retrieval completes).

Match rule (D118) — unknown dates are **never skipped**:
```python
def pending_matches(rec: PendingAmendmentRecord, gd: GoverningDates) -> bool:
    if rec.planned_effective_date is None:
        return True                                  # unknown ⇒ potentially relevant
    d = getattr(gd, rec.governing_date_kind.value)
    return d is not None and rec.planned_effective_date <= d
```
Outcome mapping:
| Situation | Outcome |
|---|---|
| Match with known date, answer could change | `legal_status_review_pending` + warning + citations |
| Match with `planned_effective_date is None` and the answer depends on it | `temporal_rule_unsupported` + clarification |
| Match with unknown date but answer provably independent | `legal_status_review_pending` + warning |
| Match with `unresolved_reason = TARGET_UNRESOLVED` intersecting the query scope | `legal_status_review_pending` + warning |

Sanction-safety gate (D120): when the intent is penalty/sanction and any matched pending record could change the outcome, the generator is instructed and post-checked to emit **no** monetary amounts, point deductions, or confident conclusions. The response contains the current provision's citation, the pending amendment's evidence citation, and a clarification/warning. A post-generation regex guard (currency/points patterns) blocks a violating answer and downgrades it to the warning form.

### 10.4 AMENDS Operation Schemas (D123)
```python
class LegalLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    citation_text: str                     # verbatim human citation
    legal_document_key: Optional[str]      # resolved; None when unresolved
    doc_identity_text: str                 # as written in source
    article: Optional[str] = None          # r"\d+[a-z]?"
    clause: Optional[str] = None           # r"\d+[a-z]?"
    point: Optional[str] = None            # single Vietnamese point letter
    resolution_status: Literal["resolved", "ambiguous", "unresolved"]

class EvidenceLocator(BaseModel):          # D124
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_revision_identity_hash: str     # manifest id (§4.2)
    source_document_id: str
    normalized_text_hash: str
    normalizer_version: str
    parser_version: str
    offset_unit: Literal["unicode_code_points"]
    start_offset: int
    end_offset: int
    excerpt_encoding: Literal["utf-8"]
    excerpt_sha256: str
    # Validation before review/apply: load the normalized artifact identified by
    # (normalized_text_hash, normalizer_version); assert 0 <= start < end <= len(text);
    # assert sha256(text[start:end].encode("utf-8")) == excerpt_sha256.

class PatchSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["replace_phrase", "delete_phrase"]
    target_phrase: str
    replacement_phrase: Optional[str] = None       # required iff kind == replace_phrase
    occurrence_rule: Literal["first", "all"] | str # "nth:{k}" also allowed
    expected_successor_sha256: Optional[str] = None
    expected_successor_text: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.kind == "replace_phrase" and self.replacement_phrase is None:
            raise ValueError("replacement_phrase required for replace_phrase")
        if self.kind == "delete_phrase" and self.replacement_phrase is not None:
            raise ValueError("replacement_phrase forbidden for delete_phrase")
        if not (self.expected_successor_sha256 or self.expected_successor_text):
            raise ValueError("expected_successor_sha256 or expected_successor_text required")
        return self

class AddProvisionOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["ADD_PROVISION"]
    amending_locator: LegalLocator
    new_provision_locator: LegalLocator
    parent_provision_uid: str
    order_index: int
    successor_text: Optional[str] = None
    successor_source_span: Optional[EvidenceLocator] = None
    successor_provision_version_id: Optional[str] = None   # set at materialization
    evidence_locator: EvidenceLocator
    effective_rule_id: str

    @model_validator(mode="after")
    def _check(self):
        if not (self.successor_text or self.successor_source_span):
            raise ValueError("successor_text or successor_source_span required")
        return self

class ReplaceTextOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["REPLACE_TEXT"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    target_provision_uid: str
    base_provision_version_id: Optional[str] = None
    base_text_sha256: Optional[str] = None
    successor_text: Optional[str] = None
    exact_patch: Optional[PatchSpec] = None
    successor_provision_version_id: Optional[str] = None   # set at materialization
    evidence_locator: EvidenceLocator
    effective_rule_id: str

    @model_validator(mode="after")
    def _check(self):
        if not (self.base_provision_version_id or self.base_text_sha256):
            raise ValueError("base_provision_version_id or base_text_sha256 required")
        if not (self.successor_text or self.exact_patch):
            raise ValueError("successor_text or exact_patch required")
        return self

class RepealOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["REPEAL"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    target_provision_uid: str
    cascade_to_children: bool = True
    evidence_locator: EvidenceLocator
    effective_rule_id: str

class AmendOp(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: Literal["AMEND"]            # generic; never materializes until refined
    amending_locator: LegalLocator
    target_locator: LegalLocator
    target_provision_uid: Optional[str] = None
    evidence_locator: EvidenceLocator
    effective_rule_id: str

AmendsOp = Annotated[
    Union[AddProvisionOp, ReplaceTextOp, RepealOp, AmendOp],
    Field(discriminator="operation"),
]
```
Materialization requires: (a) `effective_rule_id` exists in the release's ledger, has `review_status = reviewed`, and its `scope_locator` covers the target; (b) for `ReplaceTextOp`/`AddProvisionOp`, a concrete successor text — either verbatim or a patch that applies deterministically to the identified base and whose result matches `expected_successor_sha256`/`expected_successor_text`; (c) a `successor_provision_version_id` is created **before** the predecessor interval is closed. Failure of any condition → quarantine plus a `PendingAmendmentRecord`.

### 10.5 Edge Identity
`edge_id = SHA256(canonical({operation, amending_snapshot_node_id, target_snapshot_node_id, effective_rule_id, evidence_locator.excerpt_sha256}))`. Two distinct amendments therefore never MERGE into one edge.

### 10.6 reviewed_overrides Artifact (D125)
```python
class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REFINE = "refine"

class ReviewedOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    promotion_id: str
    source_review_log_entry_id: str
    reviewer: str
    decided_at: datetime
    decision: ReviewDecision
    target_scope: str                      # provision_uid or legal_document_key
    target_scope_specificity: int
    effective_interval: ValidityInterval    # concrete materialized interval
    regime_id: str
    evidence_locator: EvidenceLocator
    rationale: str

class ReviewedOverridesArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    artifact_version: Literal[1] = 1
    overrides: tuple[ReviewedOverride, ...]
    artifact_hash: str                     # SHA256(canonical of overrides)
```
Promotion flow: a `review_log` entry records the human decision (annotation only) → the promotion step validates evidence offsets and scope → a `ReviewedOverride` row is emitted into the artifact → the release build materializes intervals from the artifact. A `review_log` entry alone never changes eligibility.

---

## 11. Embeddings and Indexing

### 11.1 Embedding Reuse
```
embedding_input_fingerprint = SHA256(canonical({
    model_revision, segmenter_version, tokenizer_version,
    preprocessing_config, dimension, provision_text }))
```
A vector is reused only on an exact fingerprint match; any pipeline change forces re-embedding. Payload, context text, and `derived_state_hash` are always recomputed even when the vector is reused.

Model: `bkai-foundation-models/vietnamese-bi-encoder` at a pinned revision, 768-dim, cosine-normalized, with mandatory `pyvi.ViTokenizer` segmentation applied identically to indexed text and queries. Documents are embedded in batch offline; a single query is embedded online at request time.

### 11.2 Qdrant Payload
```python
class QdrantPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provision_uid: str
    provision_version_id: str
    source_document_id: str
    legal_document_key: str
    doc_identity_text: str
    doc_type: str
    label: Literal["Article", "Clause", "Point"]
    number: Optional[str]
    parent_article: Optional[str]
    parent_clause: Optional[str]
    validity_intervals: tuple[dict, ...]   # each: from/to ISO + from_days/to_days + state
                                           # + governing_date_kind + regime_id
                                           # + effective_rule_id + basis + precedence
    effect_date: Optional[str]
    expire_date: Optional[str]
    field_names: tuple[str, ...]
    organ_names: tuple[str, ...]
    release_id: int
    embedding_version: str
    embedding_input_fingerprint: str
    derived_state_hash: str
    context_text: str
    # citation projection (D139)
    title: str
    official_url: str
    human_citation: str
    provision_path: tuple[str, ...]
    snippet: str
```
Payload indexes are created for `governing_date_kind`, `state`, `regime_id`, `from_days`, `to_days` (nested), plus `release_id`, `label`, `legal_document_key`, `field_names`, `organ_names`. No static applicability field is stored: applicability is computed per request.

---

## 12. Retrieval and Generation

### 12.1 Neo4j BM25 Query Plan with Real Bounds (D137)
```cypher
CALL db.index.fulltext.queryNodes($index_name, $text, {limit: $overfetch}) YIELD node, score
WHERE node.release_id = $release_id
WITH node, score
WHERE any(iv IN node.validity_intervals WHERE
        iv.governing_date_kind = $kind
    AND iv.state = 'in_force'
    AND iv.regime_id IN $regimes
    AND iv.from_days <= $d
    AND (iv.to_days IS NULL OR iv.to_days > $d))
RETURN node.provision_version_id AS uid, score
ORDER BY score DESC
LIMIT $k
```
Bounds, enforced by the driver and the query rather than by prose:
- `overfetch = MAX_FULLTEXT_SCAN = 500` is passed to `queryNodes` as a hard `limit`, so the fulltext scan itself is capped.
- Transaction timeout `BM25_TX_TIMEOUT_MS = 2000` is set on the session, so a slow scan fails fast.
- `release_id` is filtered on the row immediately after the yield, before the validity predicate and before the final `LIMIT $k`; a release N-1 row can therefore never occupy a slot in the top-k for release N.
- Budget exceeded (timeout or overfetch saturation) → BM25 contributes an empty result, `degraded_components += ["bm25"]`, and dense retrieval proceeds. Never a silent partial merge without the degraded flag.

One query is issued per compiled branch (kind, regimes, date); results are merged by RRF with the dense results.

### 12.2 DB-Level Expansion Bounds (D138)
```cypher
MATCH (n {release_id: $release_id})
WHERE n.provision_version_id IN $seed_uids
MATCH (n)-[:HAS_CLAUSE|HAS_POINT*1..2]->(child {release_id: $release_id})
WHERE any(iv IN child.validity_intervals WHERE /* same compiled predicate */ )
RETURN child.provision_version_id AS uid,
       child.label AS label,
       child.content AS content
LIMIT $expansion_row_cap
```
Depth is bounded in the pattern (`*1..2` = `MAX_GRAPH_EXPANSION_DEPTH`), rows by `LIMIT $expansion_row_cap` (`MAX_GRAPH_EXPANSION_ROWS = 50`), and time by an explicit transaction timeout (`EXPANSION_TX_TIMEOUT_MS = 1500`). Nothing large is fetched and then trimmed. After expansion, results are de-duplicated by `provision_version_id`, then the final selection enforces both caps: at most `CONTEXT_K = 8` contexts and at most `TOKEN_BUDGET = 4096` tokens.

### 12.3 Per-Context Applicability
After the database has already narrowed candidates using the compiled static filter (§5.5), the **server-side Python** selection step picks, for each returned context, the interval whose `[from, to)` covers the governing date belonging to that interval's own `governing_date_kind`, with `state = in_force` and `regime_id` inside that kind's regime set. This dictionary lookup happens in application code, never inside a Qdrant or Cypher predicate. It then emits:
```python
class LegalTimeBasis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    governing_date_kind: GoverningDateKind
    governing_date: date
    regime_id: str
    rule_kind: RuleKind
    effective_rule_id: str
    selected_interval: ValidityInterval
    applicability: Literal["applicable", "not_applicable", "review_pending"]
```
This object is attached to every `sources[]` item (D140) and passed per context to the generator prompt.

### 12.4 Text2Cypher Templates (complete, D141)
```python
class TemplateKind(str, Enum):
    COUNT_ARTICLES_BY_DOC = "count_articles_by_doc"
    LIST_SIGNERS_OF_DOC = "list_signers_of_doc"
    LIST_DOCS_BY_YEAR = "list_docs_by_year"
    COUNT_DOCS_BY_FIELD = "count_docs_by_field"
    GET_DOC_METADATA = "get_doc_metadata"
    LIST_DOCS_BY_ORGAN = "list_docs_by_organ"

class CountArticlesByDocRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["count_articles_by_doc"]
    legal_document_key: str

class ListSignersOfDocRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["list_signers_of_doc"]
    legal_document_key: str

class ListDocsByYearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["list_docs_by_year"]
    year: int = Field(ge=1945, le=2100)

class CountDocsByFieldRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["count_docs_by_field"]
    field_name: str

class GetDocMetadataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["get_doc_metadata"]
    legal_document_key: str

class ListDocsByOrganRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["list_docs_by_organ"]
    organ_name: str

TemplateCall = Annotated[
    Union[
        CountArticlesByDocRequest,
        ListSignersOfDocRequest,
        ListDocsByYearRequest,
        CountDocsByFieldRequest,
        GetDocMetadataRequest,
        ListDocsByOrganRequest,
    ],
    Field(discriminator="kind"),
]
```
Every template maps to a fixed server-owned parameterized Cypher string with a hard `LIMIT`, a transaction timeout, a declared result schema, the compiled validity predicate, `release_id` scoping, and an output that always includes `provision_version_id` plus a `CitationContext`. Document arguments are `legal_document_key`, resolved through the ambiguity-rejecting resolver — never raw `doc_identity`. An LLM selection that fails discriminated-union validation is routed to normal retrieval.

### 12.5 Generator Prompt Wrapper
A versioned wrapper receives `governing_dates`, the per-context `LegalTimeBasis` list, the citations, the eligible contexts, the `review_pending` flags, and any coverage gaps. It instructs the model to answer only from the supplied contexts, to state the governing date basis, and — when the sanction-safety gate is active — to withhold amounts and conclusions. The reference project's prompt is a terminology source only; it is not reused verbatim because it prioritizes "Ngày hiện tại" (verified at `prompts.py` lines 132, 147, 174), which contradicts conduct-time reasoning.

### 12.6 Execution Model
A bounded thread-pool executor with a bounded admission queue (`MAX_EXECUTOR_QUEUE = 100`) fronts all blocking driver calls; saturation returns `429` or `503` with `Retry-After`. GPU reranking is serialized by an in-process semaphore, valid because the API runs a single worker process. Every stage has a timeout and the request has an overall deadline. VRAM exhaustion degrades to CPU. The streaming accumulator is bounded.

---

## 13. API and SSE Contract

### 13.1 Limits (D97)
```
MAX_REQUEST_BODY_BYTES     = 262144
MAX_QUERY_CHARS            = 2000
MAX_HISTORY_MESSAGES       = 40
MAX_HISTORY_TOKENS         = 8000
MAX_FILTER_VALUES_PER_FIELD= 10
MAX_DECOMPOSED_QUERIES     = 6
MAX_GRAPH_EXPANSION_DEPTH  = 2
MAX_GRAPH_EXPANSION_ROWS   = 50
MAX_EXECUTOR_QUEUE         = 100
MAX_FULLTEXT_SCAN          = 500
BM25_TX_TIMEOUT_MS         = 2000
EXPANSION_TX_TIMEOUT_MS    = 1500
CONTEXT_K                  = 8
TOKEN_BUDGET               = 4096
EVAL_K                     = 8
```

### 13.2 Request
```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    chat_history: tuple[ChatMessage, ...] = ()
    as_of_date: Optional[date] = None
    occurred_at: Optional[date] = None
    ended_at: Optional[date] = None
    detected_date: Optional[date] = None
    event_date: Optional[date] = None      # DEPRECATED alias for occurred_at
    violation_state: Optional[Literal["completed", "ongoing", "unknown"]] = None
    legal_locators: tuple[str, ...] = ()   # optional explicit citations
    filters: Optional[ChatFilters] = None
```
Server-controlled and therefore **not** accepted from clients: `top_k`, `fetch_k`, `rerank_top`, `context_k`, `token_budget`, `provider`, `model`, `labels`.

### 13.3 Response Models (D141)
```python
class Outcome(str, Enum):                  # D142 — every path named
    FULL = "full"
    DEGRADED = "degraded"
    CACHED = "cached"
    CLARIFIED = "clarified"
    NO_ELIGIBLE_CONTEXT = "no_eligible_context"
    COVERAGE_UNSUPPORTED = "coverage_unsupported"
    CORPUS_COVERAGE_INCOMPLETE = "corpus_coverage_incomplete"
    TEMPORAL_RULE_UNSUPPORTED = "temporal_rule_unsupported"
    LEGAL_STATUS_REVIEW_PENDING = "legal_status_review_pending"
    GENERATOR_FAILED = "generator_failed"

class ErrorCode(str, Enum):
    VALIDATION_FAILED = "validation_failed"
    DATE_CONFLICT = "date_conflict"
    BODY_TOO_LARGE = "body_too_large"
    RATE_LIMITED = "rate_limited"
    CORPUS_NOT_READY = "corpus_not_ready"
    ALLOWLIST_NOT_POPULATED = "allowlist_not_populated"
    UPSTREAM_STORE_UNAVAILABLE = "upstream_store_unavailable"
    GENERATOR_FAILED = "generator_failed"
    INTERNAL = "internal"

class WarningKind(str, Enum):
    PENDING_AMENDMENT = "pending_amendment"
    COVERAGE_GAP = "coverage_gap"
    DEGRADED_COMPONENT = "degraded_component"
    AMBIGUOUS_CITATION = "ambiguous_citation"

class Warning(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: WarningKind
    message: str
    evidence_ref: Optional[str] = None
    related_provision_uid: Optional[str] = None

class ProblemDetails(BaseModel):           # RFC 7807 shape
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: str
    title: str
    status: int
    detail: str
    code: ErrorCode
    trace_id: Optional[str] = None
    errors: tuple[dict, ...] = ()

class CitationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provision_uid: str
    provision_version_id: str
    human_citation: str
    title: str
    official_url: str
    provision_path: tuple[str, ...]
    legal_time_basis: LegalTimeBasis
    review_pending: bool
    pending_evidence_ref: Optional[str] = None
    snippet: str

class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    accepted: bool
    trace_id: str
    idempotency_key: str
    duplicate: bool
```

### 13.4 SSE Events
Order is always `meta` → `sources`? → `token*` → exactly one terminal (`done` | `error`).
```
event: meta
data: {"state":"started","trace_id":"...","release_id":N,"publish_fence":N,
       "release_fingerprint":"...","as_of_date":"...","occurred_at":"...",
       "ended_at":null,"detected_date":null,"violation_state":"completed",
       "governing_dates":{"query_time":"...","conduct_time":"..."},
       "regime_map":{"query_time":["..."],"conduct_time":["..."]},
       "streaming_mode":"streaming"}

event: sources
data: [ CitationContext, ... ]          # full legal_time_basis per item (D140)

event: token
data: {"text":"..."}

event: done
data: {"trace_id":"...","outcome":"full","warnings":[Warning,...],
       "degraded_components":[]}

event: error
data: {"trace_id":"...","code":"generator_failed","message":"..."}
```
`meta` carries only immutable request/release facts plus `state:"started"`. Buffered providers emit exactly one `token` event containing the full text and declare `streaming_mode:"buffered"`.

### 13.5 Pre-Stream vs In-Stream Behavior (D146)
| Situation | Detected | Transport |
|---|---|---|
| Validation / date conflict / body too large | before stream | HTTP 4xx `ProblemDetails`; no SSE |
| `corpus_not_ready`, `allowlist_not_populated` | before stream | HTTP 503 `ProblemDetails` |
| Admission queue saturated | before stream | HTTP 429/503 + `Retry-After` |
| `clarify` | before retrieval, stream opened | `meta` → one `token` → `done(outcome=clarified)` |
| `coverage_unsupported` | before retrieval, stream opened | `meta` → one `token` (explanation) → `done(outcome=coverage_unsupported)` |
| `temporal_rule_unsupported` | before retrieval, stream opened | `meta` → one `token` → `done(outcome=temporal_rule_unsupported)` |
| `no_eligible_context` | after retrieval | `meta` → `sources`(empty) → one `token` → `done(outcome=no_eligible_context)` |
| `corpus_coverage_incomplete` | after coverage check | normal stream + `Warning(COVERAGE_GAP)` → `done(outcome=corpus_coverage_incomplete)` |
| `legal_status_review_pending` | pending lookup | normal stream (answer withheld per D120) + `Warning(PENDING_AMENDMENT)` → `done(outcome=legal_status_review_pending)` |
| Generator fails before first token | in stream | `error(code=generator_failed)`; **no** answer cache write |
| Generator fails after first token | in stream | truncate + `error(code=generator_failed)`; **no** answer cache write |
| Client disconnect | in stream | cancel and log; no `done` emitted; **no** answer cache write |

Buffered/non-cancellable providers (D144): the executor slot and accumulator are held until the provider's own timeout expires. The spec does not claim resources are released at disconnect for that case.

### 13.6 Date Validation Rules (D147)
```
1. All dates are date-only, parsed as ISO-8601, interpreted in Asia/Ho_Chi_Minh.
2. event_date present and occurred_at absent  → occurred_at = event_date.
3. event_date and occurred_at both present and different → 422 DATE_CONFLICT.
4. occurred_at and ended_at both present and ended_at < occurred_at → 422 DATE_CONFLICT.
5. detected_date and occurred_at both present and detected_date < occurred_at → 422 DATE_CONFLICT.
6. violation_state == "ongoing" and detected_date is None → 422 (or clarify for sanction intents).
7. violation_state == "completed" and detected_date present without occurred_at/ended_at
   → 422 DATE_CONFLICT (a completed conduct is governed by conduct time).
8. as_of_date in the future beyond today+1 day → 422 DATE_CONFLICT.
9. Any date outside [1945-01-01, today+365d] → 422 VALIDATION_FAILED.
```

### 13.7 Feedback
```python
class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace_id: str
    rating: Literal["helpful", "not_helpful"]
    comment: Optional[str] = Field(default=None, max_length=1000)
    idempotency_key: str
```
Validation uses the **local durable trace registry** (D145): a SQLite file on the `manifest_store` volume with `traces(trace_id PK, created_at, release_fingerprint, outcome)` and `feedback(idempotency_key PK, trace_id, rating, comment, created_at)`. A row is inserted for every request before streaming starts, so Langfuse being unavailable never loses feedback linkage. Unknown `trace_id` → 404. Duplicate `idempotency_key` → 200 with `duplicate=true`. Cache invalidation remains CLI-only; there is no admin HTTP endpoint in v1.

### 13.8 Health Endpoints (D156)
- `GET /livez` — process liveness only, no dependencies.
- `GET /readyz` — passes only when all of: active descriptor loadable; `coverage_manifest_hash` matches the artifact; `pending_index_hash` matches the artifact; `metadata_revision_set` present and immutable; Neo4j fulltext smoke query succeeds; Qdrant nested-filter smoke query succeeds; allow-list artifact present and non-empty; release reconciliation status recorded as passed. Otherwise `503` with the specific `ErrorCode`.
- `GET /metrics` — Prometheus exposition; no high-cardinality labels (never query text, provision ids, or trace ids as label values).

---

## 14. Caching

### 14.1 Canonicalization (D148/D149)
```python
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def cache_key(prefix: str, payload: dict) -> str:
    return f"{prefix}:{hashlib.sha256(canonical_json(payload).encode('utf-8')).hexdigest()}"
```
Every layer — answer, retrieval, context, amends — uses `SHA256(canonical JSON object)`. No `:`-joined composite strings are used as keys. Logical sets are sorted before hashing: `filters` values, `selected_regime_ids`, and `decomposed_queries` are sorted; any list whose order is semantically meaningful is preserved and flagged with an explicit `ordered: true` field inside the payload object so canonicalization does not silently reorder it.

### 14.2 Key Payloads
```python
answer_key = cache_key("cache:ans", {
    "rewritten_query": q, "intent": intent, "filters": sorted_filters,
    "governing_dates": gd.model_dump(mode="json"),
    "regime_map": rm.model_dump(mode="json"),
    "violation_state": vs, "release_fingerprint": desc.fingerprint,
    "embedding_version": ev, "reranker_version": rv,
    "model_name": mn, "prompt_wrapper_version": pwv,
    "retrieval_policy_fingerprint": rpf, "runtime_policy_fingerprint": rtf,
})
retrieval_key = cache_key("cache:ret", {
    "decomposed_queries": sorted(dq), "filters": sorted_filters,
    "governing_dates": gd.model_dump(mode="json"),
    "regime_map": rm.model_dump(mode="json"),
    "violation_state": vs, "release_fingerprint": desc.fingerprint,
    "embedding_version": ev, "reranker_version": rv,
    "retrieval_policy_fingerprint": rpf,
})
context_key = cache_key("cache:ctx", {
    "provision_version_id": pvid, "release_fingerprint": desc.fingerprint,
})
amends_key = cache_key("cache:amends", {
    "provision_uid": puid, "release_fingerprint": desc.fingerprint,
    "governing_dates": gd.model_dump(mode="json"),
    "regime_map": rm.model_dump(mode="json"),
})
```
`retrieval_policy_fingerprint = SHA256(canonical({aggregate, bm25_enabled, rerank_model_version, fetch_k, rerank_top, context_k, heuristic_enabled, code_schema_version}))`.
`runtime_policy_fingerprint = SHA256(canonical({prompt_wrapper_version, generator_config_version, sanction_guard_version}))`.

### 14.3 Lookup Points and Write Timing
| Layer | Lookup | Write |
|---|---|---|
| answer | after route + rewrite, for **both** `retrieve` and `cypher_query` branches (D94) | only after terminal `done` success (D143) |
| retrieval | after decompose, before search (retrieve branch) | immediately after retrieval + rerank success, even if generation later fails |
| context | before context materialization | after context build success (raw text, no date-dependent labels) |
| amends | before amends decoration | after amends resolution (date-resolved value) |

Invalidation is version-based via `release_fingerprint` inside every key; entries also carry a TTL (answer 24h, retrieval 6h, context 6h, amends 6h). Negative caching is not used. Redis roles are separate: `cache:*` failure is fail-open, `descriptor:*` failure falls back to the local file, `arq:*` failure stops ingestion without affecting serving.

---

## 15. Graph Schema

### 15.1 Scoping
Release-scoped nodes: content hierarchy (`Document`, `Part`, `Chapter`, `Section`, `Article`, `Clause`, `Point`) keyed by `snapshot_node_id`, plus AMENDS relationships. Immutable-per-revision nodes: metadata nodes keyed by `metadata_node_key` and `ExternalDocument` stubs keyed by `(external_document_key, external_revision)`.

### 15.2 Relationships and Canonical Edge Tuples
```
HAS_PART | HAS_CHAPTER | HAS_SECTION | HAS_ARTICLE | HAS_CLAUSE | HAS_POINT
  tuple = (release_id, rel_type, from_snapshot_node_id, to_snapshot_node_id)

BELONGS_TO_GROUP | HAS_TYPE | HAS_STATUS | ISSUED_BY | SIGNED_BY | IN_FIELD
  tuple = (release_id, rel_type, from_snapshot_node_id, metadata_node_key)

RELATED_TO  (→ ExternalDocument)
  tuple = (release_id, "RELATED_TO", from_snapshot_node_id,
           external_document_key, external_revision)

AMENDS {operation, edge_id, effective_rule_id, review_status, validity_basis,
        evidence_locator, extractor_version, prompt_version, confidence}
  tuple = (release_id, "AMENDS", amending_snapshot_node_id,
           target_snapshot_node_id, effective_rule_id, edge_id)
```
`edge_digest = SHA256(canonical(sorted(all edge tuples)))` and is compared during reconcile together with endpoint existence.

### 15.3 Constraints
Unique on `snapshot_node_id` for release-scoped nodes; unique on `metadata_node_key`; unique on `(external_document_key, external_revision)`; unique on `AMENDS.edge_id`. All writes are `MERGE`-based and idempotent. Content relationships connect nodes of the same release only; relationships to metadata/external nodes are release-scoped edges pointing at immutable revisions.

---

## 16. Monitoring and Evaluation

### 16.1 Real Dataset (D151/D135)
Verified source numbers (measured, not estimated):
- `qa_dataset/QA_NLP.csv`: **94 rows**, **94 unique questions**, 0 duplicates.
- `qa_dataset/QA_Part2345.csv`: **200 rows**, **199 unique questions**.
- Combined `QA_NLP + QA_Part2345`: **294 rows**, **293 unique questions**, **1 duplicate**.
- **100 rows** carry comma-separated multi-references.
- Referenced document identities: `100/2019/NĐ-CP`, `123/2021/NĐ-CP`, `15/2012/QH13`, `168/2024/NĐ-CP`, `35/2024/QH15`, `36/2024/QH15`, `44/2019/QH14`, `67/2020/QH14`.
- Reference strings look like `36/2024/QH15::article::38::clause::1`, i.e. legacy `docIdentity`-prefixed UIDs.

Migration (D135) produces `eval_dataset_v1`:
```python
class EvalLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label_id: str
    question: str
    legal_locators: tuple[str, ...]              # original reference strings
    source_document_ids: tuple[str, ...]         # migrated
    legal_document_keys: tuple[str, ...]         # migrated
    resolved_provision_version_ids: tuple[str, ...]
    successor_provision_version_ids: tuple[str, ...]
    as_of_date: date
    occurred_at: Optional[date]
    ended_at: Optional[date]
    detected_date: Optional[date]
    violation_state: Literal["completed", "ongoing", "unknown"]
    governing_dates: GoverningDates
    bound_governing_kind: GoverningDateKind
    release_fingerprint: str
    label_class: Literal["current", "historical", "ambiguous"]
    held_out: bool
    reference_answer: str
```
Migration rules: split `reference` on commas; parse each legacy UID into `(doc_identity, article, clause, point)`; resolve `doc_identity` → `legal_document_key` → `provision_uid` → `provision_version_id` for the target release; mark labels whose target document is superseded (for example `100/2019/NĐ-CP` after NĐ 168) as `historical`; mark labels that cannot be resolved unambiguously as `ambiguous`. Deduplicate the one duplicate question. Group-split by `legal_document_key` before the 80/20 split so no document appears in both splits. `historical` and `ambiguous` labels are excluded from the held-out set.

### 16.2 Freeze / Replay (D150)
```
fixture_key = SHA256(canonical({
    stage: "router" | "rewrite" | "decompose",
    query, chat_history, prompt_version,
    model_id, temperature, max_tokens }))
```
Fixtures live in `eval/fixtures/{stage}/{fixture_key}.json`. CI runs retrieval evaluation with `LLM_MODE=replay`; a missing fixture is a hard failure rather than a live call, so CI never depends on a provider. Pinned parameters (`model_id`, `temperature=0`, `max_tokens`, `prompt_version`) are recorded in the run fingerprint.

### 16.3 Metrics, Baseline, Promotion
`EVAL_K = 8`. Primary metric `Recall@8`; secondary `Precision@8`, `MRR@8`, `nDCG@8`; plus a token-budget assertion that no evaluated context set exceeds `TOKEN_BUDGET`.

Relevance (D153) is release-aware: a retrieved `provision_version_id` counts as relevant when, for the label's `release_fingerprint` and `governing_dates`, it either equals a `resolved_provision_version_id` or is reachable as a successor through **reviewed** AMENDS whose materialized interval is applicable at the label's bound governing date. Prefix matching on UIDs is never used.

Committed artifacts: `eval/baseline/{release_fingerprint}.json` (baseline metrics), `eval/splits/eval_dataset_v1.split.json` (explicit split membership), `eval/dataset/eval_dataset_v1.jsonl`, and a run fingerprint recording dataset hash, release fingerprint, model ids, prompt versions, and code schema version. Metric units are fractions in `[0,1]`.

Promotion rule: promote only when `Recall@8` is at least `baseline − 0.02` and `nDCG@8` is at least `baseline − 0.03`, both absolute, on the held-out split, and the token-budget assertion passes. The gate blocks deployment promotion, not every pull request.

Legacy evaluator removal (D152): prefix-match / `top_k=30` evaluators are deleted from the CI path; CI fails if such an entrypoint is invoked.

### 16.4 Online Observability
Langfuse receives a trace per request with spans for route, rewrite, decompose, pending lookup, search, rerank, and generation, plus feedback scores. Prometheus scrapes latency percentiles per stage, cache hit ratios per layer, store health, ingestion outcomes, and outcome counts by `Outcome` enum value. Grafana dashboards are provisioned from committed JSON.

---

## 17. Error Handling Summary

| Failure | Behavior |
|---|---|
| Descriptor unavailable (Redis and local file) | `503 CORPUS_NOT_READY` |
| Router LLM failure | fall back to `retrieve` |
| Rewriter failure | use the original query |
| Decomposer failure | single sub-query = original |
| Qdrant unavailable | dense retrieval impossible → `503 UPSTREAM_STORE_UNAVAILABLE` |
| Neo4j unavailable | serve from Qdrant payload + full degraded citation (D139); `degraded_components += ["neo4j","bm25","expansion","templates"]` |
| Both stores unavailable | `503 UPSTREAM_STORE_UNAVAILABLE` |
| Reranker failure | keep fused ordering; mark degraded |
| BM25 timeout / overfetch saturation | empty BM25 contribution; mark degraded |
| Generator failure before first token | one bounded retry, then `error(generator_failed)`; no cache write |
| Generator failure mid-stream | truncate + `error(generator_failed)`; no cache write |
| Redis cache failure | fail-open |
| Langfuse or Prometheus failure | fail-open; local trace registry still records the trace |
| Missing required facts | `clarify` before retrieval |
| Date in a temporal gap or outside verified range | `coverage_unsupported` |
| Unproven dependency closure in scope | `corpus_coverage_incomplete` + warning + gap in prompt |
| Empty regime map for all kinds | `temporal_rule_unsupported` |
| Pending amendment matched | `legal_status_review_pending` (+ sanction guard) |
| Admission queue saturated | `429`/`503` + `Retry-After` |
| `INSECURE_TLS=true` in production | startup FAILS |

---

## 18. Testing Strategy

### 18.1 Unit Tests (no network, no services, no DB/LLM import — D159)
Parser: alphanumeric clause numbers (`2a`, `18a`), footer split, quote-block tracking, NFC normalization, appendix preserve-by-default.
Identity: collision-map assignment and suffixing; resolver ambiguity rejection; `provision_version_id` full-hash with a collision registry over fixture texts; `snapshot_node_id` derivation; `metadata_node_key` collision-safety; ExternalDocument identity/version policy.
Provenance: `source_revision_identity_hash` changes when raw body, parser, or normalizer changes; manifest immutability; `RawManifest` schema round-trip; `EvidenceLocator` offset validation against a fixture normalized artifact (bounds and `excerpt_sha256`).
Temporal: `effectDate`/`expireDate` conversion table including nulls, `expireDate < effectDate`, and boundary days; deterministic interval split with one general plus multiple exceptions; no-overlap assertion; `RegimeMap` per-kind construction; empty-regime-set → `temporal_rule_unsupported`; intent→kind binding table.
Filter compiler: exact qdrant-client model tree; three branches; branch omission for absent dates; `to_days` null handling; Neo4j Cypher equivalence on the same fixtures.
Pending: `PendingAmendmentRecord` schema; `planned_effective_date is None` matches; the four resolution sources; sanction-guard regex blocking amounts; state-machine transitions.
AMENDS: all four op models plus `PatchSpec` validators; materialization preconditions; `edge_id` uniqueness; `reviewed_overrides` schema and promotion validation.
Release: descriptor schema and fingerprint determinism; `release_id` vs `publish_fence` independence; CAS ownership-token logic (simulated); artifact-hash-before-reconcile ordering.
Cache: canonical JSON key construction for all four layers; set ordering; policy fingerprints; version bump changes keys.
API: limits enforcement; all nine date-validation rules; `Outcome`/`ErrorCode`/`Warning`/`ProblemDetails`/`CitationContext`/`FeedbackResponse` schemas; SSE serialization; per-outcome pre-stream vs in-stream mapping.
Templates: all six request models; `extra="forbid"`; discriminated-union rejection routes to retrieval.
Eval: fixture-key derivation; replay-mode missing-fixture failure; release-aware relevance; `Recall@8`/`nDCG@8` computation on synthetic fixtures; label migration from comma-separated references.
Import-time: importing `legal_rag` performs no network or DB connection and constructs no LLM client; bounded-retry helper respects its deadline.

### 18.2 Integration Tests (docker-compose test profile)
Two concurrent builds write disjoint staging namespaces and neither overwrites the other's collection or nodes (D106/D107).
Reconcile catches a deliberately corrupted edge, a missing endpoint, a stale `derived_state_hash`, an unready fulltext index, and a Qdrant config drift.
Artifact hashes are written before reconcile; a post-publish artifact write attempt is rejected.
Promote-after-reconcile: a failed reconcile leaves the active pointer untouched and no `legal_v{N}` collection is created.
CAS: stale fingerprint returns 0; stale fence after rollback returns 0; equal base with a lower fence loses; bootstrap under a foreign lock token returns -1; crash between CAS and local-file write recovers from Redis.
Backup/restore executes in the mandated order and `/readyz` stays 503 until the pointer is activated.
Qdrant filter fixture executes the exact compiled filter against a real Qdrant container and returns only intervals matching kind, state, regime, and date bounds within a single interval element (D136).
BM25 bounds: `queryNodes` limit caps the scan; a release N-1 row never enters the top-k for release N; a forced timeout produces an empty BM25 contribution with the degraded flag.
Expansion bounds are enforced inside Cypher (depth, row cap, tx timeout) and the final selection still caps at 8 contexts and the token budget.
Temporal: NĐ 168 general 2025-01-01 versus the 2026-01-01 exception yields no overlapping segment for exception provisions; conduct-time 2023 selects the NĐ 100 regime while query-time 2026 selects NĐ 168, and the two never mix in one unlabeled answer.
Pending: an auto amendment whose target is filtered out of top-k still produces `legal_status_review_pending`; a pending record with `planned_effective_date = None` produces `temporal_rule_unsupported` when the answer depends on it; a penalty question under a pending amendment returns no amounts.
Parse-fail carry-forward preserves the document node, provisions, metadata revision reference, pending records, and coverage entry, with `carried_forward = true`.
Metadata immutability: release N updating metadata leaves release N-1 title, URL, effectivity, status, and citation unchanged; reconcile verifies the referenced revision set.
GC reference guard refuses to delete raw artifacts referenced by a retained release, review log, reviewed override, eval label, or citation registry.
Cache: answer cache is not written on pre-token failure, mid-stream error, or disconnect; retrieval cache is written even when generation later fails; the Cypher branch performs both lookup and write.
Local trace registry records every request and accepts feedback while Langfuse is stopped.
`/readyz` fails when the coverage hash, pending hash, metadata revision set, fulltext smoke query, Qdrant smoke query, allow-list artifact, or reconciliation status is missing.
Compose exposure: only `nextjs` and `api` publish host ports; Neo4j, Qdrant, Redis, and Langfuse services are unreachable from the host.
`INSECURE_TLS=true` fails startup under the production profile and is accepted only under the dev profile.

### 18.3 Acceptance Tests
| # | Scenario | Expected |
|---|---|---|
| A1 | Two workers build concurrently | disjoint staging; one publishes; other rebases |
| A2 | Stale build after rollback | CAS returns 0 |
| A3 | Bootstrap with foreign lock token | CAS returns -1 |
| A4 | Restore then activate | `/readyz` 503 until pointer activated |
| A5 | Post-publish artifact write | rejected |
| A6 | Conduct-time 2023 penalty question | NĐ 100 regime only, labeled basis |
| A7 | Same question with `as_of_date` 2026 and no conduct facts | clarify before retrieval |
| A8 | Explicit before/after comparison | two labeled branches, separate citations |
| A9 | NĐ 168 exception provision at 2025-06 | not applicable; exception owns 2026 onward |
| A10 | Pending amendment, target not in top-k | `legal_status_review_pending` + warning |
| A11 | Pending amendment with unknown effective date | `temporal_rule_unsupported` |
| A12 | Penalty question under pending amendment | no amounts; warning + citations |
| A13 | Empty regime map for all kinds | `temporal_rule_unsupported` |
| A14 | Date inside a temporal gap | `coverage_unsupported` |
| A15 | Unproven dependency closure | `corpus_coverage_incomplete` + gap in prompt |
| A16 | Parse failure for one document | full carry-forward; document still answerable |
| A17 | Metadata updated in N | N-1 citation text unchanged |
| A18 | docIdentity collision | suffixed keys; resolver rejects ambiguity; both documents ingested |
| A19 | Neo4j down | Qdrant-only answer with full degraded citation |
| A20 | BM25 timeout | empty BM25 contribution + degraded flag |
| A21 | Expansion attempts 500 rows | capped in DB at 50; final ≤8 contexts |
| A22 | Mid-stream generator failure | truncated stream, error event, no answer cache |
| A23 | Client disconnect | cancel + log; no `done`; no cache write |
| A24 | Buffered provider disconnect | slot held to provider timeout; no false release claim |
| A25 | Langfuse down, feedback submitted | accepted via local trace registry |
| A26 | Duplicate idempotency key | 200 with `duplicate=true` |
| A27 | `ended_at < occurred_at` | 422 `DATE_CONFLICT` |
| A28 | `violation_state=ongoing` without `detected_date` | 422 or clarify |
| A29 | Empty allow-list artifact | `/readyz` 503 `allowlist_not_populated` |
| A30 | Eval in replay mode with a missing fixture | CI failure, no live LLM call |
| A31 | Legacy prefix-match evaluator invoked | CI failure |
| A32 | `Recall@8` below baseline − 0.02 | promotion blocked |
| A33 | Label whose target is repealed | classified `historical`, excluded from held-out |
| A34 | Embedding model revision changed | vectors re-embedded, not reused |
| A35 | Only `nextjs` and `api` reachable from host | stores unreachable |
| A36 | Workspace secret scan | fails on any real credential in the tracked tree |
| A37 | Submodule acceptance | `.gitmodules` + gitlink committed; CI asserts SHA |

### 18.4 CI Jobs
`unit` (no network, no services) on every pull request; `schema` validating every Pydantic model plus canonical-JSON round-trips; `integration` on the compose test profile for changes to parser, ingestion, release, retrieval, cache, or API; `acceptance` on release candidates; `eval-replay` for retrieval-affecting changes; `secret-scan` across the whole workspace including `NLP-LegalQA/` and `docs/`; `submodule-assert` verifying `.gitmodules`, the gitlink, and the pinned SHA. The promotion gate runs only after `eval-replay` and `acceptance` pass.

---

## 19. Delivery, Security, Repo Truth

### 19.1 Verified Repo State (D157)
Measured at v9 authoring time:
```
$ git ls-files
docs/superpowers/specs/2026-08-14-traffic-law-rag-design.md      # 1 file total

$ git status --short
?? NLP-LegalQA/                                                  # untracked
```
There is no `src/`, no `frontend/`, no `docker-compose.yml`, no `tests/`, no `eval/`, and no `.gitmodules`. **No part of this system is production-ready.** The spec describes the target; nothing in it may be reported as implemented.

### 19.2 Delivery Acceptance Gates
1. `.gitmodules` committed with the `NLP-LegalQA` entry at a pinned SHA.
2. The gitlink tree entry for that SHA committed.
3. CI checks out with `--recurse-submodules`.
4. CI asserts the submodule SHA equals the expected pin.
5. `docker-compose.yml` plus `docker-compose.dev.yml` and `docker-compose.test.yml` committed, with pinned image digests, healthchecks, named volumes, and host ports published only for `nextjs` and `api`.
6. `src/legal_rag/**`, `frontend/**`, `tests/{unit,integration,acceptance}/**`, and `eval/**` committed.
7. All CI jobs in §18.4 green.
Until gates 1–7 pass, the project is a design plus reference study and must be described as such.

### 19.3 Security (D158)
Workspace-wide secret scanning (gitleaks or trufflehog) runs over the entire tree, including the reference path and documentation, and fails the build on any finding. Verified finding to respect: `NLP-LegalQA/CLAUDE.md` around lines 217–219 contains a real-looking Neo4j host and password (`nguyenhoangquan.com:7687`, `Neoneo4j`). Those values are inside the read-only reference; they are never copied into this project's code, tests, docs, compose files, or fixtures, and the reference owner should rotate them. This project's `.env.example` uses placeholders only.

TLS verification is on by default. `INSECURE_TLS=true` causes startup to fail under the production profile and is honoured only under the dev profile. The reference project's `verify=False` pattern is never used in the serving path. Neo4j uses two principals: read-only for the API, and write/schema/publish for the worker. Secrets come only from environment files that are gitignored.

### 19.4 Deployment Profile
All images pinned by digest: Python base, Node base, Neo4j 5.x, Qdrant 1.x, Redis 7.x, Langfuse web and worker, Postgres, ClickHouse, Valkey, MinIO, Prometheus, Grafana. Volumes: `neo4j_data`, `qdrant_storage`, `redis_data`, `raw_corpus`, `manifest_store` (release manifests, coverage, pending index, reviewed overrides, review log, collision map, `active_release.json`, trace registry), `quarantine_store`, `langfuse_postgres`, `langfuse_clickhouse`, `langfuse_valkey`, `langfuse_minio`, `prometheus_data`, `grafana_data`.

Resource guidance for a single machine: Langfuse stack roughly 4 GB RAM and 20 GB disk; Neo4j roughly 2 GB heap plus 1 GB page cache; Qdrant roughly 1 GB for this corpus size; the API needs GPU VRAM for the reranker or falls back to CPU. The API runs a single uvicorn worker so the in-process GPU semaphore is sufficient; scaling to multiple workers would require a distributed lease and is out of v1 scope.

### 19.5 Open Items (environment-dependent only)
1. `allowlist_bootstrap.json` must be generated by a facet-discovery run and committed with its hash before the first release (blocking, tracked as a bootstrap task rather than a design gap).
2. Exact HuggingFace model revision hashes to pin.
3. Langfuse secrets generation and host resource allocation.
4. `updDateTime` capability-probe result, which selects the delta strategy.

---

## 20. Resolved / Remaining / Explicitly Out of Scope

### 20.1 Resolved in v9 (schema + failure behavior + test named)
| Item | Contract | Test |
|---|---|---|
| Atomic release_id reservation and build-scoped staging | §8.2, D106/D107 | A1, 18.2 concurrent builds |
| Immutable artifacts hashed before reconcile; no post-publish writes | §8.5, D108 | A5, 18.2 artifact ordering |
| Pending index immutable per release; Redis cache-only | §8.5, D109 | 18.2 reconcile/pending hash |
| Restore order with pointer last | §8.6, D110 | A4 |
| CAS ownership-token bootstrap | §8.3, D111 | A3 |
| Per-kind regime mapping | §5.3, D112 | A6, A8 |
| Empty regime set behavior | §5.3, D113 | A13 |
| Intent→governing-kind binding | §5.4, D114 | A6, A7, A8 |
| effectDate/expireDate exact conversion | §5.7, D115 | 18.1 conversion table |
| Deterministic interval split | §5.6, D116 | A9, 18.1 split |
| Full temporal schemas | §5.1/5.7/5.8, D117 | 18.1 schema |
| Unknown pending effective date never skipped | §10.3, D118 | A11 |
| Pending lookup from free text | §10.3, D119 | A10 |
| Sanction-safety suppression | §10.3, D120 | A12 |
| Pending record schema and state machine | §10.1/10.2, D121/D122 | 18.1 pending |
| Full AMENDS + PatchSpec + LegalLocator schemas | §10.4, D123 | 18.1 AMENDS |
| EvidenceLocator offsets and validation | §10.4, D124 | 18.1 offset validation |
| reviewed_overrides schema and promotion | §10.6, D125 | 18.1 promotion |
| Allow-list bootstrap artifact (no inline placeholder) | §6.1, D126 | A29 |
| corpus_coverage_incomplete surfaced | §6.3, D127 | A15 |
| Coverage gap taxonomy, per kind/regime/document | §5.8, D128 | A14 |
| Delta scan contract | §6.4, D129 | 18.2 delta |
| Parse-fail full carry-forward | §6.5, D130 | A16 |
| metadata_node_key collision safety | §4.4, D131 | 18.1 metadata key |
| ExternalDocument identity/version | §4.4, D132 | 18.1 external stub |
| Raw manifest full schema | §7.2, D133 | 18.1 manifest |
| GC reference guard | §7.3, D134 | 18.2 GC guard |
| QA label migration with real numbers | §16.1, D135/D151 | A33, 18.1 migration |
| Qdrant filter fixture on real instance | §5.5, D136 | 18.2 filter fixture |
| BM25 executable bounds | §12.1, D137 | A20, 18.2 BM25 |
| DB-level expansion bounds | §12.2, D138 | A21 |
| Full degraded citation | §11.2/§12.3, D139/D140 | A19 |
| Complete API models and outcome enum | §13.3, D141/D142 | 18.1 API schema |
| Streaming order and cache timing | §13.5/§14.3, D143 | A22, A23 |
| Buffered provider slot honesty | §13.5, D144 | A24 |
| Local durable trace registry | §13.7, D145 | A25, A26 |
| Per-outcome SSE behavior | §13.5, D146 | 18.1 SSE mapping |
| Date validation rules | §13.6, D147 | A27, A28 |
| Canonical keys all layers and ordering | §14.1/14.2, D148/D149 | 18.1 cache keys |
| Eval freeze/replay | §16.2, D150 | A30 |
| Legacy evaluator removed from CI | §16.3, D152 | A31 |
| Release-aware relevance | §16.3, D153 | 18.1 relevance |
| Langfuse full closure and exposure policy | §3.1, D154/D155 | A35 |
| /readyz full checks | §13.8, D156 | 18.2 readyz |
| Repo-truth statement | §19.1, D157 | A37 |
| Workspace secret scan | §19.3, D158 | A36 |
| Root CI requirements | §18.4, D159 | 18.1 import-time |

### 20.2 Remaining (implementation work, gated)
| Item | Gate |
|---|---|
| Generate and commit `allowlist_bootstrap.json` with non-empty ids | `/readyz` blocks serving until present |
| Pin HuggingFace model revision hashes | recorded in release descriptor |
| Create `src/`, `frontend/`, compose files, tests, eval | §19.2 gates 5–6 |
| Wire `.gitmodules` and gitlink | §19.2 gates 1–4 |
| Build `eval_dataset_v1` from the 294-row source and commit splits | `eval-replay` job |
| Generate Langfuse secrets and size host resources | compose up |

### 20.3 Explicitly Out of Scope (v1)
Multi-worker GPU coordination via distributed lease; fine-tuning training; free-form LLM Cypher; per-document custom temporal evaluators beyond the RuleKind enum; multi-tenant auth; OCR; reverse-amendment discovery beyond reachable documents; distributed ANN or sharding.

---

## Changelog: v8 → v9

| Area | v8 problem | v9 resolution |
|---|---|---|
| Release reservation | Two workers could pick the same N and share staging | Atomic `INCR release_ctr` plus per-build `build_id`; staging namespaces `staging_{build_id}`; promote only after reconcile (D106/D107) |
| Artifact mutability | Artifacts could be written after publish; pending index lived mutable in Redis | All release-referenced artifacts written immutably and hashed before reconcile; Redis pending is a cache only (D108/D109) |
| Restore ordering | Unspecified | Mandatory order with pointer activation last (D110) |
| CAS bootstrap | Checked lock existence only | Verifies ownership token (D111) |
| Regime resolution | Single collapsed reference date | Per-kind `RegimeMap`; each branch uses its own regime set; empty-all → `temporal_rule_unsupported` (D112/D113) |
| Law mixing | Possible implicit mixing | Explicit intent→kind binding; comparisons served as labeled branches (D114) |
| Date semantics | Inclusive/exclusive unstated | `effectDate` inclusive → `from`; `expireDate` inclusive → `to = +1 day`; null/invalid/conflict rules (D115) |
| Interval split | Described as "truncate" | Deterministic boundary-set algorithm with total-order rule selection (D116) |
| Temporal schemas | Partial | Full schemas for ledger, coverage gap, regime coverage, document coverage, interval, governing dates, regime map (D117) |
| Pending unknown date | `None` failed the `<=` test and was skipped | `None` always matches; maps to `temporal_rule_unsupported` or review-pending (D118) |
| Pending from free text | Required known locator | Four resolution sources including a lexical index (D119) |
| Sanction safety | Not enforced | Generator instruction plus post-generation guard blocking amounts (D120) |
| AMENDS schemas | Names only for some ops | Complete Pydantic for four ops, PatchSpec, LegalLocator, successor version requirement (D123) |
| EvidenceLocator | Offset unit and manifest id unstated | Manifest id, parser/normalizer versions, unicode-code-point offsets, excerpt encoding/hash, pre-apply validation (D124) |
| reviewed_overrides | Prose only | Full schema with promotion id, reviewer, decision, scope, interval, evidence, hash (D125) |
| Allow-list | `<populated at bootstrap>` placeholder inside a "concrete" section | Required bootstrap artifact with schema and hash; serving blocked until present (D126) |
| Corpus completeness | Answered as if complete | `corpus_coverage_incomplete` outcome with gap in prompt and citations (D127) |
| Coverage model | Single window | Gap taxonomy plus per kind/regime/document checks (D128) |
| Delta scan | Watermark semantics unstated | Full cursor contract with overlap, capability probe, persistence, full reconciliation (D129) |
| Parse-fail | Text-only carry-forward | Document, provisions, metadata reference, pending records, coverage all carried (D130) |
| Metadata key | Metadata hash alone | `SHA256({source_document_id, metadata_revision_hash})` (D131) |
| ExternalDocument | Identity policy unstated | Keyed identity kinds plus monotonic revisions (D132) |
| Raw manifest | Partial fields | Full JSON schema including parse-quality metrics and provenance chain (D133) |
| GC | Could orphan evidence | Reference guard across releases, review log, overrides, eval labels, citations (D134) |
| Eval dataset | Invented counts | Real measured numbers: 94 / 294 rows, 293 unique, 1 duplicate, 100 multi-reference rows, 8 document identities (D151) |
| Label migration | Unspecified | Explicit migration and `historical`/`ambiguous` classification, group-split (D135) |
| Qdrant filter | Pseudocode | Exact qdrant-client model tree with nested conditions plus a real-instance fixture test (D136) |
| BM25 bounds | Prose constant | `queryNodes` limit, tx timeout, row cap, degraded behavior (D137) |
| Expansion bounds | Post-fetch truncation | Bounds inside the Cypher query (D138) |
| Degraded citation | Partial fields | Full citation payload including regime, rule kind, effective rule, governing date, pending evidence (D139/D140) |
| API models | Names only | Complete models plus ten-value `Outcome` enum and error codes (D141/D142) |
| Streaming order | Diagram said "write cache → stream" | Emit first, tee to accumulator, cache only after terminal success (D143) |
| Buffered provider | Claimed immediate release | Slot held to provider timeout (D144) |
| Feedback | Depended on Langfuse | Local durable SQLite trace registry (D145) |
| SSE per outcome | Unspecified | Pre-stream vs in-stream table (D146) |
| Date validation | Partial | Nine explicit rules (D147) |
| Cache keys | `:`-joined for context/amends | Canonical JSON hashing everywhere plus ordering rules (D148/D149) |
| Eval reproducibility | Prose | Fixture keys, replay mode, pinned parameters, committed baseline and splits, promotion rule (D150) |
| Legacy evaluator | Labeled deprecated | Removed from CI; invocation fails (D152) |
| Relevance | Static successor graph | Release-aware with applicable intervals (D153) |
| Langfuse closure | Single service line | Web, worker, Postgres, ClickHouse, Valkey, MinIO with healthchecks and volumes (D154) |
| Exposure | Unstated | Only `nextjs` and `api` publish ports (D155) |
| /readyz | Partial | Eight concrete checks including two smoke queries (D156) |
| Repo truth | Aspirational | Verified `git ls-files` = 1 file; explicit non-production statement (D157) |
| Secret scan | Repo-only | Workspace-wide including reference and docs; verified finding recorded (D158) |
| CI | Partial | Compose test profile, no import-time DB/LLM, bounded retries (D159) |
