# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-14  
**Status:** Draft v4 (post-Codex round-3 revision — snapshot isolation, historical version model, temporal semantics, AMENDS materialization)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (vendored as git submodule at pinned SHA). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Code and infrastructure are production-shape; deployable to cloud if desired.
- Right-sized for a bounded corpus (Vietnamese traffic-law documents, thousands of chunks). No fake scale claims.
- Legally sound validity model at provision level with explicit temporal semantics (`as_of_date`, `event_date`).
- **True snapshot isolation**: every serving request pins exactly one immutable release descriptor; no half-visible state during ingest.
- **Legal history preserved**: tombstone closes validity intervals only; never physical-deletes history due to reparse success or snapshot retention.

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters. Out of scope for this corpus.
- Fine-tuning pipeline (QLoRA/Unsloth). Eval harness supports evaluating fine-tuned models but training is not shipped.
- Public deployment or multi-tenant auth in v1. Single-user demo.
- OCR. Source API returns structured HTML/JSON, not scanned PDFs.
- LLM-based legal validation of amendments. Pydantic validates structure, not legal meaning.
- Free-form Cypher from LLM in v1. Text2Cypher uses server-owned read templates only.

---

## 2. Decisions Log

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Reality bar | Self-host docker-compose, production-shape | Right-sized for portfolio |
| D2 | Feature scope | Core RAG + amendment graph + chat/routing/template-Cypher + eval/monitoring. No fine-tune | Covers all 5 pillars |
| D3 | Data store | Neo4j (graph + BM25) + Qdrant (ANN + metadata + context snapshot) | Separation matches "metadata-first + dedicated ANN" |
| D4 | LLM provider | Provider-neutral OpenAI-compatible for ALL LLM calls including AMENDS extraction and template selection | Same factory everywhere |
| D5 | Monitoring | Langfuse (official self-host compose) + Prometheus + Grafana | Complementary concerns |
| D6 | UI | Next.js/React with SSE streaming via fetch() ReadableStream | POST JSON body incompatible with native EventSource |
| D7 | Ingestion queue | arq (async Redis queue) | Async-native, built-in cron + retry, right-sized vs Celery |
| D8 | Cache | Redis (answer + retrieval + context), key includes full dependency fingerprint | Cost + latency controller |
| D9 | Service topology | Modular monolith (API container) + separate arq-worker container + infra sidecars | GPU contention avoided via profile; same codebase |
| D10 | Vector storage | Qdrant only; drop Neo4j vector index | Avoids duplicate embed |
| D11 | BM25 placement | Neo4j fulltext (Lucene) as baseline hypothesis | Validated through quality gate, not assumed superior |
| D12 | Corpus scope | Versioned allow-list; NO effect_status gate; dependency closure for amending docs | NĐ 168 = "Hết Hiệu lực một phần" must be included |
| D13 | TLS verification | Default ON. Dev-only override via `INSECURE_TLS=true` env flag | Production default verifies credentials |
| D14 | Chunk limit | Max 8 serving contexts AND token budget (default 4096); expansion counts toward budget | Dual cap; eval_k separate from serving_k |
| D15 | Identity scheme | Seven-tier identity (see §4); content hash NOT in stable legal keys | docIdentity collides; stable keys needed for AMENDS/citations/QA labels |
| D16 | Qdrant point ID | UUIDv5(fixed_namespace, provision_version_id) per-release collection | Snapshot isolation via separate collections; UUID within collection |
| D17 | Temporal model | Half-open `[from, to)`; validity states {active, inactive, conditional, unknown}; effectivity_extract stage | Interval correctness; conditional not eligible by default |
| D18 | Text2Cypher v1 | TemplateCall{kind, params} discriminated union; fixed server queries; mandatory LIMIT + validity predicate + canonical UID return | Eliminates Cypher injection; invalid selection routes to retrieval |
| D19 | SSE contract | meta(state:started) → sources? → token* → done\|error. Buffered mode = one token event with full text. HTTP problem before stream open. | Explicit protocol; consistent across intents |
| D20 | Secrets policy | Env vars only, `.env` gitignored, test placeholders, docs `<placeholder>` | Credential safety |
| D21 | Release/snapshot isolation | Immutable release_descriptor per publish; Qdrant per-release collection; Neo4j release_id filter; single pointer (descriptor); durable local fallback | True snapshot isolation; no half-visible state |
| D22 | AMENDS operations | ADD_PROVISION / REPLACE_TEXT / REPEAL / AMEND; LegalLocator output; auto = annotation only; reviewed = eligibility change; evidence locator verified | Safe semantics; auto edges don't change eligibility |
| D23 | Clause numbering | Parser accepts alphanumeric `\d+[a-z]?` (e.g., `2a`, `18a`) | Raw data contains alphanumeric clauses |
| D24 | Appendix handling | Default preserve when no reviewed rule; exclude only via verifiable rule + test fixture | Phụ lục can contain binding legal rules |
| D25 | External references | ExternalDocument stubs keyed by source_document_id (docGUID), not doc_identity | Unambiguous internal key |
| D26 | Health endpoints | /livez (no deps), /readyz (checks active reconciled release), /metrics. No remote LLM call in probes | Fast cheap probes |
| D27 | GPU profile | API reranker GPU, ingestion embedder CPU default (optional off-peak GPU) | Two containers; Python semaphore doesn't cross processes |
| D28 | Redis roles | Three namespaces: arq:, cache:, descriptor. Descriptor has durable local fallback (active_release.json). Broker outage ≠ snapshot failure. | Clear separation of concerns |
| D29 | Historical versions | Each release contains ALL provision versions within temporal support window; node identity = provision_version_id; tombstone = interval close only | Legal history preserved; multiple versions per lineage |
| D30 | Provision version granularity | provision_version_id based on provision_text_hash (not whole-doc hash) | Amending Điều 1 doesn't churn other provisions' point IDs |
| D31 | Reconcile manifest | Publish requires manifest match: {provision_version_id set, derived_state_hash per point, count}. Not just count. | Detects silent corruption |
| D32 | GC grace | Delete old release snapshots only after grace > max request/SSE deadline. Never delete provision version history within temporal support window. | In-flight requests complete safely |
| D33 | Delivery | Git submodule at pinned SHA; commit .gitmodules + gitlink; CI checkout recursive + assert SHA | Reproducible vendoring |
| D34 | Cache lookups | Two distinct points: (1) answer cache after route/rewrite; (2) retrieval cache after decompose, before search/rerank | Correct ordering; key depends on rewritten query + intent |
| D35 | Context cache | Store raw context text (no date-dependent labels); decorate validity per-request based on reference_date | Date-independent caching |
| D36 | Retry generation | Only before first token streamed; once streaming started, no retry | Post-stream retry produces garbled/duplicate output |
| D37 | Effect status not scope gate | Allow-list does NOT filter by document effect_status | NĐ 168 = "Hết Hiệu lực một phần"; gating excludes it incorrectly |
| D38 | Dependency closure | Documents amending/repealing in-scope provisions ingested fully regardless of keyword/field match | Amendment coverage |
| D39 | Search strategy | Literal search_queries sent upstream; regex/local matching post-download; no wildcard API keywords; updDateTime used only after capability probe | Respect API capabilities |
| D40 | Conditional eligibility | conditional/unknown NOT eligible by default without reviewed resolution; warning/clarification instead | Prevents serving unverified conditional provisions as effective law |
| D41 | requires_event_date | Server-side classifier determines if query requires event_date; penalty questions without event_date → clarify or conditional answer BEFORE retrieval | Executable rule, not a promise |
| D42 | Backup manifest | One common backup_manifest covering Neo4j dump, Qdrant collections, raw corpus, manifests, quarantine. Cache/arq jobs ephemeral, NOT restored as serving state. | Consistent restore |
| D43 | Citation projection | Qdrant payload includes title, official_url, human_citation, provision_path, validity_intervals, applicability_basis, snippet | Neo4j-down still renders sources[] |

---

## 3. Architecture Overview

### Service Map (Two Application Containers)
```
┌─────────────────────────── docker-compose ───────────────────────────┐
│                                                                      │
│  FRONTEND                                                            │
│  ├─ nextjs          : React chat UI, dark theme,                     │
│  │                   SSE via fetch() ReadableStream,                  │
│  │                   citation panel, feedback button                 │
│                                                                      │
│  APP CONTAINERS                                                      │
│  ├─ api             : FastAPI — validate→trace→dates→pin release     │
│  │                   →route→rewrite→cache(1)→decompose              │
│  │                   →cache(2)→retrieve→rerank(GPU)→gen             │
│  │                   Endpoints: POST /v1/chat, POST /v1/feedback,    │
│  │                   GET /livez, GET /readyz, GET /metrics           │
│  ├─ arq-worker      : ingestion offline (scrape→parse→effectivity    │
│  │                   →amends→embed(CPU)→import→reconcile→publish)   │
│  │                   Same codebase, different entrypoint             │
│                                                                      │
│  DATA STORES                                                         │
│  ├─ neo4j           : graph hierarchy + AMENDS + Lucene BM25         │
│  │                   Nodes keyed by provision_version_id            │
│  │                   Filtered by release_id per snapshot            │
│  │                   Text2Cypher read-only principal                │
│  ├─ qdrant          : Per-release collections (legal_v{N})          │
│  │                   HNSW dense + payload metadata + context        │
│  │                   Full citation projection (§12/D43)             │
│  └─ redis           : arq broker (arq:*)                            │
│                       cache (cache:*)                               │
│                       active_release_descriptor (descriptor:*)       │
│                       Config: AOF ON, noeviction                    │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : LLM trace, feedback, eval experiment          │
│  │                   (official self-host compose profile)            │
│  ├─ prometheus      : infra metrics, scrapes GET /metrics           │
│  └─ grafana         : dashboard                                     │
└──────────────────────────────────────────────────────────────────────┘
```

Langfuse deployed per https://langfuse.com/self-hosting/ with pinned versions, persistent volumes, secrets via env. Resource budget documented in deployment guide.

### INVAR 1 (Snapshot Isolation)
Every serving request pins exactly one release_descriptor at ingress. All reads — Qdrant collection, Neo4j release_id, BM25, hierarchy traversal, Text2Cypher templates, cache lookup, citations, SSE meta — use that pinned descriptor. No read crosses releases.

### Request Path (Hot Path, Pillar 3)
```
POST /v1/chat {query, chat_history?, as_of_date?, event_date?, filters?}
  │
  ▼ validate input (schema, lengths, ISO dates, timezone Asia/Ho_Chi_Minh)
  ▼ create fresh trace (new trace_id for EVERY request, including cache hits)
  ▼ PIN RELEASE DESCRIPTOR (read once, pass to all downstream reads)
     If unavailable: serve last pinned from local fallback; if none: 503
  ▼ resolve dates:
     reference_date = event_date if provided else as_of_date (default today)
     legal_time_basis = "event" | "current"
  ▼ requires_event_date check (§5):
     If penalty question lacks event_date → respond conditionally or ask clarification BEFORE retrieval
  ▼ route (LLM, temp=0, max_tokens=64)
     intent ∈ {greeting, cypher_query, retrieve, reject}
  │
  ├─ reject → template refusal, write trace, return done
  ├─ greeting → deterministic response (no LLM, no retrieval), write trace, return done
  ├─ cypher_query → rewrite if history → TemplateCall selection (LLM)
  │   → execute read template (pinned release + validity predicate + LIMIT)
  │   → gen natural language → write cache + trace → stream
  │
  └─ retrieve:
       ▼ rewrite (multi-turn; skip if no history)
       ▼ CACHE LOOKUP (1) — ANSWER CACHE (after route+rewrite):
          key = sha256(rewritten_query + intent + filters + as_of_date
              + event_date + reference_date + legal_time_basis
              + release_descriptor.fingerprint
              + embedding_version + reranker_version
              + model_name + prompt_version + policy_version)
          HIT → new trace (cache_origin_trace_id recorded), skip everything below
          MISS → continue
       ▼ decompose (LLM, JSON sub-queries, append original;
          code enforces max_subqueries=6 + dedup by normalized string)
       ▼ CACHE LOOKUP (2) — RETRIEVAL CACHE (after decompose, before search):
          key = sha256(decomposed_queries_sorted + filters + reference_date
              + release_descriptor.fingerprint
              + embedding_version + reranker_version)
          HIT → skip search+rerank, continue to context build
          MISS → search+rerank below
       ▼ multi_search (sub-queries parallel, PINNED release collection):
          Each sub-query:
            • Qdrant: eligibility predicate [from,to) against reference_date
              (nested interval evaluation) + metadata filter
              → HNSW within eligible set in pinned collection
            • Neo4j: BM25 fulltext with same eligibility predicate
              filtered by pinned release_id
          Fuse RRF per sub-query → aggregate across sub-queries
       ▼ fetch context from Qdrant payload context_text (primary)
          or Neo4j hierarchy traversal (fallback, pinned release_id)
       ▼ cross-encoder rerank (GPU, FP16, batch_size small)
       ▼ heuristic rerank (recency bonus AFTER eligibility gate, eval-gated)
       ▼ select top serving contexts (max 8 AND token budget 4096)
       ▼ expand context (optional: sibling Points, Article children;
          expansion counts toward token budget; pinned release)
       ▼ decorate validity labels per-request using reference_date
          (NOT cached — computed fresh from eligibility state)
       ▼ build context_str (tag [ĐÃ BÃI BỎ]/[SỬA ĐỔI] per reference_date,
          amends note, provisional annotations for auto-review-status edges)
       ▼ gen (LLM, _QA_FEW_SHOT_SYSTEM_PROMPT)
          Retry ONLY before first token. Once streaming starts, no retry.
       ▼ write back caches (answer + retrieval)
       ▼ stream SSE events: meta(state:started) → sources → token* → done|error
       ▼ write trace to Langfuse + metrics to Prometheus
```

### Ingestion Path (Offline, Pillar 1) — Separate Container
```
cron/manual trigger → arq enqueue (arq: namespace)
  ▼ scrape phapluat.gov.vn (versioned allow-list, NO effect_status gate,
    dependency closure for amending docs, delta via updDateTime
    + content_hash + overlap window, capability-probed).
    TLS verify ON (dev override available). Throttle 0.5–1s.
    Save raw immutable: HTML + JSON + normalized text + official URL
    + fetched_at + parser/normalizer version + evidence offsets.
    Content-addressed blob store (path = raw_blob_hash). Dedupe blobs.
    Per-revision manifest always written (metadata-only updates preserved).
    Never overwrite existing raw files.
  ▼ parse hierarchy (regex accepts alphanumeric clause numbers \d+[a-z]?).
    Footer split with configurable appendix decision (default preserve).
    Quote-block tracking. NFC normalize. Idempotent keyed by
    (source_document_id, source_metadata_revision_hash).
    Multi-value field normalization.
  ▼ parse-quality gate: raw length sanity, anchor presence (Điều markers),
    hierarchy count/coverage ratio vs previous revision.
    If partial/bad → keep last-good revision, do NOT tombstone.
  ▼ effectivity_extract: extract "Hiệu lực thi hành" + "Điều khoản chuyển tiếp"
    clauses. Produce effectivity ledger entries with evidence locator
    (offset in normalized text) + review status + validity_basis.
    Cross-check against known patterns (general date, exceptions, conditional,
    time-of-act rule). Feed into validity_intervals.
  ▼ amends_extract: LLM outputs LegalLocator (citation + structured path),
    NOT internal identifiers. Server resolver maps LegalLocator → provision_uid
    via legal_document_key + article/clause/point lookup.
    Operation classification: ADD_PROVISION | REPLACE_TEXT | REPEAL | AMEND.
    Enum normalizer. Numbering accepts \d+[a-z]?.
    Failed/ambiguous → quarantine edge (logged, does NOT block document ingest).
    Evidence locator: {amending_source_document_id, source_revision_id,
    start_offset, end_offset, excerpt_hash}, verified from normalized raw.
  ▼ embed batch (vietnamese-bi-encoder + pyvi tokenize, CPU default,
    optional off-peak GPU). Record segmenter_version + model_revision.
    Skip unchanged provision_text_hash (reuse vector).
    Recompute payload/context/derived-state hash for affected nodes even if vector reused.
  ▼ build release N (STAGING):
    Create Qdrant collection legal_v{N}:
      - Carry forward unchanged provisions from N-1 (reuse vectors, update payload).
      - Embed changed/new provisions.
      - Full serving projection: all provisions + historical versions
        within temporal_support_window.
    Write Neo4j nodes/rels with release_id=N:
      - Node identity = provision_version_id.
      - Multiple versions per provision_uid allowed (historical versions retained).
      - Relationships connect versions consistently within release.
      - Materialize validity from REVIEWED AMENDS:
        REPLACE_TEXT creates successor version with new interval BEFORE closing predecessor.
        REPEAL cascades to children (Clause, Point inherit inactive).
        ADD_PROVISION creates new provision (target need not pre-exist).
        Record validity_basis per provision.
      - Tombstone = close validity interval (reviewed evidence only).
        NEVER physical-delete text/evidence due to reparse or snapshot retention.
  ▼ reconcile: compare manifest {provision_version_id set, derived_state_hash per point, count}
    against both stores. Both must match. Discrepancy → DO NOT publish. Alert. Manual intervention.
  ▼ publish (ATOMIC):
    Construct release_descriptor:
      {corpus_version: N, qdrant_collection: "legal_v{N}",
       neo4j_release_id: N, embedding_version, schema_version,
       manifest_hash, published_at, temporal_support_window}
    SET descriptor:active_release = descriptor (Redis atomic SET).
    Write active_release.json to persistent volume (atomic: write-tmp + rename).
    Load into API in-memory cache (via pub/sub or poll).
    BOTH stores now addressed via the new descriptor. Single pointer = descriptor.
  ▼ garbage_collect:
    Delete Qdrant collections + Neo4j release_id nodes for releases older than retention,
    BUT ONLY after grace period > max request/SSE deadline (in-flight requests complete).
    NEVER delete provision version history within temporal support window of active release.
  ▼ write ingestion_manifest, emit metrics.
```

### INVAR 2 (Full Projection)
Each release is a FULL serving projection: all provisions + all historical versions within the temporal support window. Delta detection only optimizes compute/vector reuse; it does NOT create a snapshot containing only changed docs.

### INVAR 3 (Single Pointer)
The release_descriptor is the SINGLE publish pointer. Qdrant collection name and Neo4j release_id are both derived from it. No independent Qdrant alias is maintained (avoids a second divergent pointer). Re-embed produces a new release through the same publish protocol.

---

## 4. Identity Model

Seven distinct identifiers. Content hash NEVER appears in stable legal keys used by AMENDS, QA labels, or citations.

| Identifier | Format | Stability | Purpose |
|---|---|---|---|
| `source_document_id` | docGUId (UUID from API) | Immutable per source artifact | Identifies downloaded document file |
| `raw_blob_hash` | SHA256(raw HTML bytes) | Per blob content | Content-addressed blob dedup |
| `source_metadata_revision_hash` | SHA256(metadata JSON) | Per metadata revision | Detects metadata-only updates |
| `normalized_text_hash` | SHA256(normalized text) | Per normalized content | Parse idempotency |
| `provision_text_hash` | SHA256(single provision text) | Per provision content | Provision version granularity (D30) |
| `legal_document_key` | Normalized docIdentity validated unique (see below) | Stable per legal act | Internal non-ambiguous key for provision UID construction |
| `provision_uid` | `{legal_document_key}::article::{n}::clause::{n}::point::{letter}` | Stable logical lineage | Machine-stable key independent of content version. Citations. AMENDS target/source. QA label reference. |
| `provision_version_id` | `{provision_uid}::rev::{provision_text_hash[:12]}` | Per provision content version | Specific content version. Embedded. Qdrant point ID = UUIDv5(NAMESPACE, provision_version_id). Neo4j node identity. |

### legal_document_key Resolution
Assigned at ingest. Derived from docIdentity when unambiguous. Collision detection: if a source artifact's docIdentity matches an existing legal_document_key but represents a distinct legal document (not a version/amendment of it), quarantine for manual resolution; do not auto-merge or auto-pick. For the normal case legal_document_key = normalized docIdentity.

Citations/legal_locator resolve via legal_document_key lookup. Ambiguous citation (multiple candidate legal_document_keys) → report ambiguity/quarantine, never auto-pick.

### Mapping Round-Trip
```
provision_version_id  →  UUIDv5(NAMESPACE, provision_version_id)  →  Qdrant point id (within per-release collection)
Qdrant payload.provision_uid  ←→  Neo4j node.provision_uid
Neo4j node.provision_version_id  →  provision_uid (property)
legal_locator  →  legal_document_key  →  provision_uid (via resolver: article/clause/point lookup)
```

NAMESPACE is a fixed project-specific UUID constant committed to code.

### INVAR 6 (Provision Version Granularity)
provision_version_id is based on provision_text_hash, NOT whole-document hash. Amending Điều 1 does NOT churn point IDs for Điều 2, 3, etc. This enables vector reuse for unaffected provisions during re-ingest.

### docIdentity Collision Handling
docIdentity can collide across different source artifacts (confirmed in NLP-LegalQA CLAUDE.md line 155 and raw data). Therefore:
- source_document_id (docGUId) is the true internal document-level key.
- docIdentity is a display/legal-citation property, NOT a primary key.
- legal_document_key provides the internal unambiguous mapping; collisions quarantined.
- provision_uid derives from legal_document_key (stable lineage).

### INVAR 5 (History Preserved)
Tombstone = close validity interval (set `to`), mark version inactive, based on reviewed evidence only. NEVER physical-delete text/evidence due to reparse success or snapshot retention=2. Physical deletion of provision version history only after temporal support window expires + grace period.

---

## 5. Temporal Model

### Parameters
| Parameter | Meaning | Default |
|---|---|---|
| `as_of_date` | Point in time we're asking about the state of the law | Today |
| `event_date` | When the factual event (violation) occurred | None (optional) |
| `reference_date` | Drives eligibility: `event_date` if provided, else `as_of_date` | Derived |
| `legal_time_basis` | Annotation: `"event"` or `"current"` | Derived |

All dates resolved once at ingress to `Asia/Ho_Chi_Minh` timezone, validated as ISO 8601. Propagated through retrieval, Cypher/templates, generation, citations, cache, SSE.

### INVAR 7 (Half-Open Intervals)
Validity intervals use half-open notation: `[from, to)`. Predicate:
```
from <= reference_date AND (to IS NULL OR to > reference_date)
```
For multiple validity_intervals per provision, Qdrant and Neo4j evaluate from/to/status on the SAME interval element (nested object conditions), not cross-filtering across intervals.

### Validity States
- `active`: In force at reference_date. Served normally.
- `inactive`: Not in force at reference_date. Excluded unless explicitly queried historically.
- `conditional`: Effectiveness depends on external condition ("có hiệu lực khi Chính phủ ban hành nghị định hướng dẫn"). NOT forced to a fake date.
- `unknown`: Insufficient information to determine.

### INVAR 8 (Conditional Not Eligible By Default)
`conditional` and `unknown` provisions WITHOUT a reviewed condition-resolution are NOT eligible by default. They are excluded from standard retrieval. System returns warning/clarification ("Quy định này có hiệu lực có điều kiện — chưa xác định được ngày áp dụng cụ thể") rather than serving them as effective law. Only provisions with reviewed resolution become eligible.

### INVAR 9 (requires_event_date Rule)
Server-side classifier/rule determines if a query requires event_date. Penalty questions ("phạt bao nhiêu", "xử phạt", "mức phạt") are classified as requiring event_date. When requires_event_date is true and event_date is missing:
1. If safe default exists (e.g., assume current law with explicit caveat), answer conditionally: "Theo luật hiện hành (NĐ 168/2024): X. Nếu hành vi xảy ra trước 01/01/2025 thì theo NĐ 100/2019: Y."
2. Otherwise, ask for clarification BEFORE retrieval.

This is an executable server-side rule, not a documentation promise. Applied before retrieval begins.

### Effectivity Extract Stage
Ingestion extracts effectivity rules from document text — "Hiệu lực thi hành" (effective date clauses) and "Điều khoản chuyển tiếp" (transitional provisions). Produces effectivity ledger entries with:
- Evidence locator (start/end offset in normalized text)
- Review status (auto/reviewed)
- validity_basis

These feed into provision validity_intervals. Cross-checks known patterns: general effective date, exception dates, conditional dependencies, time-of-act rules.

Example: NĐ 168/2024/NĐ-CP:
- General effectivity: 01/01/2025
- Exception: some provisions effective 01/01/2026
- Conditional: dependency on environmental law implementation decree
- Time-of-act rule: penalties apply based on violation date

---

## 6. Corpus Scope and Allow-List

### INVAR: No Effect Status Gate (D37)
Allow-list does NOT filter by document effect_status. NĐ 168/2024/NĐ-CP currently has status "Hết Hiệu lực một phần" — gating by effect_status would incorrectly exclude it. Scope determined by keyword/field/docGroup only.

### Versioned Allow-List
`allowlist_v1.yaml` (committed, versioned):
```yaml
version: 1
source: phapluat.gov.vn
doc_group_ids: [...]        # Seeded from NLP-LegalQA search params
field_ids: [...]            # Giao thông đường bộ field IDs
search_queries:             # Literal queries sent to upstream API
  - "giao thông đường bộ"
  - "trật tự an toàn giao thông"
  - "xử phạt vi phạm giao thông"
local_matching:             # Regex/local filters run AFTER download
  doc_name_pattern: "giao thông|trật tự.*giao thông|xử phạt.*giao thông"
  exclusion_patterns: [...] # Administrative circulars without normative content
discovery:
  pagination: rowAmount=100, iterate pageIndex
  delta_strategy: updDateTime_scan_after_capability_probe
  fallback: periodic_full_scoped_reconciliation + compare_detail_hashes
dependency_closure: true    # Ingest amending/repealing docs even if out of keyword scope
```

### Dependency Closure (D38)
Any document that amends or repeals an in-scope provision MUST be fully ingested regardless of whether its keyword/field/title matches the allow-list. Detected via AMENDS extraction from already-ingested docs and via docListOther references. Ensures amendment coverage.

### Search Strategy (D39)
- `search_queries`: literal strings sent to upstream API. No wildcards (`.*`).
- `local_matching`: regex/filters applied AFTER downloading metadata/text. More precise than API keyword search.
- `updDateTime` filter/sort: used ONLY after capability probe confirms API support. If unsupported, fall back to periodic full scoped reconciliation + compare detail metadata/content hashes.

### Appendix Handling (D24)
Default: PRESERVE appendix/table content when no reviewed exclusion rule exists. HTML normalization preserves cell/row boundaries. Exclude appendix ONLY via verifiable rule + test fixture. Never silently treat all "Phụ lục" as footer.

---

## 7. Raw Source Storage

Immutable content-addressed store. Path structure: `raw/{raw_blob_hash}/`.

Each raw entry contains:
- `original.html`: Full HTML from `/detail?tabName=noidung`
- `metadata.json`: Full JSON from `/detail?tabName=tomtat`
- `normalized.txt`: Cleaned text (BeautifulSoup + NFC)
- `manifest.json`: `{ source_document_id, raw_blob_hash, source_metadata_revision_hash, normalized_text_hash, official_url, fetched_at, parser_version, normalizer_version, evidence_offsets }`

Blob deduplication: identical raw_blob_hash → shared blob file. But manifest ALWAYS exists for each `(source_document_id, source_metadata_revision_hash)` combination, so metadata-only updates are preserved even when content blob is shared.

Re-scrape behavior: if raw_blob_hash already exists → skip blob write (already have this exact content). Always write/update manifest for this source_document_id + revision. Never overwrite existing raw files.

---

## 8. Release and Snapshot Isolation

This section defines the mechanism ensuring true snapshot isolation. It supersedes any prior description of "staging" or "corpus_version" filtering.

### Release Descriptor
An immutable record published atomically for each corpus release:
```python
@dataclass(frozen=True)
class ReleaseDescriptor:
    corpus_version: int              # Monotonic integer
    qdrant_collection: str           # e.g., "legal_v5"
    neo4j_release_id: int            # Same as corpus_version
    embedding_version: str           # Segmenter + model revision
    schema_version: str              # Graph/payload schema version
    manifest_hash: str               # SHA256 of release manifest
    published_at: datetime           # ISO 8601, Asia/Ho_Chi_Minh
    temporal_support_window_from: date  # Oldest provision version retained
```

### Storage
- Primary: Redis key `descriptor:active_release` (atomic SET).
- Durable fallback: Local file `active_release.json` on persistent volume, written atomically (write to tmp + rename) on each successful publish. Loaded at API startup. Refreshed on each publish (via poll or pub/sub notification).
- API holds in-memory copy of current descriptor for fast access.

### INVAR 1 (Request Pinning)
At ingress, API reads active release descriptor ONCE. Passes pinned descriptor to every downstream read: Qdrant (collection name), Neo4j (release_id filter), BM25, hierarchy traversal, Text2Cypher templates, cache key (descriptor fingerprint), citations, SSE meta. All reads within one request use the same pinned release.

### INVAR 2 (Full Projection)
Each release is a full serving projection: all provisions + all historical versions within temporal_support_window. Carry-forward copies unchanged provisions from N-1 to N (reuse vectors, update payload). Delta detection only optimizes which provisions need re-embedding.

### INVAR 3 (Single Pointer)
release_descriptor is the single publish pointer. Qdrant collection name and Neo4j release_id are both fields of the descriptor. No independent Qdrant alias is maintained. Re-embed produces a new release through the same publish protocol.

### INVAR 4 (Reconcile by Manifest)
Publish requires manifest match:
- Set of provision_version_ids matches between manifest and both stores.
- derived_state_hash matches per point/node.
- Count matches.

Not just count. Detects silent corruption, missing points, stale payloads.

### Rollback
Flip release descriptor back to previous version (kept in a small release history in Redis + local). Both Qdrant collection and Neo4j release_id addressed via the restored descriptor. Atomic.

### Garbage Collection
Delete Qdrant collections + Neo4j release_id nodes for releases older than retention (default retention=2: active + previous for rollback).

INVAR 32 (GC Grace): GC executes only after grace period > max request/SSE deadline (so in-flight requests pinned to old release complete). Never delete provision version history within temporal support window of the active release (history preservation, INVAR 5).

### Redis Descriptor Outage (D28)
If Redis is unavailable:
- Serve from in-memory descriptor (last loaded) + local active_release.json fallback.
- Serving continues against last pinned release. Safe.
- Cache lookups fail-open (cache miss → run pipeline). Arq ingestion unavailable (broker down).
- If NO release ever published (fresh volume, no active_release.json, no in-memory) → /readyz returns 503 `corpus_not_ready`; /v1/chat returns 503.

Redis is NOT sole source of truth for the snapshot pointer. Local file + in-memory cache provide durable verified fallback.

### /readyz Checks
Returns 200 only if ALL of:
- Active reconciled release descriptor loadable (Redis or local fallback).
- Qdrant collection for descriptor.qdrant_collection exists.
- Point count matches manifest.
- Neo4j release_id snapshot present + constraints valid.
- manifest_hash matches stored manifest.

Fresh volume not bootstrapped → 503 `corpus_not_ready`.

### Backup/Restore (D42)
One common `backup_manifest` covering: Neo4j dump, Qdrant collection snapshots (per release), raw corpus store, release manifests + active_release.json, quarantine store. Restore restores a consistent release set. Cache and arq jobs are ephemeral — NOT restored as serving state.

---

## 9. Ingestion Pipeline (Pillar 1)

Fully offline and async. Runs in separate arq-worker container. Never touches request path.

### Stages
1. **scrape**: Call phapluat.gov.vn API with versioned allow-list. NO effect_status gate. Dependency closure for amending docs. Delta detection via updDateTime + content_hash + overlap window (capability-probed). TLS verify ON. Throttle 0.5–1s. Save raw immutable (§7). Rate-limit aware.
2. **parse**: ContentParser with FIXED regex: `_RE_CLAUSE = r"^(\d+[a-z]?)\.\s+(.*)"` accepting alphanumeric clause numbers. Footer split with appendix default-preserve. Quote-block tracking. NFC normalize. Idempotent keyed by (source_document_id, source_metadata_revision_hash). Multi-value field normalization.
3. **parse-quality gate**: Check raw length ≥ minimum threshold, Điều anchor presence, hierarchy count/coverage ratio vs previous revision. If partial/truncated/degraded → keep last-good revision, log warning, do NOT tombstone existing provisions.
4. **effectivity_extract**: Extract "Hiệu lực thi hành" + "Điều khoản chuyển tiếp" clauses. Produce effectivity ledger entries with evidence locator (start/end offset in normalized text) + review status + validity_basis. Cross-check known patterns. Feed into validity_intervals.
5. **amends_extract**: LLM outputs LegalLocator format (§10). Server resolver maps LegalLocator → provision_uid via legal_document_key + article/clause/point lookup. Operation classification: ADD_PROVISION | REPLACE_TEXT | REPEAL | AMEND. Enum normalizer. Numbering accepts `\d+[a-z]?`. Failed/ambiguous → quarantine edge (logged, does NOT block document ingest). Evidence locator verified from normalized raw.
6. **embed**: Batch embed via vietnamese-bi-encoder (768d, cosine-normalized). Pre-tokenize with pyvi.ViTokenizer.tokenize. Record segmenter_version + model_revision. Skip unchanged provision_text_hash (reuse vector). Recompute payload/context/derived-state hash for all affected nodes even if vector reused. CPU default (D27). Optional off-peak GPU scheduling.
7. **build_release**: Create staging for release N. See §8 publish protocol. Full projection. Carry forward unchanged. Materialize validity from REVIEWED AMENDS. Tombstone = interval close only.
8. **reconcile**: Compare manifest against both stores (§8 INVAR 4). Mismatch → DO NOT publish. Alert.
9. **publish**: Atomic descriptor flip (§8).
10. **garbage_collect**: After grace period, delete old releases (§8 INVAR 32).
11. **report**: Emit metrics.

### Error Handling (Ingestion)
| Failure | Behavior |
|---|---|
| Scrape HTTP 200 + non-null `error` | Upstream error, retry. NOT end-of-catalog. |
| Scrape `docs[]` empty + `error==null` | End-of-catalog, stop pagination. |
| Parse fail on one doc | Skip doc, log, continue batch. |
| Parse-quality gate fails | Keep last-good revision. Log warning. Do NOT tombstone. |
| AMENDS extraction invalid schema | Quarantine that edge (logged). Document ingest continues. |
| AMENDS ADD_PROVISION target-not-found | Valid — new provision being added. Create target. |
| AMENDS target resolution fails for REPEAL/REPLACE | Quarantine edge. Document ingest continues. |
| Embed/import fail | Retry with backoff. After N failures mark failed, alert. |
| Reconcile mismatch | DO NOT publish. Alert. Manual intervention. |

### Re-Ingest Special Case
Nghị định 238/2026/NĐ-CP amending Nghị định 168/2024/NĐ-CP, effective 2026-08-15. On/after that date, re-ingest picks up the amendment. The materialized validity updates affected provisions' eligibility windows. Queries with event_date before 2026-08-15 correctly serve NĐ 168 pre-amendment provisions; queries with event_date after serve amended versions.

### Secrets Policy
All credentials loaded from environment variables. `.env` gitignored. Tests use placeholder credentials. Documentation uses `<placeholder>` syntax.

---

## 10. AMENDS Edge Model

### LLM Output Contract
LLM extracts amendments into LegalLocator format, NOT internal identifiers:
```json
{
  "operation": "ADD_PROVISION | REPLACE_TEXT | REPEAL | AMEND",
  "amending_locator": {
    "citation": "khoản 2 Điều 52 Nghị định 168/2024/NĐ-CP",
    "doc_identity": "168/2024/NĐ-CP",
    "article": "52",
    "clause": "2",
    "point": null
  },
  "target_locator": {
    "citation": "khoản 6 điểm d Điều 28 Nghị định 100/2019/NĐ-CP",
    "doc_identity": "100/2019/NĐ-CP",
    "article": "28",
    "clause": "6",
    "point": "d"
  },
  "effective_date": "2025-01-01",
  "confidence": 0.85,
  "evidence_span": "2. Sửa đổi, bổ sung điểm d khoản 6 Điều 28 như sau: ..."
}
```

Server resolver maps LegalLocator → provision_uid via legal_document_key + article/clause/point lookup. Resolution failure for ADD_PROVISION = valid (new provision). Resolution failure for REPEAL/REPLACE = quarantine.

### Operation Semantics
| Operation | Meaning | Target Required? | Cascade | Eligibility Change |
|---|---|---|---|---|
| ADD_PROVISION | Creates new provision | No (target doesn't exist yet) | N/A | Reviewed only |
| REPLACE_TEXT | Replaces content of existing provision | Yes | N/A | Reviewed only; creates successor version BEFORE closing predecessor |
| REPEAL | Bãi bỏ provision entirely | Yes | Cascades to children (Clause→Point) | Reviewed only |
| AMEND | Generic when specific op unclear | Yes | Reviewer refines | Never until refined |

Numbering: `\d+[a-z]?` accepted throughout parser, schema, resolver.

### Edge States
| State | Meaning | Eligibility Impact |
|---|---|---|
| `quarantined` | Invalid/unresolved/ambiguous extraction | None. Logged for manual review. |
| `auto` | Locator resolved but not reviewed | Annotation only. Context decorated with "[CÓ THỂ đã được sửa đổi bởi ... — chờ xác minh]". Does NOT change eligibility. |
| `reviewed` | Human or corroborated-evidence verified | Changes eligibility. Materializes validity. Sets validity_basis. |

### INVAR 10 (AMENDS States)
Only `reviewed` edges change eligibility. `auto` edges are annotation-only. `quarantined` edges are logged. This prevents LLM hallucination from automatically repealing legal provisions.

### ADD_PROVISION Schema
Does NOT require old target to exist. Schema includes:
- `new_provision_locator`: LegalLocator for the new provision
- `parent_path`: Parent provision_uid + position/order
- `verbatim_successor_text` OR `source_span`: Deterministically materializable content
- `evidence_locator`: Verified from normalized raw

### REPLACE_TEXT Materialization
Creates successor provision version with new text and new validity interval BEFORE closing predecessor interval. Both versions coexist with adjacent intervals. Old version remains retrievable for historical event_dates.

### Evidence Locator (Audit Trail)
Immutable evidence locator per edge:
```python
@dataclass(frozen=True)
class EvidenceLocator:
    amending_source_document_id: str
    source_revision_id: str
    start_offset: int       # Character offset in normalized text
    end_offset: int
    excerpt_hash: str       # SHA256 of extracted excerpt
```
Verified from normalized raw before review/apply. Tamper-evident.

### Real Example: NĐ 168/2024/NĐ-CP
- Bổ sung khoản 2a Điều 28 → ADD_PROVISION, numbering `2a`, target didn't exist before
- Bãi bỏ Điều 5–11 → REPEAL cascade: Article 5–11 + all their Clauses/Points marked inactive
- Hiệu lực có điều kiện → some provisions marked `conditional`, not forced to fake date
- Điều khoản chuyển tiếp → transitional provisions with specific applicability windows

---

## 11. Embeddings + Indexing (Pillar 2)

Batch-first for documents. Query embedding happens online at request time (required for ANN). Metadata filtering is the first gate; vector search is the fallback.

### Document Embedding Job (offline, arq stage 6)
- Model: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base-v2, 768-dim, cosine-normalized). Exact HuggingFace revision pinned in manifest.
- Mandatory: `pyvi.ViTokenizer.tokenize` applied to every text before embedding, both index and query. Segmenter + model revision recorded in manifest.
- Batch size 32–64 texts. CPU default (D27). Optional off-peak GPU scheduling.
- Idempotent: skip provisions with unchanged provision_text_hash (reuse vector).
- Recompute payload/context/derived-state hash for all affected nodes even if vector reused.
- Documents embedded in batch offline. Queries embedded online at request time (single query, low latency).

### Qdrant Per-Release Collection Layout
```
Collection: "legal_v{N}" (one per release, named in release_descriptor)
  point id   = UUIDv5(NAMESPACE, provision_version_id)
  vector     = embedding 768d (HNSW, cosine)
  payload    = {
     provision_uid,              // stable legal key (no hash)
     provision_version_id,       // versioned key (has provision_text_hash)
     source_document_id,         // docGUId
     legal_document_key,         // internal unambiguous doc key
     doc_identity,               // legal citation (display)
     doc_type,
     validity_intervals[],       // [{from, to, status}] nested objects
     issue_date, upd_datetime,
     field[], organ[],           // multi-value arrays
     label (Article/Clause/Point),
     number, parent_article, parent_clause,
     corpus_version,
     embedding_version,
     context_text,               // immutable context-ready text snapshot
     derived_state_hash,         // hash of all payload fields except vector
     
     // Citation projection (§12/D43):
     title,                      // Provision/article title
     official_url,               // vanban.chinhphu.vn or equivalent
     human_citation,             // Rendered legal_locator
     provision_path,             // ["Điều 6", "Khoản 3", "Điểm a"]
     applicability_basis,        // Selected validity status for display
     snippet                     // Short context excerpt
  }
```
Payload indexes enabled on filter fields. Nested indexes on validity_intervals for correct interval evaluation. `context_text` + citation projection fields enable degraded-but-functional rendering when Neo4j is unavailable.

### INVAR 7 (Eligibility Predicate)
Every Qdrant query includes the eligibility predicate checking `validity_intervals` against `reference_date`:
```
ANY interval WHERE:
  interval.from <= reference_date
  AND (interval.to IS NULL OR interval.to > reference_date)
  AND interval.status IN ('active')   // conditional/unknown excluded by default (D40)
```
Applied identically to Neo4j BM25 (WHERE clause with same logic). Both filtered BEFORE RRF fusion. Nested interval evaluation ensures from/to/status checked on the same interval element.

### BM25 Keyword — Baseline Hypothesis (D11)
BM25 lives in Neo4j fulltext (Lucene). This is a **baseline hypothesis**, not a claimed-superior configuration. The evaluation harness (§16) validates whether hybrid (dense+BM25 RRF) outperforms dense-only on this specific corpus. Configuration chosen through quality gate, not assumed.

App-layer RRF fuses dense (Qdrant) + keyword (Neo4j fulltext). Same eligibility predicate applied to both before fusion.

### Neo4j Role
Pure graph (hierarchy traversal, AMENDS, metadata) + Lucene fulltext keyword + provision-level validity storage + Text2Cypher template execution (read-only principal). No dense vector index.

---

## 12. Retrieval + Generation (Pillar 3)

Tight request path. Dual cap: max 8 serving contexts AND token budget. Early-exit wherever possible.

### Key Design Decisions
1. **Dual cap**: Maximum 8 serving contexts AND token budget (default 4096 tokens, configurable). Expansion (sibling Points, Article children) counts toward the token budget. If expansion would exceed budget, truncate. Evaluation retrieval_k and serving context_k are separate parameters.
2. **Decompose contract**: Decomposer prompt decides sub-query count. Code enforces `max_subqueries=6` and deduplicates by normalized query string. Prompt alone insufficient; code contract authoritative.
3. **Eligibility predicate mandatory**: Every Qdrant and Neo4j BM25 query carries the eligibility predicate with `reference_date` (pinned release). No raw unfiltered search ever.
4. **Early-exit patterns**:
   - Cache hit (either layer) → skip remaining stages. Routing + rewriting already executed for correct cache key computation.
   - Intent = reject/greeting → skip retrieval entirely.
   - Empty retrieval result → return "không tìm thấy", skip gen.
5. **Context source**: Primary = Qdrant payload `context_text` (immutable snapshot from ingest time, in pinned release collection). Fallback = Neo4j hierarchy traversal (if Qdrant payload missing or stale, filtered by pinned release_id). Ensures degraded-but-functional retrieval when Neo4j is down.
6. **Validity decoration**: Context text cached raw (without date-dependent labels). Validity labels (`[ĐÃ BÃI BỎ]`, `[SỬA ĐỔI]`, `[HIỆU LỰC CÓ ĐIỀU KIỆN]`) decorated per-request based on `reference_date` against provision validity_intervals. This prevents date-dependent cache invalidation.
7. **Citation projection**: When Neo4j is down, sources[] rendered from Qdrant payload citation projection fields (title, official_url, human_citation, provision_path, validity_intervals, applicability_basis, snippet). Degraded but functional.

### Execution Model
FastAPI is async but Neo4j/Qdrant drivers + reranker are blocking:
- Blocking DB calls: bounded thread pool executor (configurable max_workers).
- GPU reranker: concurrency within API process (semaphore, single-process). 
- Per-stage timeout/deadline. Total request deadline.
- Overload: 429 (rate limited) or 503 with `Retry-After` header when executors saturated.
- OOM protection: VRAM monitoring, graceful degradation to CPU if GPU OOM.
- Cancellation: propagate cancellation through stages; release resources on client disconnect.

### Generation
Uses `_QA_FEW_SHOT_SYSTEM_PROMPT` from NLP-LegalQA. Retry generation ONLY before first token streamed. Once any token has been sent to client via SSE, no retry (would produce garbled/duplicate output). After exhaustion pre-token, return friendly error message.

If LLM provider doesn't support streaming/cancel: use buffered mode — generate full answer server-side then emit as exactly ONE `token` event with full text. Declare buffered mode in SSE meta event. Never fake token-by-token streaming from buffered output.

---

## 13. API + SSE Contract

### POST /v1/chat

Request:
```typescript
{
  query: string              // max 2000 chars
  chat_history?: Array<{role: "user"|"assistant", content: string}>  // max 20 turns, each content max 4000 chars
  as_of_date?: string        // ISO 8601 date, default today, Asia/Ho_Chi_Minh
  event_date?: string        // ISO 8601 date, optional
  filters?: {                // optional metadata filters
    doc_type?: string[]
    field?: string[]
    organ?: string[]
  }
}
```

Browser/client CANNOT control: `top_k`, `provider`, `fetch_k`, `rerank_top`, `labels`. These are server configuration, not request parameters.

Response: SSE stream (`Content-Type: text/event-stream`). Client reads via `fetch()` + `ReadableStream` (native `EventSource` only supports GET, incompatible with POST JSON body).

### SSE Event Sequence
ALL valid intents follow the same sequence: `meta` (first, always) → `sources` (optional) → `token*` (zero or more) → exactly one terminal: `done` OR `error`.

```
event: meta
data: {"state": "started", "trace_id": "...", "corpus_version": ..., "qdrant_collection": "...", "neo4j_release_id": ..., "as_of_date": "...", "event_date": "...", "reference_date": "...", "legal_time_basis": "event|current", "streaming_mode": "streaming|buffered"}

event: sources
data: [{"citation": "Điểm a Khoản 3 Điều 6 NĐ 168/2024/NĐ-CP", "title": "...", "official_url": "...", "provision_path": ["Điều 6", "Khoản 3", "Điểm a"], "validity_window": {"from": "2025-01-01", "to": null}, "applicability_basis": "active", "snippet": "..."}]

event: token
data: {"text": "..."}

event: done
data: {"trace_id": "...", "outcome": "full|degraded|cached", "degraded_components": [], "warnings": []}

event: error
data: {"code": "...", "message": "...", "trace_id": "..."}
```

Notes:
- `meta` contains ONLY immutable request/release fields and `state:"started"`. No outcome/warnings/degraded_components (those belong in done/error).
- `sources[]` items expose human citation, title, official_url, provision_path, validity_window, applicability_basis, snippet. Internal identifiers (source_document_id, raw_blob_hash) NOT exposed.
- `done` carries final outcome, warnings, degraded_components.
- Client disconnect: server cancels generation + logs. Does NOT emit `done` after disconnect (client gone, can't receive).
- `streaming_mode`: "streaming" if provider supports streaming, "buffered" if not. Buffered mode emits exactly one `token` event with full text.
- Validation errors / 429 / 503 BEFORE opening stream → HTTP problem response (standard HTTP status code + JSON body). After stream opens → SSE `error` event.

### POST /v1/feedback
```typescript
{
  trace_id: string
  rating: "helpful" | "not_helpful"
  comment?: string          // max 1000 chars
  idempotency_key: string   // Client-generated UUID, prevents double-submission
}
```
Validates trace ownership (must reference a real trace from this deployment). Returns 200 on success, 409 if idempotency_key already submitted.

No admin HTTP cache-invalidation endpoint in v1. Cache invalidation via CLI/local-only command only.

### Health Endpoints
- `GET /livez`: Process alive. No dependencies checked. Returns 200 immediately.
- `GET /readyz`: Checks active reconciled release (descriptor loadable), Qdrant collection exists + count matches manifest, Neo4j release_id snapshot + constraints, manifest_hash matches. Fresh volume not bootstrapped → 503 `corpus_not_ready`. NO remote LLM call.
- `GET /metrics`: Prometheus exposition format. Infrastructure metrics only.

Health probes never call remote LLM (would burn quota + slow probes).

---

## 14. Caching (Pillar 4)

Three-layer Redis cache in `cache:` namespace. Keys include full dependency fingerprint. Two distinct lookup points in request flow.

### Cache Lookup Ordering
1. **Answer cache**: After route + rewrite (key depends on rewritten query + intent).
2. **Retrieval cache**: After decompose, before search/rerank.

Both lookups occur AFTER release descriptor is pinned and dates resolved.

### Cache Key Design
Keys capture everything affecting the cached value:
- **Answer cache**: `cache:ans:{sha256(rewritten_query + intent + filter_params + as_of_date + event_date + reference_date + legal_time_basis + release_descriptor.fingerprint + embedding_version + reranker_version + model_name + prompt_version + policy_version)}`
- **Retrieval cache**: `cache:ret:{sha256(decomposed_queries_sorted + filter_params + reference_date + release_descriptor.fingerprint + embedding_version + reranker_version)}`
- **Context cache**: `cache:ctx:{provision_uid}:{corpus_version}` (raw text only, no date-dependent labels)
- **Amends cache**: `cache:amends:{provision_uid}:{corpus_version}`

Normalization for hashing: case-fold + whitespace collapse + unicode NFC normalize. NO stopword stripping (can change meaning).

### Layer 1: Answer Cache
- Value: full answer + sources + intent.
- TTL: long (e.g., 24h). Invalidated by release_descriptor.fingerprint in key.
- HIT: skip retrieval + generation. Routing + rewriting already ran. New trace created (cache_origin_trace_id recorded internally).

### Layer 2: Retrieval Cache
- Value: top_k chunk references (provision_uid + label + score).
- TTL: medium (e.g., 6h).
- HIT: skip search + rerank. Still run generation.

### Layer 3: Context/Amends Cache
- Raw context text (no validity labels). Decorated per-request.
- Invalidated by corpus_version in key.

### Invalidation Strategy
Version-based. Release descriptor fingerprint embedded in keys means stale entries naturally miss on release change. Old entries expire via TTL. Negative caching avoided or given very short TTL.

### Feedback Interaction
User 👎 does not immediately invalidate cache. Logged to Langfuse. Queries accumulating multiple 👎 flagged for review; manual cache invalidation via CLI only (no HTTP endpoint in v1).

### Cache Metrics (Prometheus)
`cache_hit_ratio` per layer, `cache_miss_latency` vs `cache_hit_latency`. High-cardinality fields (query text, UID, trace_id) NOT used as Prometheus label values — use enum/bucket labels only.

### Redis Cache Outage vs Descriptor Outage
- Cache (cache: namespace) failure → fail-open, skip cache, run pipeline. Optimization loss, not correctness failure.
- Descriptor (descriptor: namespace) failure → durable local fallback (§8). Different semantics.
- Broker (arq: namespace) failure → ingestion stops. Serving unaffected.

---

## 15. Graph Schema + Validity

### Node Labels (Hierarchy)
```
Document → Part → Chapter → Section → Article → Clause → Point
```
Levels optional in real documents. Parser attaches to nearest present ancestor; never invents placeholder levels.

Metadata nodes: DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field. Fields multi-value.

### Identity (see §4)
Neo4j node identity = provision_version_id (unique per content version). Properties: provision_uid (lineage), legal_document_key, release_id, validity_intervals, etc.

Multiple versions per provision_uid allowed within a release (historical versions retained for temporal queries). Relationships connect versions consistently within a release.

### Relationships
```
Hierarchy : HAS_PART / HAS_CHAPTER / HAS_SECTION / HAS_ARTICLE / HAS_CLAUSE / HAS_POINT
Metadata  : BELONGS_TO_GROUP / HAS_TYPE / HAS_STATUS / ISSUED_BY / SIGNED_BY / IN_FIELD
Cross-doc : RELATED_TO  (from API docListOther, target is ExternalDocument stub if outside corpus)
          : AMENDS {operation, evidence_locator, effective_date,
                    extractor_version, prompt_version, confidence, review_status, validity_basis}
```

### External References (D25)
`docListOther` entries whose legal_document_key is NOT in the corpus → `ExternalDocument` node keyed by source_document_id (docGUID), not doc_identity. Lightweight stub for citation only. NOT retrievable (no embedding), NOT validity-bearing.

### Constraints
Uniqueness constraint on provision_version_id for all hierarchy nodes. Composite uniqueness for metadata nodes. All MERGE-based, idempotent.

---

## 16. Monitoring + Evaluation (Pillar 5)

### A. Evaluation Harness (Offline)
Dataset: frozen versioned dataset (`eval_dataset_v1.json`). Each label:
```json
{
  "question": "...",
  "legal_locators": ["Điều 6 Khoản 3 Điểm a NĐ 168/2024/NĐ-CP"],
  "as_of_date": "2026-08-14",
  "event_date": "2026-08-14",
  "resolved_provision_uids": ["168/2024/NĐ-CP::article::6::clause::3::point::a"],
  "reference_answer": "...",
  "notes": "..."
}
```
Labels resolved against specific corpus release (not legacy UID/docIdentity). Held-out 80/20 split stratified by legal_document_key. Versioned — new corpus release may require label re-resolution.

**Retrieval quality**: Recall@8, Precision@k, MRR, nDCG. Eval retrieval_k separate from serving context_k. Token budget assertion (generated context ≤ budget). Run whenever embedding model, reranker, filter logic, or decompose prompt changes.

**Reproducibility**: Router/rewrite/decompose are provider-neutral LLM calls; temperature=0 alone is insufficient for reproducibility across providers. CI replays versioned query-plan fixtures (recorded canonical decomposed sub-queries) OR records canonical outputs per run. Run fingerprint (corpus_version + model revisions + prompt version + timestamp) enforced. Resume of eval run must not mix outputs from different fingerprints (fingerprint check on resume).

**AMENDS-aware relevance**: Successor UIDs (post-amendment) marked as acceptable references for queries with event_date after amendment effective date. Answer citing successor provision is correct, not wrong.

**Answer quality**: LLM-as-judge scores 5 criteria. ROUGE/BERTScore supplementary. LLM-judge runs nightly/manual (expensive). Lexical/recall runs per-PR.

**Quality gate**: Recall@8 must not drop more than 2% vs established baseline (absolute threshold). nDCG must not drop more than 3%. Gates promotion/deploy, NOT every PR. Baseline artifact pinned. Split membership fixed. Threshold semantics absolute (not relative to noisy runs). Promotion rule documented.

**Output**: JSON report + pushed to Langfuse dataset/experiment.

### B. Online Observability

**Langfuse (self-hosted)**: Deployed per official compose profile. Trace every request with spans. Attach feedback score. Include retrieval UIDs + scores.

**Prometheus + Grafana**: Latency p50/p95/p99 per-stage. Cache hit-rate per layer. Qdrant/Neo4j health. Ingestion metrics. Request rate by intent. Alert rules. Telemetry redaction/retention policy. NO high-cardinality labels.

### C. Data Drift Detection
Compare ingestion_manifest across runs. Flag unexpected drift. On model/segmenter change: offline eval BEFORE swapping production index. Deploy only if quality gate passes.

### D. Feedback Loop
```
user 👎 → Langfuse → identify problematic queries
  → review → fix (prompt/filter/data/AMENDS review) → re-evaluate offline
  → quality gate passes → deploy → monitor 👎 decrease
```

---

## 17. Error Handling

Philosophy: fail gracefully, never crash request path. Honest degradation, not false claims.

### Request Path (Online)
| Component failure | Behavior |
|---|---|
| Release descriptor unavailable (Redis + local fallback both fail) | 503. Cannot determine active release. |
| Router LLM | Fallback to intent `retrieve` |
| Rewriter | Use original query, skip rewrite |
| Decomposer | Fallback to `[{"query": original}]` |
| Qdrant unavailable | Return degraded error. Cannot claim functional retrieval without ANN. |
| Neo4j unavailable | Retrieval uses Qdrant payload context_text + citation projection (degraded but functional). BM25/expansion/Text2Cypher unavailable. Tag response as degraded in SSE done event. |
| Both stores unavailable | Friendly "system temporarily unavailable" message. |
| Reranker | Use ANN ordering, skip rerank. |
| Generator LLM (pre-token) | Retry once with exponential backoff. After exhaustion: friendly error. |
| Generator LLM (mid-stream) | NO retry. Stream truncated. Emit error event. |
| Redis cache (cache: namespace) | Fail-open, skip cache, run pipeline. |
| Redis descriptor (descriptor: namespace) | Durable local fallback (§8). |
| Langfuse/Prometheus | Fail-open. Log locally. |

### Ingestion Path (Offline)
See §9 error table. Key: AMENDS extraction failure quarantines edge, does not block document ingest. Parse-quality gate prevents data loss from bad scrape. Reconcile failure blocks publish.

### Cross-Cutting
- Structured JSON logging with `trace_id` spanning components.
- Timeouts on all LLM calls and DB queries. No hung requests.
- Bounded executor for blocking calls. GPU semaphore within API process. Deadline propagation.
- OOM: VRAM monitoring, graceful degradation to CPU.
- Cancellation: propagate through stages; release resources on client disconnect.

---

## 18. Testing Strategy

### Unit Tests (No Network, No Services)
- Parser hierarchy: regex matching including alphanumeric clause numbers (2a, 18a). Footer split. Quote-block tracking. NFC normalization. Optional-level handling. Fixture-based.
- Identity: legal_document_key derivation, provision_uid building, provision_version_id from provision_text_hash, UUIDv5 determinism, round-trip mapping.
- AMENDS schema validation: valid/invalid JSON, missing fields, unresolvable targets → quarantine path. Enum normalizer correctness. ADD_PROVISION with non-existent target = valid. Evidence locator verification.
- Cache key construction: verify all dependency dimensions present (including release_descriptor.fingerprint, dates, legal_time_basis). Same inputs → same key. Version bump → key change.
- Eligibility predicate: various validity_intervals combinations, reference_date boundary cases, half-open [from,to) semantics, conditional/unknown excluded by default, nested interval evaluation (same-element).
- RRF aggregation.
- Heuristic rerank: recency calculation, post-eligibility application.
- Query normalization for cache (no stopwords).
- Text2Cypher TemplateCall: valid param extraction, invalid query → route to retrieval. Strict discriminator.
- Subquery dedup + max_subqueries enforcement.
- Token budget enforcement in context building.
- SSE event serialization.
- requires_event_date classifier.

### Integration Tests (Single Docker-Compose Test Profile)
- Import parsed JSON → verify node/relationship counts + constraints hold.
- Batch embed → verify Qdrant point count + payload filter works + context_text populated + citation projection populated.
- End-to-end retrieval on fixture document: query → expected top_k provision_uids match reference.
- Dual-write reconciliation: simulate partial failure → verify no release published.
- Tombstone: REVIEWED REPEAL → validity interval closed, text/evidence preserved.
- Cache invalidation: publish new release → verify old keys miss.
- Atomic publish: verify serving sees consistent state after flip.
- Partial parse: verify last-good kept, no tombstone.
- AMENDS cascade: REVIEWED REPEAL of Article → Clauses/Points inactive.
- Eligibility predicate: event_date before/after amendment → different results.
- Staging N doesn't affect serving N-1 (snapshot isolation).
- Unchanged corpus carry-forward (no-op cron doesn't bump version).
- Rollback: flip back to previous release → serving reverts correctly.
- Same-day replacement boundary ([from, to) half-open semantics).
- Multi-interval predicate (nested evaluation, no cross-filtering).
- NĐ 168 exceptions (general 2025-01-01, exception 2026-01-01, conditional, time-of-act).
- ADD_PROVISION successor creation.
- REPLACE_TEXT successor-before-predecessor-close.
- Metadata-only revision (different source_metadata_revision_hash, same raw_blob_hash).
- Degraded Qdrant citations (Neo4j down → sources[] from payload).
- Redis descriptor outage → durable fallback serves last pinned release.
- Fresh volume bootstrap/restore → /readyz returns 503 corpus_not_ready until bootstrap complete.
- Two-container GPU contention (API rerank + worker embed concurrent → worker CPU, no contention).

Never touch production data. Isolated test database/collection. Placeholder credentials only. Demo/live-service scripts NOT in unit suite.

### Acceptance Tests (Concrete Scenarios)
| Scenario | Expected |
|---|---|
| Before/after amendment | Same legal_locator with event_date before/after amendment effective → different applicable provision/penalty |
| Event_date transition | NĐ 100 (pre-2025) vs NĐ 168 (post-2025) for same violation → correct provision selected by event_date |
| Amendment adds 2a | ADD_PROVISION with numbering `2a` → retrievable, cited correctly |
| Repeal cascades to children | REVIEWED REPEAL of Article → its Clauses/Points inactive, validity_basis = cascade |
| Partial-ingest not exposed | Truncated/bad fetch → last-good kept, serving path unaffected, no tombstone |
| Stale payload after amendment | Amendment changes validity → payload/context hash recomputed, vector reused if text same |
| Truncated parser no tombstone | Parse-quality gate fails → last-good revision retained |
| SSE disconnect | Client disconnects → server cancels, no done emitted, resources freed |
| Cache-hit fresh trace | Cache hit → new trace_id, cache_origin_trace_id recorded internally |
| Fresh-volume bootstrap/restore | Fresh volumes → bootstrap from backup or re-ingest produces consistent serving |
| Staging isolation | During ingest of release N, serving continues on N-1 with no visible changes |
| Rollback | Flip back to N-1 → all reads revert to N-1 collection/release_id |
| Same-day replacement boundary | Event_date = effective_date → [from,to) selects NEW provision (half-open) |
| Multi-interval provision | Provision with two disjoint active intervals → predicate evaluates each interval independently |
| NĐ 168 conditional provision | conditional status → excluded from standard retrieval, warning returned |
| Degraded citations | Neo4j down → sources[] rendered from Qdrant payload citation projection |
| Redis descriptor outage | Serve last pinned release from local fallback; no 500 |
| Metadata-only revision | New source_metadata_revision_hash, same raw_blob_hash → manifest updated, blob not re-written |
| Two-container GPU | API rerank on GPU + worker embed on CPU → no contention, no OOM |

### Evaluation
See §16A. Recall@8, nDCG, token-budget assertion, bounded retries/deadline, run fingerprint, replay fixtures.

### CI
- `pytest` unit tests on every PR (no network).
- Integration tests on parser/importer/retrieval/security/identity/release changes (require docker-compose test profile).
- Acceptance tests on release candidates.
- Eval (lexical/recall) on retrieval-change PRs. LLM-judge nightly/manual.
- Security tests (template injection, credential leaks) on every PR.
- Quality gate on deploy promotion.

---

## 19. Delivery and Repository Layout

### Repository Structure
Root repo (`Agentic`) vendors `NLP-LegalQA` as a **git submodule pinned at a fixed commit SHA**. Read-only reference — no modifications to vendored code.

Delivery acceptance criteria:
- `.gitmodules` file committed with submodule entry.
- Gitlink (SHA) committed in root tree.
- CI checks out with `--recurse-submodules`.
- CI asserts pinned SHA matches expected.

```
Agentic/
├── .gitmodules                          # Submodule definition
├── NLP-LegalQA/                         # Git submodule at pinned SHA
├── src/                                 # Application source
│   ├── legal_rag/                       # Main package
│   │   ├── identity.py                  # Identity model (§4)
│   │   ├── temporal.py                  # Temporal model (§5)
│   │   ├── release.py                   # Release descriptor + publish protocol (§8)
│   │   ├── ingestion/                   # Ingestion pipeline (§9)
│   │   ├── amends/                      # AMENDS extraction + resolver (§10)
│   │   ├── embeddings/                  # Embedding jobs (§11)
│   │   ├── retrieval/                   # Retrieval pipeline (§12)
│   │   ├── templates/                   # Text2Cypher read templates (§12)
│   │   ├── cache/                       # Cache layers (§14)
│   │   ├── api/                         # FastAPI endpoints (§13)
│   │   ├── sse/                         # SSE contract (§13)
│   │   └── config/                      # Allow-list, prompts, constants
│   └── ...
├── frontend/                            # Next.js React app
├── tests/
│   ├── unit/                            # No-network unit tests
│   ├── integration/                     # Docker-compose test profile
│   └── acceptance/                      # Concrete scenario tests
├── eval/                                # Eval harness + datasets + fixtures
├── scripts/                             # CLI tools (cache invalidation, bootstrap, backup)
├── docker-compose.yml                   # All services
├── docker-compose.test.yml              # Test profile
├── Dockerfile                           # App image (shared API + worker entrypoints)
├── uv.lock                              # Python dependency lockfile
├── package-lock.json                    # Node dependency lockfile
└── docs/superpowers/specs/              # Design specs
```

### Pinning
Everything pinned for reproducibility:
- Python version (`.python-version`)
- Node version (`.nvmrc`)
- Base Docker image tags (exact digest)
- All Docker service image tags (Neo4j, Qdrant, Redis, Langfuse, Prometheus, Grafana)
- HuggingFace model revisions (exact commit hashes for vietnamese-bi-encoder + Vietnamese_Reranker)
- NLP-LegalQA submodule SHA

All fingerprints recorded in corpus manifest + eval manifest + release descriptor. An eval run is reproducible only if all fingerprints match.

### Volumes, Bootstrap, Backup (D42)
| Volume | Content | Backup Strategy |
|---|---|---|
| `neo4j_data` | Graph + constraints | neo4j-admin dump/restore |
| `qdrant_storage` | Per-release collections | Qdrant snapshot API per collection |
| `redis_data` | Cache + arq queue + descriptor | Redis RDB/AOF backup |
| `raw_corpus` | Immutable content-addressed raw store | File-level backup |
| `manifest_store` | Ingestion manifests + release manifests + active_release.json | File-level backup |
| `quarantine_store` | Quarantined AMENDS edges | File-level backup |

One common `backup_manifest` covering all volumes. Restore restores a consistent release set. Cache and arq jobs are ephemeral — NOT restored as serving state.

Bootstrap procedure: fresh volumes → restore from backup OR full re-ingest from scratch. Documented step-by-step.

### GPU Profile (D27)
Two containers (API + arq-worker). Python semaphore does NOT serialize GPU across processes.

**Default profile**: API reranker on GPU. Ingestion embedder on CPU. No contention. Slower ingestion but safe and simple.

**Optional off-peak GPU**: Worker scheduled during low API load periods with GPU access. Requires operational discipline (no concurrent heavy API load during ingestion).

OOM protection: VRAM monitoring in API. Graceful degradation to CPU if GPU OOM detected. Timeout/cancellation propagated through stages.

---

## 20. Open Items (Environment/Data Dependent Only)

These are NOT design gaps — they are data discovery or environment setup tasks that happen during implementation. None affect spec correctness.

1. **Exact allow-list IDs**: docGroup/field IDs for traffic law need live API discovery. Seeded from NLP-LegalQA's search params, refined via facet exploration during first ingestion run.
2. **Langfuse resource budget**: Official self-host compose recommends minimum resources. Actual allocation depends on host machine. Documented sizing guide provided; user confirms hardware.
3. **HuggingFace model download**: First-time download of vietnamese-bi-encoder + Vietnamese_Reranker weights. Requires internet access during initial setup. Cached in volume for subsequent runs.
4. **UpdDateTime capability probe result**: Whether phapluat.gov.vn API supports updDateTime filtering/sorting. Probed at first ingestion run; fallback strategy (periodic full reconciliation) activated if unsupported.

All design decisions affecting correctness are finalized in this spec.
