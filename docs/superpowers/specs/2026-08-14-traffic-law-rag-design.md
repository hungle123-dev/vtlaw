# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-15  
**Status:** Draft v5 (post-Codex round-4 revision — Neo4j snapshot isolation, temporal factual inputs, stored validity vs per-request applicability, discriminated AMENDS, provenance separation)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (vendored as git submodule at pinned SHA, read-only reference). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Code and infrastructure are production-shape; deployable to cloud if desired.
- Right-sized for a bounded corpus (Vietnamese traffic-law documents, thousands of chunks). No fake scale claims.
- Legally correct validity model with explicit temporal semantics (`reference_date`, factual inputs).
- **True snapshot isolation**: every serving request pins exactly one immutable release descriptor; no half-visible state during ingest. Active + previous releases coexist safely.
- **Legal history preserved**: tombstone closes validity intervals only; never physical-deletes history due to reparse success or snapshot retention.
- **Executable correctness**: AMENDS as discriminated union schemas; generator prompt versioned and temporally grounded; cache write only after terminal success.

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters. Out of scope for this corpus.
- Fine-tuning pipeline (QLoRA/Unsloth). Eval harness supports evaluating fine-tuned models but training is not shipped.
- Public deployment or multi-tenant auth in v1. Single-user demo.
- OCR. Source API returns structured HTML/JSON, not scanned PDFs.
- LLM-based legal validation of amendments. Pydantic validates structure, not legal meaning.
- Free-form Cypher from LLM in v1. Text2Cypher uses server-owned parameterized templates only.

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
| D15 | Identity scheme | Eight-tier identity (§4); content hash NOT in stable legal keys; snapshot_node_id for Neo4j physical key | docIdentity collides; multiple releases coexist |
| D16 | Qdrant point ID | UUIDv5(POINT_NAMESPACE, provision_version_id) per-release collection | Snapshot isolation via separate collections |
| D17 | Temporal model | Half-open `[from, to)`; stored states {in_force, conditional, unknown}; factual inputs {violation_state, occurred_at, detected_date}; effectivity ledger with precedence | Interval correctness; conditional not applicable by default; facts required for sanction/transitional |
| D18 | Text2Cypher v1 | Enum TemplateCall + typed params + server-owned parameterized templates; hard LIMIT/timeout/result schema/citation mapping; no free-form Cypher | Eliminates injection risk |
| D19 | SSE contract | meta(state:started) → sources? → token* → done\|error. Buffered mode = one token event. HTTP problem before stream open. Clarify intent = one deterministic token then done. Cache write only after terminal done success. | Explicit protocol; consistent across intents |
| D20 | Secrets policy | Env vars only, `.env` gitignored, test placeholders, docs `<placeholder>` | Credential safety |
| D21 | Release/snapshot isolation | Immutable release_descriptor per publish; Qdrant per-release collection; Neo4j snapshot_node_id per (release_id, provision_version_id); single pointer (descriptor); durable local fallback | True snapshot isolation; active+previous coexist safely |
| D22 | AMENDS operations | Discriminated union: AddProvisionOp / ReplaceTextOp / RepealOp / AmendOp; LegalLocator output; auto = annotation only; reviewed = eligibility change; evidence locator verified against defined IDs | Safe executable semantics |
| D23 | Clause numbering | Parser accepts alphanumeric `\d+[a-z]?` (e.g., `2a`, `18a`) | Raw data contains alphanumeric clauses; legacy parser misses them |
| D24 | Appendix handling | Default preserve when no reviewed rule; exclude only via verifiable rule + test fixture | Phụ lục can contain binding legal rules |
| D25 | External references | ExternalDocument stubs keyed by source_document_id (docGUID), not doc_identity | Unambiguous internal key |
| D26 | Health endpoints | /livez (no deps), /readyz (checks active reconciled release), /metrics. No remote LLM call in probes | Fast cheap probes |
| D27 | GPU profile | API reranker GPU, ingestion embedder CPU default (optional off-peak GPU) | Two containers; Python semaphore doesn't cross processes |
| D28 | Redis roles | Three namespaces: arq:, cache:, descriptor:. Descriptor has durable local fallback. Broker outage ≠ snapshot failure. | Clear separation of concerns |
| D29 | Historical versions | Each release contains ALL provision versions within temporal support window; snapshot_node_id per (release, provision_version); tombstone = interval close only | Legal history preserved; multiple versions per lineage coexist across releases |
| D30 | Provision version granularity | provision_version_id based on provision_text_hash (not whole-doc hash) | Amending Điều 1 doesn't churn other provisions' point IDs |
| D31 | Reconcile manifest | Publish requires manifest match: node set, relationship digest/count, fulltext-index readiness, derived_state_hash per point. Not just count. | Detects silent corruption, relationship loss, stale indexes |
| D32 | GC grace | Delete old release snapshots only after grace > max request/SSE deadline. Never delete provision version history within temporal support window. | In-flight requests complete safely |
| D33 | Delivery | Git submodule at pinned SHA; commit .gitmodules + gitlink; CI checkout recursive + assert SHA | Reproducible vendoring |
| D34 | Cache lookups | Two distinct points: (1) answer cache after route/rewrite; (2) retrieval cache after decompose, before search/rerank. Write only after terminal success. | Correct ordering; no partial-failure cache pollution |
| D35 | Context cache | Store raw context text (no date-dependent labels); decorate validity per-request based on reference_date | Date-independent caching |
| D36 | Retry generation | Only before first token streamed; once streaming started, no retry | Post-stream retry produces garbled/duplicate output |
| D37 | Effect status not scope gate | Allow-list does NOT filter by document effect_status | NĐ 168 = "Hết Hiệu lực một phần"; gating excludes it incorrectly |
| D38 | Dependency closure | Documents amending/repealing in-scope provisions ingested fully regardless of keyword/field match | Amendment coverage |
| D39 | Search strategy | Literal search_queries sent upstream; regex/local matching post-download; no wildcard API keywords; updDateTime used only after capability probe | Respect API capabilities |
| D40 | Conditional eligibility | conditional/unknown NOT applicable by default without reviewed resolution; warning/clarification instead | Prevents serving unverified conditional provisions as effective law |
| D41 | Factual inputs | violation_state (completed/ongoing/unknown) + occurred_at + detected_date; missing facts → deterministic clarification BEFORE retrieval | Sanction/transitional questions require facts; no specific penalty X/Y before retrieval |
| D42 | Backup manifest | One common backup_manifest covering Neo4j dump, Qdrant collections, raw corpus, manifests, quarantine. Cache/arq jobs ephemeral, NOT restored as serving state. | Consistent restore |
| D43 | Citation projection | Qdrant payload includes title, official_url, human_citation, provision_path, validity_intervals (immutable rule states), snippet. SSE sources carry selected interval/basis computed fresh per request. | Neo4j-down still renders sources[]; applicability is per-request |
| D44 | Stored validity vs applicability | Stored: in_force/conditional/unknown (immutable rule state). Applicability (active/inactive at reference_date) computed fresh per request. | Clean separation; no static applicability_basis |
| D45 | Future-effective ingestion | Ingest promulgated-but-not-yet-effective documents; future intervals self-handle boundary. Don't wait for effective date. | Pre-loading; boundary handled by eligibility predicate |
| D46 | Coverage unsupported response | reference_date outside verified temporal_support_window/coverage → historical_coverage_unsupported (distinct from "not found") | Honest signal |
| D47 | Generator prompt wrapper | Versioned prompt wrapper receiving reference_date, legal_time_basis, selected intervals/basis, immutable citations; answer ONLY from supplied eligible contexts. NOT legacy prompt verbatim. | Temporally grounded; no "current date" bias |
| D48 | Neo4j principals | API read principal (read-only). Worker ingestion principal (write/schema/publish). Separate credentials. | Least privilege |
| D49 | Provenance separation | HTML blob by raw_blob_hash; metadata by source_metadata_revision_hash; normalized artifact by normalized_text_hash + normalizer_version; per-source-revision manifest immutable. Evidence offsets bind to specific normalized artifact. | Supports metadata-only update + normalizer revision without collision |
| D50 | Retrieval policy fingerprint | retrieval_policy_fingerprint = canonical hash of RRF/BM25/rerank/fetch_k/top_k/heuristic/code-schema knobs. Included in retrieval cache key. | Cache invalidation on config change |
| D51 | Bounded admission | Executor has bounded admission queue (not unbounded ThreadPoolExecutor queue). On saturation → 429/503 + Retry-After. Streaming accumulator bounded. | Backpressure; OOM protection |
| D52 | Review log artifact | Versioned review-log/override artifact in release manifest. Records human-reviewed AMENDS decisions. | Audit trail for legality changes |
| D53 | Serving corpus owner | src/legal_rag/ingestion parser/adapter owns serving corpus guarantees (alphanumeric clause, appendix). Submodule NLP-LegalQA is read-only reference only. | Legacy parser lacks these guarantees |
| D54 | ReleaseDescriptor.fingerprint | Canonical deterministic derivation: SHA256(canonical JSON of descriptor fields). Used consistently in cache/eval/pinning. | Deterministic cache keys |

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
│  ├─ api             : FastAPI — validate→trace→dates+facts           │
│  │                   →pin release→route→rewrite→cache(1)            │
│  │                   →decompose→cache(2)→retrieve→rerank(GPU)       │
│  │                   →gen(prompt wrapper)                            │
│  │                   Endpoints: POST /v1/chat, POST /v1/feedback,    │
│  │                   GET /livez, GET /readyz, GET /metrics           │
│  ├─ arq-worker      : ingestion offline (scrape→parse→effectivity    │
│  │                   →amends→embed(CPU)→import→reconcile→publish)   │
│  │                   Same codebase, different entrypoint             │
│  │                   Uses write/schema/publish Neo4j principal       │
│                                                                      │
│  DATA STORES                                                         │
│  ├─ neo4j           : graph hierarchy + AMENDS + Lucene BM25         │
│  │                   Nodes keyed by snapshot_node_id                │
│  │                   Filtered by release_id per snapshot            │
│  │                   API read-only principal                        │
│  ├─ qdrant          : Per-release collections (legal_v{N})          │
│  │                   HNSW dense + payload metadata + context        │
│  │                   Full citation projection                       │
│  └─ redis           : arq broker (arq:*)                            │
│                       cache (cache:*)                               │
│                       active_release_descriptor (descriptor:*)       │
│                       Config: AOF ON, noeviction                    │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : LLM trace, feedback, eval experiment          │
│  ├─ prometheus      : infra metrics, scrapes GET /metrics           │
│  └─ grafana         : dashboard                                     │
└──────────────────────────────────────────────────────────────────────┘
```

Langfuse deployed per https://langfuse.com/self-hosting/ with pinned versions, persistent volumes, secrets via env. Resource budget documented in deployment guide.

### INVAR 1 (Snapshot Isolation)
Every serving request pins exactly one release_descriptor at ingress. All reads — Qdrant collection, Neo4j release_id filter, BM25, hierarchy traversal, Text2Cypher templates, cache lookup, citations, SSE meta — use that pinned descriptor. No read crosses releases. Active + previous releases coexist safely in both stores.

### Request Path (Hot Path, Pillar 3)
```
POST /v1/chat {query, chat_history?, as_of_date?, event_date?,
               violation_state?, occurred_at?, detected_date?, filters?}
  │
  ▼ validate input (schema, lengths, ISO dates, timezone Asia/Ho_Chi_Minh,
    filter cardinality limits, history length limits)
  ▼ create fresh trace (new trace_id for EVERY request, including cache hits)
  ▼ PIN RELEASE DESCRIPTOR (read once, pass to all downstream reads)
     If unavailable: serve last pinned from local fallback; if none: 503
  ▼ resolve dates + facts:
     reference_date = derived from violation_state + occurred_at/detected_date/as_of_date
     legal_time_basis = "event" | "detected" | "current"
  ▼ requires_clarification check (§5):
     If sanction/transitional question lacks required facts
       → clarify intent: emit one deterministic token then done. BEFORE retrieval.
     If reference_date outside verified coverage
       → historical_coverage_unsupported response. BEFORE retrieval.
  ▼ route (LLM, temp=0, max_tokens=64)
     intent ∈ {greeting, cypher_query, retrieve, reject, clarify}
  │
  ├─ reject → template refusal, write trace, return done
  ├─ greeting → deterministic response (no LLM, no retrieval), return done
  ├─ clarify → one deterministic clarification token, return done
  ├─ cypher_query → rewrite if history → TemplateCall selection (LLM)
  │   → execute parameterized template (pinned release + validity predicate
  │     + LIMIT + timeout + result schema + citation mapping)
  │   → gen natural language → accumulate → on done success: write cache + trace
  │   → stream
  │
  └─ retrieve:
       ▼ rewrite (multi-turn; skip if no history)
       ▼ CACHE LOOKUP (1) — ANSWER CACHE (after route+rewrite):
          key = sha256(rewritten_query + intent + filters_canonical
              + as_of_date + event_date + reference_date + legal_time_basis
              + violation_state + occurred_at + detected_date
              + release_descriptor.fingerprint
              + embedding_version + reranker_version
              + model_name + prompt_wrapper_version + policy_version)
          HIT → new trace (cache_origin_trace_id recorded), skip everything below
          MISS → continue
       ▼ decompose (LLM, JSON sub-queries, append original;
          code enforces max_subqueries=6 + dedup by normalized string)
       ▼ CACHE LOOKUP (2) — RETRIEVAL CACHE (after decompose, before search):
          key = sha256(decomposed_queries_sorted + filters_canonical + reference_date
              + violation_state
              + release_descriptor.fingerprint
              + embedding_version + reranker_version
              + retrieval_policy_fingerprint)
          HIT → skip search+rerank, continue to context build
          MISS → search+rerank below
       ▼ multi_search (sub-queries parallel, PINNED release collection/release_id):
          Each sub-query:
            • Qdrant: eligibility predicate [from,to) against reference_date
              (nested interval evaluation on stored in_force state)
              + metadata filter → HNSW within eligible set in pinned collection
            • Neo4j: BM25 fulltext with SAME eligibility predicate
              filtered by pinned release_id + validity BEFORE final LIMIT
          Fuse RRF per sub-query → aggregate across sub-queries
       ▼ fetch context from Qdrant payload context_text (primary)
          or Neo4j hierarchy traversal (fallback, pinned release_id)
       ▼ expand context (optional: sibling Points, Article children;
          apply same eligibility predicate; expand + dedupe;
          final-select ≤8 contexts AND token budget)
       ▼ compute per-request applicability: for each result, select the
          interval whose [from,to) covers reference_date with state=in_force;
          derive applicability_label (applicable/not_applicable/conditional_warning)
       ▼ cross-encoder rerank (GPU, FP16, batch_size small)
       ▼ heuristic rerank (recency bonus AFTER eligibility, eval-gated)
       ▼ final-select top serving contexts (max 8 AND token budget 4096)
       ▼ build context_str with per-request applicability labels
          (NOT cached — computed fresh). Tag [ĐÃ BÃI BỎ]/[SỬA ĐỔI]
          per reference_date. Amends note. Auto-review annotations.
       ▼ gen (versioned prompt wrapper: reference_date + legal_time_basis
          + selected intervals/basis + immutable citations + eligible contexts;
          answer ONLY from supplied eligible contexts)
          Retry ONLY before first token. Once streaming starts, no retry.
       ▼ tee tokens to client + bounded accumulator
       ▼ on terminal done SUCCESS: write caches (answer + retrieval)
          NOT written on pre-token failure, mid-stream error, disconnect
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
    Save raw immutable with provenance separation (§7).
    Ingest promulgated-but-not-yet-effective documents (D45);
    future intervals self-handle boundary.
  ▼ parse hierarchy using src/legal_rag/ingestion parser/adapter (owner
    of serving corpus guarantees: alphanumeric clause \d+[a-z]?, appendix
    default-preserve). Submodule NLP-LegalQA is read-only reference only (D53).
    Footer split with configurable appendix decision (default preserve).
    Quote-block tracking. NFC normalize. Idempotent keyed by
    (source_document_id, source_metadata_revision_hash).
    Multi-value field normalization.
  ▼ parse-quality gate: raw length sanity, anchor presence (Điều markers),
    hierarchy count/coverage ratio vs previous revision.
    If partial/bad → keep last-good revision, do NOT tombstone.
  ▼ effectivity_extract: extract "Hiệu lực thi hành" + "Điều khoản chuyển tiếp"
    clauses into effectivity ledger entries (§5). Reviewed entries override
    baseline effectDate per precedence. expireDate = outer cap.
  ▼ amends_extract: LLM outputs LegalLocator format (§10). Server resolver
    maps LegalLocator → provision_uid via legal_document_key + article/clause/point lookup.
    Discriminated union operation classification:
    AddProvisionOp | ReplaceTextOp | RepealOp | AmendOp.
    REPLACE requires base_provision_version_id or base_text_hash + verbatim
    successor OR deterministic exact patch verified on base hash.
    ADD requires new locator + parent/order + text or deterministic source span.
    Patch ("replace/delete phrase") only materializes if deterministic on exact
    base hash; else reviewer provides full successor or edge is annotation/quarantined.
    Enum normalizer. Numbering accepts \d+[a-z]?.
    Failed/ambiguous → quarantine edge (logged, does NOT block document ingest).
    EvidenceLocator uses defined IDs (§10). Verified from normalized artifact.
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
      - Node identity = snapshot_node_id (per release + provision_version).
      - Multiple versions per provision_uid allowed (historical versions retained).
      - Relationships connect nodes within same release only.
      - Materialize validity from REVIEWED AMENDS:
        REPLACE creates successor version with new interval BEFORE closing predecessor.
        REPEAL cascades to children (Clause, Point inherit closed intervals).
        ADD creates new provision (target need not pre-exist).
        Record validity_basis per provision.
      - Tombstone = close validity interval (reviewed evidence only).
        NEVER physical-delete text/evidence due to reparse or snapshot retention.
    Build/update fulltext indexes scoped to release_id.
  ▼ reconcile: compare manifest against both stores:
      - Node set match (provision_version_id set per release).
      - Relationship digest/count match.
      - Derived_state_hash per point match.
      - Fulltext-index readiness verified.
    Any mismatch → DO NOT publish. Alert. Manual intervention.
  ▼ publish (ATOMIC):
    Construct release_descriptor with canonical fingerprint (D54):
      {corpus_version: N, qdrant_collection: "legal_v{N}",
       neo4j_release_id: N, embedding_version, schema_version,
       manifest_hash, published_at, temporal_support_window_from,
       fingerprint: SHA256(canonical JSON of above)}
    SET descriptor:active_release = descriptor (Redis atomic SET).
    Write active_release.json to persistent volume (atomic: write-tmp + rename).
    Load into API in-memory cache (via pub/sub or poll).
    BOTH stores now addressed via the new descriptor. Single pointer = descriptor.
  ▼ garbage_collect:
    Delete Qdrant collections + Neo4j release_id nodes for releases older than retention,
    BUT ONLY after grace period > max request/SSE deadline.
    NEVER delete provision version history within temporal support window of active release.
  ▼ write ingestion_manifest + review_log artifact, emit metrics.
```

### INVAR 2 (Full Projection)
Each release is a FULL serving projection: all provisions + all historical versions within the temporal support window. Delta detection only optimizes compute/vector reuse; it does NOT create a snapshot containing only changed docs.

### INVAR 3 (Single Pointer)
The release_descriptor is the SINGLE publish pointer. Qdrant collection name and Neo4j release_id are both fields of the descriptor. No independent Qdrant alias is maintained. Re-embed produces a new release through the same publish protocol.

---

## 4. Identity Model

Eight distinct identifiers. Content hash NEVER appears in stable legal keys used by AMENDS, QA labels, or citations.

| Identifier | Format | Stability | Purpose |
|---|---|---|---|
| `source_document_id` | docGUId (UUID from API) | Immutable per source artifact | Identifies downloaded document file |
| `raw_blob_hash` | SHA256(raw HTML bytes) | Per blob content | Content-addressed blob dedup |
| `source_metadata_revision_hash` | SHA256(metadata JSON) | Per metadata revision | Detects metadata-only updates |
| `normalized_text_hash` | SHA256(normalized text) | Per normalized content | Parse idempotency; evidence offset binding |
| `provision_text_hash` | SHA256(single provision text) | Per provision content | Provision version granularity (D30) |
| `legal_document_key` | Normalized docIdentity validated unique (§4.1) | Stable per legal act | Internal non-ambiguous key for provision UID construction |
| `provision_uid` | `{legal_document_key}::article::{n}::clause::{n}::point::{letter}` | Stable logical lineage | Machine-stable key independent of content version. Citations. AMENDS target/source. QA label reference. |
| `provision_version_id` | `{provision_uid}::rev::{provision_text_hash[:12]}` | Per provision content version | Canonical content ID. Property in Neo4j. Used to derive snapshot_node_id. Qdrant point ID within per-release collection. |
| `snapshot_node_id` | UUIDv5(RELEASE_NAMESPACE, `{release_id}:{provision_version_id}`) | Per (release, provision_version) | Neo4j physical node key. Unique per release-version pair. Enables active+previous coexistence. |

### 4.1 legal_document_key Resolution
Assigned at ingest. Derived from docIdentity when unambiguous. Collision detection: if a source artifact's docIdentity matches an existing legal_document_key but represents a distinct legal document (not a version/amendment of it), quarantine for manual resolution; do not auto-merge or auto-pick. For the normal case legal_document_key = normalized docIdentity.

Citations/legal_locator resolve via legal_document_key lookup. Ambiguous citation (multiple candidate legal_document_keys) → report ambiguity/quarantine, never auto-pick.

### Mapping Round-Trip
```
Qdrant (within collection legal_v{N}):
  point id = UUIDv5(POINT_NAMESPACE, provision_version_id)
  payload.provision_uid → legal lineage
  payload.provision_version_id → canonical content ID

Neo4j (within release_id=N):
  node.snapshot_node_id = UUIDv5(RELEASE_NAMESPACE, "{N}:{provision_version_id}")
  node.provision_version_id = canonical content ID (property)
  node.provision_uid = legal lineage (property)

Cross-store:
  Qdrant payload.provision_version_id ←→ Neo4j node.provision_version_id (same value)
  Both grouped by provision_uid for lineage queries

Resolution:
  legal_locator → legal_document_key → provision_uid (via resolver: article/clause/point lookup)
  provision_uid + provision_text_hash → provision_version_id
```

POINT_NAMESPACE and RELEASE_NAMESPACE are fixed project-specific UUID constants committed to code (different values to avoid cross-domain collisions).

### INVAR 6 (Provision Version Granularity)
provision_version_id is based on provision_text_hash, NOT whole-document hash. Amending Điều 1 does NOT churn point IDs for Điều 2, 3, etc. This enables vector reuse for unaffected provisions during re-ingest.

### docIdentity Collision Handling
docIdentity can collide across different source artifacts (confirmed in NLP-LegalQA CLAUDE.md line 155 and raw data). Therefore:
- source_document_id (docGUId) is the true internal document-level key.
- docIdentity is a display/legal-citation property, NOT a primary key.
- legal_document_key provides the internal unambiguous mapping; collisions quarantined.
- provision_uid derives from legal_document_key (stable lineage).

### INVAR 5 (History Preserved)
Tombstone = close validity interval (set `to`), mark interval state unchanged (still in_force, just bounded), based on reviewed evidence only. NEVER physical-delete text/evidence due to reparse success or snapshot retention. Physical deletion of provision version history only after temporal support window expires + grace period.

### Neo4j Coexistence
Active release N and previous release N-1 coexist in Neo4j. Nodes are distinguished by snapshot_node_id (which encodes release_id). Constraints are on snapshot_node_id (unique globally). Queries always filter by release_id. Relationships connect nodes within the same release only.

---

## 5. Temporal Model

### Factual Inputs
Sanction and transitional questions require more than a single event_date. Minimal factual inputs:
```typescript
{
  violation_state?: "completed" | "ongoing" | "unknown",
  occurred_at?: string,      // ISO date: when violation/event happened
  detected_date?: string,    // ISO date: when ongoing violation was detected
  as_of_date?: string        // ISO date: query context date (default today)
}
```

**Reference date derivation:**
- completed violation: reference_date = occurred_at (time-of-act rule)
- ongoing violation: reference_date = detected_date if provided, else as_of_date
- unknown/no facts: clarification required before retrieval

All dates resolved once at ingress to `Asia/Ho_Chi_Minh` timezone, validated as ISO 8601. Propagated through retrieval, Cypher/templates, generation, citations, cache, SSE.

### INVAR 41 (Clarification Before Retrieval)
Server-side classifier determines if a query requires factual inputs (sanction/transitional questions). When required facts are missing:
1. Route to `clarify` intent.
2. Emit ONE deterministic clarification token (e.g., "Để trả lời chính xác mức phạt, vui lòng cho biết: (1) Hành vi xảy ra khi nào? (2) Hành vi đã kết thúc hay đang tiếp diễn?").
3. Return done. NO retrieval performed.

No specific penalty X/Y is given before retrieval. If answering multiple branches (e.g., "before/after amendment"), MUST perform retrieval + citations for EACH branch.

### INVAR 46 (Coverage Unsupported)
If reference_date is outside the verified temporal_support_window/coverage of the active release → respond with `historical_coverage_unsupported` (distinct from "không tìm thấy"). E.g., "Dữ liệu pháp luật hiện có chỉ bao phủ từ năm 2019 đến nay. Vui lòng kiểm tra nguồn khác cho giai đoạn trước đó."

### INVAR 7 (Half-Open Intervals)
Validity intervals use half-open notation: `[from, to)`. Predicate:
```
from <= reference_date AND (to IS NULL OR to > reference_date)
```
For multiple validity_intervals per provision, Qdrant and Neo4j evaluate from/to/state on the SAME interval element (nested object conditions), not cross-filtering across intervals.

### Stored Validity States (Immutable Rule State)
Stored in validity_intervals as immutable rule state:
- `in_force`: The provision is in force during this interval (subject to reference_date check).
- `conditional`: Effectiveness depends on external condition. NOT forced to a fake date.
- `unknown`: Insufficient information to determine applicability.

These are IMMUTABLE properties of the provision version, independent of any particular reference_date.

### Per-Request Applicability (Computed Fresh)
Applicability is computed fresh per request based on reference_date + factual context:
- An interval with state=`in_force` where `[from, to)` covers reference_date → applicable.
- An interval with state=`conditional` covering reference_date → conditional_warning (not applicable by default unless reviewed resolution exists).
- An interval with state=`unknown` covering reference_date → unknown_warning (not applicable by default).
- No interval covering reference_date → not_applicable.

Applicability labels are NOT stored statically. They are computed per request and carried in SSE sources[].

### INVAR 8 (Conditional Not Applicable By Default)
`conditional` and `unknown` intervals WITHOUT a reviewed condition-resolution are NOT applicable by default. System returns warning/clarification rather than serving them as effective law. Only intervals with reviewed resolution become applicable.

### INVAR 9 (requires_event_date/facts Rule)
Server-side classifier determines if a query requires factual inputs. Penalty questions ("phạt bao nhiêu", "xử phạt", "mức phạt") and transitional questions are classified as requiring facts. Missing facts → clarify intent (see INVAR 41). This is an executable server-side rule, applied before retrieval begins.

### Effectivity Ledger
Ingestion extracts effectivity rules from document text — "Hiệu lực thi hành" (effective date clauses) and "Điều khoản chuyển tiếp" (transitional provisions). Produces a reviewed effectivity ledger with entries:
```python
@dataclass(frozen=True)
class EffectivityLedgerEntry:
    scope_locator: str           # provision_uid or document-level
    rule_kind: str               # general_effectivity | exception | transitional | conditional_dependency | time_of_act
    precedence: int              # Higher = takes priority
    condition: Optional[str]     # For conditional rules
    evidence_locator: EvidenceLocator
    validity_basis: str          # e.g., "effectivity_ledger_entry_123_reviewed"
    review_status: str           # auto | reviewed
```

**Precedence rules:**
- Reviewed provision exception/transitional rule > general effectivity
- Reviewed general effectivity > source metadata effectDate
- Source metadata effectDate = baseline
- Source metadata expireDate = outer cap (all intervals capped)

Reviewed entries override baseline per precedence. Auto entries are annotation only.

### Future-Effective Ingestion (D45)
Promulgated-but-not-yet-effective documents are ingested immediately. Their validity_intervals have `from` in the future. The eligibility predicate naturally excludes them for reference_dates before `from`. On/after the effective date, they become applicable automatically. No special "wait for effective date" logic needed.

---

## 6. Corpus Scope and Allow-List

### INVAR: No Effect Status Gate (D37)
Allow-list does NOT filter by document effect_status. NĐ 168/2024/NĐ-CP currently has status "Hết Hiệu lực một phần" — gating by effect_status would incorrectly exclude it. Scope determined by keyword/field/docGroup only.

### Versioned Allow-List
`allowlist_v1.yaml` (committed, versioned):
```yaml
version: 1
source: phapluat.gov.vn
doc_group_ids: [...]
field_ids: [...]
search_queries:             # Literal queries sent to upstream API
  - "giao thông đường bộ"
  - "trật tự an toàn giao thông"
  - "xử phạt vi phạm giao thông"
local_matching:             # Regex/local filters run AFTER download
  doc_name_pattern: "giao thông|trật tự.*giao thông|xử phạt.*giao thông"
  exclusion_patterns: [...]
discovery:
  pagination: rowAmount=100, iterate pageIndex
  delta_strategy: updDateTime_scan_after_capability_probe
  fallback: periodic_full_scoped_reconciliation + compare_detail_hashes
dependency_closure: true
```

### Dependency Closure (D38)
Any document that amends or repeals an in-scope provision MUST be fully ingested regardless of whether its keyword/field/title matches the allow-list. Detected via AMENDS extraction from already-ingested docs and via docListOther references.

### Search Strategy (D39)
- `search_queries`: literal strings sent to upstream API. No wildcards.
- `local_matching`: regex/filters applied AFTER downloading metadata/text.
- `updDateTime` filter/sort: used ONLY after capability probe confirms API support. If unsupported, fall back to periodic full scoped reconciliation + compare detail metadata/content hashes.

### Appendix Handling (D24)
Default: PRESERVE appendix/table content when no reviewed exclusion rule exists. HTML normalization preserves cell/row boundaries. Exclude appendix ONLY via verifiable rule + test fixture.

---

## 7. Raw Provenance

Immutable provenance with separation supporting metadata-only updates and normalizer revisions (D49).

### Store Layout
```
raw/
  blobs/{raw_blob_hash}/original.html              # HTML blob, deduped by content
  metadata/{source_metadata_revision_hash}/metadata.json  # Metadata blob/revision
  normalized/{normalized_text_hash}/{normalizer_version}/normalized.txt  # Normalized artifact
  manifests/{source_document_id}/{source_metadata_revision_hash}/manifest.json  # Per-source-revision manifest, immutable
```

### Manifest Contents
```json
{
  "source_document_id": "...",
  "raw_blob_hash": "...",
  "source_metadata_revision_hash": "...",
  "normalized_text_hash": "...",
  "normalizer_version": "...",
  "official_url": "...",
  "fetched_at": "...",
  "parser_version": "...",
  "evidence_offsets": {"start": N, "end": M}  // Bind to specific normalized artifact
}
```

Evidence offsets bind to the specific normalized artifact identified by `(normalized_text_hash, normalizer_version)`. A new normalizer version produces a new normalized artifact path, so offsets remain valid for their specific artifact.

Metadata-only updates: new source_metadata_revision_hash → new metadata blob + new manifest, but same raw_blob_hash → HTML blob reused. Normalizer revision: new normalized_text_hash + normalizer_version → new normalized artifact + new manifest.

### Owner of Serving Corpus Guarantees
The parser/adapter in `src/legal_rag/ingestion` is the OWNER of serving corpus guarantees (alphanumeric clause numbering `\d+[a-z]?`, appendix default-preserve, effectivity ledger extraction). The submodule `NLP-LegalQA` is a READ-ONLY REFERENCE only — its legacy parser (`_RE_CLAUSE = (\d+)`) does NOT satisfy these guarantees and is NOT used for production parsing (D53).

---

## 8. Release and Snapshot Isolation

This section defines the mechanism ensuring true snapshot isolation with active + previous coexistence.

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
    fingerprint: str                 # SHA256(canonical JSON of above fields excl. fingerprint)
```

Fingerprint derivation (D54): canonical JSON serialization of all fields except `fingerprint` itself, sorted keys, no whitespace. SHA256 of result. Deterministic and reproducible.

### Storage
- Primary: Redis key `descriptor:active_release` (atomic SET).
- Durable fallback: Local file `active_release.json` on persistent volume, written atomically (write to tmp + rename) on each successful publish. Loaded at API startup. Refreshed on each publish.
- API holds in-memory copy of current descriptor for fast access.

### INVAR 1 (Request Pinning)
At ingress, API reads active release descriptor ONCE. Passes pinned descriptor to every downstream read: Qdrant (collection name), Neo4j (release_id filter), BM25, hierarchy traversal, Text2Cypher templates, cache key (descriptor.fingerprint), citations, SSE meta. All reads within one request use the same pinned release.

### INVAR 2 (Full Projection)
Each release is a full serving projection: all provisions + all historical versions within temporal_support_window. Carry-forward copies unchanged provisions from N-1 to N (reuse vectors, update payload). Delta detection only optimizes which provisions need re-embedding.

### INVAR 3 (Single Pointer)
release_descriptor is the single publish pointer. Qdrant collection name and Neo4j release_id are both fields of the descriptor. No independent Qdrant alias is maintained. Re-embed produces a new release through the same publish protocol.

### INVAR 4 (Reconcile by Manifest)
Publish requires manifest match:
- Node set match: provision_version_id set per release.
- Relationship digest/count match (hierarchy + AMENDS relationships).
- Derived_state_hash per point match.
- Fulltext-index readiness verified (BM25 indexes built and scoped to release_id).

Not just count. Detects silent corruption, relationship loss, stale indexes.

### Rollback
Flip release descriptor back to previous version (kept in a small release history in Redis + local). Both Qdrant collection and Neo4j release_id addressed via the restored descriptor. Atomic.

### Garbage Collection
Delete Qdrant collections + Neo4j release_id nodes for releases older than retention (default retention=2: active + previous for rollback).

INVAR 32 (GC Grace): GC executes only after grace period > max request/SSE deadline. Never delete provision version history within temporal support window of the active release.

### Redis Descriptor Outage (D28)
If Redis is unavailable:
- Serve from in-memory descriptor (last loaded) + local active_release.json fallback.
- Serving continues against last pinned release. Safe.
- Cache lookups fail-open (cache miss → run pipeline). Arq ingestion unavailable.
- If NO release ever published (fresh volume, no active_release.json, no in-memory) → /readyz returns 503 `corpus_not_ready`; /v1/chat returns 503.

Redis is NOT sole source of truth for the snapshot pointer. Local file + in-memory cache provide durable verified fallback.

### /readyz Checks
Returns 200 only if ALL of:
- Active reconciled release descriptor loadable.
- Qdrant collection for descriptor.qdrant_collection exists + point count matches manifest.
- Neo4j release_id snapshot present + constraints valid.
- Fulltext indexes ready and scoped to active release_id.
- manifest_hash matches stored manifest.

Fresh volume not bootstrapped → 503 `corpus_not_ready`.

### Backup/Restore (D42)
One common `backup_manifest` covering: Neo4j dump, Qdrant collection snapshots (per release), raw corpus store, release manifests + active_release.json, quarantine store, review_log artifact. Restore restores a consistent release set. Cache and arq jobs are ephemeral — NOT restored as serving state.

---

## 9. Ingestion Pipeline (Pillar 1)

Fully offline and async. Runs in separate arq-worker container. Uses write/schema/publish Neo4j principal (D48). Never touches request path.

### Stages
1. **scrape**: Call phapluat.gov.vn API with versioned allow-list. NO effect_status gate. Dependency closure. Delta detection via updDateTime + content_hash + overlap window (capability-probed). TLS verify ON. Throttle 0.5–1s. Save raw immutable with provenance separation (§7). Ingest promulgated-but-not-yet-effective documents (D45). Rate-limit aware.
2. **parse**: Use `src/legal_rag/ingestion` parser/adapter (owner of serving corpus guarantees, D53). Regex accepts alphanumeric clause numbers `\d+[a-z]?`. Footer split with appendix default-preserve. Quote-block tracking. NFC normalize. Idempotent keyed by (source_document_id, source_metadata_revision_hash). Multi-value field normalization.
3. **parse-quality gate**: Check raw length ≥ minimum threshold, Điều anchor presence, hierarchy count/coverage ratio vs previous revision. If partial/truncated/degraded → keep last-good revision, log warning, do NOT tombstone existing provisions.
4. **effectivity_extract**: Extract "Hiệu lực thi hành" + "Điều khoản chuyển tiếp" clauses into effectivity ledger entries (§5). Evidence locator binds to specific normalized artifact. Feed into validity_intervals per precedence rules.
5. **amends_extract**: LLM outputs LegalLocator format (§10). Discriminated union operation. EvidenceLocator uses defined IDs. Quarantine failures. Does NOT block document ingest.
6. **embed**: Batch embed via vietnamese-bi-encoder. Pre-tokenize with pyvi. Record segmenter_version + model_revision. Skip unchanged provision_text_hash. Recompute payload/context/derived-state hash for affected nodes. CPU default (D27).
7. **build_release**: Create staging for release N. See §8. Full projection. Carry forward unchanged. Materialize validity from REVIEWED AMENDS. Tombstone = interval close only. Build/update fulltext indexes scoped to release_id.
8. **reconcile**: Compare manifest against both stores (§8 INVAR 4). Include relationship digest/count + fulltext-index readiness. Mismatch → DO NOT publish. Alert.
9. **publish**: Atomic descriptor flip (§8). Include review_log artifact in manifest.
10. **garbage_collect**: After grace period, delete old releases (§8 INVAR 32).
11. **report**: Emit metrics.

### Error Handling (Ingestion)
| Failure | Behavior |
|---|---|
| Scrape HTTP 200 + non-null `error` | Upstream error, retry. NOT end-of-catalog. |
| Scrape `docs[]` empty + `error==null` | End-of-catalog, stop pagination. |
| Parse fail on one doc | Skip doc, log, continue batch. |
| Parse-quality gate fails | Keep last-good revision. Do NOT tombstone. |
| AMENDS extraction invalid schema | Quarantine edge. Document ingest continues. |
| AMENDS ADD_PROVISION target-not-found | Valid. Create target. |
| AMENDS REPLACE without base hash/text | Quarantine edge. |
| AMENDS patch not deterministic on base hash | Quarantine or require full successor. |
| AMENDS target resolution fails for REPEAL | Quarantine edge. Document ingest continues. |
| Embed/import fail | Retry with backoff. After N failures mark failed, alert. |
| Reconcile mismatch (node/rel/index/hash) | DO NOT publish. Alert. Manual intervention. |

### Secrets Policy
All credentials loaded from environment variables. `.env` gitignored. Tests use placeholder credentials. Documentation uses `<placeholder>` syntax. Worker uses write/schema/publish Neo4j principal; API uses read-only principal (D48).

---

## 10. AMENDS Edge Model

### Discriminated Union Schemas (D22)
AMENDS operations are a discriminated union with operation-specific schemas:

```python
class AddProvisionOp(BaseModel):
    operation: Literal["ADD_PROVISION"]
    amending_locator: LegalLocator
    new_provision_locator: LegalLocator
    parent_locator: LegalLocator
    order: int
    successor_text: Optional[str]         # Verbatim text
    source_span: Optional[EvidenceLocator] # Or deterministic source span
    # At least one of successor_text or source_span required
    
class ReplaceTextOp(BaseModel):
    operation: Literal["REPLACE_TEXT"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    base_provision_version_id: Optional[str]  # Required: one of these two
    base_text_hash: Optional[str]             # Required: one of these two
    successor_text: Optional[str]             # Verbatim successor
    exact_patch: Optional[PatchSpec]          # Or deterministic patch
    # Either successor_text OR exact_patch required
    
class RepealOp(BaseModel):
    operation: Literal["REPEAL"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    cascade: bool = True                      # Cascade to children
    
class AmendOp(BaseModel):
    operation: Literal["AMEND"]               # Generic, no eligibility change until refined
    amending_locator: LegalLocator
    target_locator: LegalLocator
    
class PatchSpec(BaseModel):
    kind: Literal["replace_phrase", "delete_phrase"]
    target_phrase: str
    replacement_phrase: Optional[str]         # Null for delete
    # Only materializes if deterministic on exact base_text_hash/base_provision_version_id
    # Otherwise edge is annotation/quarantined
```

### LLM Output Contract
LLM outputs LegalLocator format (citation text + structured path), NOT internal identifiers. Server resolver maps LegalLocator → provision_uid via legal_document_key + article/clause/point lookup. Resolution failure for ADD_PROVISION = valid (new provision). Resolution failure for REPEAL/REPLACE = quarantine.

### Operation Semantics
| Operation | Target Required? | Eligibility Change | Key Requirement |
|---|---|---|---|
| ADD_PROVISION | No | Reviewed only | New locator + parent/order + text or source span |
| REPLACE_TEXT | Yes | Reviewed only | base_provision_version_id or base_text_hash + successor or verified patch |
| REPEAL | Yes | Reviewed only | Cascade to children |
| AMEND | Yes | Never until refined | Generic placeholder |

Numbering: `\d+[a-z]?` accepted throughout parser, schema, resolver.

### Patch Materialization Rules
Patch form (`replace_phrase`, `delete_phrase`) only materializes if deterministic on the exact base (identified by base_provision_version_id or base_text_hash). Verification: apply patch to base text identified by hash, confirm result matches expected successor. If verification fails or base cannot be uniquely identified → edge becomes annotation/quarantined. Reviewer may provide full successor_text instead.

### Edge States
| State | Meaning | Eligibility Impact |
|---|---|---|
| `quarantined` | Invalid/unresolved/ambiguous/non-deterministic | None. Logged for manual review. |
| `auto` | Locator resolved, schema valid, but not reviewed | Annotation only. Does NOT change eligibility. |
| `reviewed` | Human or corroborated-evidence verified | Changes eligibility. Materializes validity. Sets validity_basis. |

### INVAR 10 (AMENDS States)
Only `reviewed` edges change eligibility. `auto` edges are annotation-only. `quarantined` edges are logged. This prevents LLM hallucination from automatically changing legal provisions.

### EvidenceLocator (Uses Defined IDs Only)
```python
@dataclass(frozen=True)
class EvidenceLocator:
    source_document_id: str              # docGUId
    source_metadata_revision_hash: str   # Defined in §4
    normalized_text_hash: str            # Defined in §4
    normalizer_version: str              # Identifies normalized artifact
    start_offset: int                    # Character offset in normalized artifact
    end_offset: int
    excerpt_hash: str                    # SHA256 of extracted excerpt
```
All IDs are defined in §4 identity model. Evidence offsets bind to the specific normalized artifact identified by `(normalized_text_hash, normalizer_version)`. Verified from normalized artifact before review/apply. Tamper-evident.

### Versioned Review Log
A `review_log` artifact is included in each release manifest (D52). Records all human-reviewed AMENDS decisions: edge ID, reviewer, timestamp, decision (approve/reject/refine), rationale. Immutable. Part of release audit trail.

---

## 11. Embeddings + Indexing (Pillar 2)

Batch-first for documents. Query embedding happens online at request time (required for ANN). Metadata filtering is the first gate; vector search is the fallback.

### Document Embedding Job (offline, arq stage 6)
- Model: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base-v2, 768-dim, cosine-normalized). Exact HuggingFace revision pinned in manifest.
- Mandatory: `pyvi.ViTokenizer.tokenize` applied to every text before embedding, both index and query.
- Batch size 32–64 texts. CPU default (D27). Optional off-peak GPU scheduling.
- Idempotent: skip provisions with unchanged provision_text_hash (reuse vector).
- Recompute payload/context/derived-state hash for all affected nodes even if vector reused.
- Documents embedded in batch offline. Queries embedded online at request time.

### Qdrant Per-Release Collection Layout
```
Collection: "legal_v{N}" (one per release, named in release_descriptor)
  point id   = UUIDv5(POINT_NAMESPACE, provision_version_id)
  vector     = embedding 768d (HNSW, cosine)
  payload    = {
     provision_uid,              // stable legal key
     provision_version_id,       // canonical content ID
     source_document_id,
     legal_document_key,
     doc_identity,               // display
     doc_type,
     validity_intervals[],       // [{from, to, state: in_force|conditional|unknown, basis}]
                                 // Stored IMMUTABLE rule state; NOT active/inactive
     issue_date, upd_datetime,
     field[], organ[],
     label (Article/Clause/Point),
     number, parent_article, parent_clause,
     corpus_version,
     embedding_version,
     context_text,               // immutable context-ready text snapshot
     
     // Citation projection (§12/D43):
     title,
     official_url,
     human_citation,
     provision_path,             // ["Điều 6", "Khoản 3", "Điểm a"]
     snippet
     // NOTE: NO static applicability_basis. Selected interval/basis
     // computed fresh per request and carried in SSE sources[].
  }
```
Payload indexes enabled on filter fields. Nested indexes on validity_intervals for correct interval evaluation.

### INVAR 7 (Eligibility Predicate)
Every Qdrant query includes the eligibility predicate checking `validity_intervals` against `reference_date`:
```
ANY interval WHERE:
  interval.from <= reference_date
  AND (interval.to IS NULL OR interval.to > reference_date)
  AND interval.state == 'in_force'   // conditional/unknown excluded by default (D40)
```
Applied identically to Neo4j BM25 (WHERE clause with same logic, scoped to release_id). Both filtered BEFORE final LIMIT and BEFORE RRF fusion. Nested interval evaluation ensures from/to/state checked on the same interval element.

### BM25 Keyword — Baseline Hypothesis (D11)
BM25 lives in Neo4j fulltext (Lucene). This is a **baseline hypothesis**, not a claimed-superior configuration. Validated through quality gate.

BM25 queries MUST filter by release_id + validity predicate BEFORE final LIMIT. This prevents release N-1 results from crowding out release N results when both coexist. Test verifies this invariant.

App-layer RRF fuses dense (Qdrant) + keyword (Neo4j fulltext). Same eligibility predicate applied to both before fusion.

---

## 12. Retrieval + Generation (Pillar 3)

Tight request path. Dual cap: max 8 serving contexts AND token budget. Early-exit wherever possible.

### Key Design Decisions
1. **Dual cap**: Maximum 8 serving contexts AND token budget (default 4096 tokens, configurable). Expansion counts toward the token budget. If expansion would exceed budget, truncate. Evaluation retrieval_k and serving context_k are separate parameters.
2. **Decompose contract**: Code enforces `max_subqueries=6` and deduplicates by normalized query string. Prompt alone insufficient; code contract authoritative.
3. **Eligibility predicate mandatory**: Every Qdrant and Neo4j BM25 query carries the eligibility predicate with `reference_date` (pinned release). No raw unfiltered search ever.
4. **Early-exit patterns**:
   - Cache hit (either layer) → skip remaining stages. Routing + rewriting already executed.
   - Intent = reject/greeting/clarify → skip retrieval entirely.
   - Empty retrieval result → return "không tìm thấy", skip gen.
   - Reference_date outside coverage → historical_coverage_unsupported.
5. **Context source**: Primary = Qdrant payload `context_text` (immutable snapshot, in pinned release collection). Fallback = Neo4j hierarchy traversal (filtered by pinned release_id). Both apply same eligibility predicate.
6. **Expansion**: Apply same eligibility predicate to expanded nodes (siblings, children). Expand + dedupe THEN final-select ≤8 contexts and token budget.
7. **Per-request applicability**: After retrieval, compute applicability fresh for each result based on reference_date + selected interval. Carry selected interval/basis in SSE sources[]. NOT a static stored value.
8. **Citation projection**: When Neo4j is down, sources[] rendered from Qdrant payload citation projection fields + per-request selected interval/basis. Degraded but functional.

### Execution Model (D51)
FastAPI is async but Neo4j/Qdrant drivers + reranker are blocking:
- Blocking DB calls: bounded thread pool executor with BOUNDED admission queue (not unbounded). Configurable max_workers + max_queue_size.
- GPU reranker: concurrency semaphore within API process.
- Per-stage timeout/deadline. Total request deadline.
- On saturation: 429 (rate limited) or 503 with `Retry-After` header.
- OOM protection: VRAM monitoring, graceful degradation to CPU if GPU OOM.
- Cancellation: propagate through stages; release resources on client disconnect.
- Streaming accumulator: bounded buffer. Prevents unbounded memory growth during long generations.

### Generator Prompt Wrapper (D47)
Versioned prompt wrapper that receives:
- reference_date
- legal_time_basis
- Selected intervals/basis per context (computed fresh per request)
- Immutable citations (canonical UIDs + human citations)
- Eligible contexts (filtered by eligibility predicate)

Constraint: answer ONLY from supplied eligible contexts. Does NOT prioritize "Ngày hiện tại" (legacy prompt flaw — lines 132, 147, 174 in NLP-LegalQA prompts.py). Temporal grounding is explicit via reference_date + legal_time_basis parameters.

Legacy QA prompt from NLP-LegalQA is NOT reused verbatim. It is a reference for domain terminology and few-shot examples, wrapped in the versioned prompt wrapper that controls temporal behavior.

### Text2Cypher (D18)
Enum TemplateCall with typed params:
```python
class TemplateKind(str, Enum):
    COUNT_ARTICLES_BY_DOC = "count_articles_by_doc"
    LIST_SIGNERS_OF_DOC = "list_signers_of_doc"
    LIST_DOCS_BY_YEAR = "list_docs_by_year"
    COUNT_DOCS_BY_FIELD = "count_docs_by_field"
    # ... extensible

class TemplateCall(BaseModel):
    kind: TemplateKind
    params: dict  # Typed per-template; strict validation, no extra fields
```

Server-owned parameterized templates. Each template has:
- Fixed server-side Cypher query (never model-generated)
- Hard LIMIT clause
- Timeout
- Result schema definition
- Citation mapping (returns canonical UIDs for citation building)
- Validity predicate built-in
- Scoped to pinned release_id

Invalid template kind or params → route to retrieval. No legacy raw-Cypher generator reused.

---

## 13. API + SSE Contract

### POST /v1/chat

Request:
```typescript
{
  query: string              // max 2000 chars
  chat_history?: Array<{role: "user"|"assistant", content: string}>  // max 20 turns, each max 4000 chars
  as_of_date?: string        // ISO 8601 date, default today, Asia/Ho_Chi_Minh
  event_date?: string        // ISO 8601 date, optional
  violation_state?: "completed" | "ongoing" | "unknown"
  occurred_at?: string       // ISO 8601 date
  detected_date?: string     // ISO 8601 date
  filters?: {
    doc_type?: string[]
    field?: string[]
    organ?: string[]
  }                          // Max cardinality enforced per field
}
```

Browser/client CANNOT control: `top_k`, `provider`, `fetch_k`, `rerank_top`, `labels`. These are server configuration.

Total request body/history/filter cardinality limited (D51). Validation errors returned as HTTP problem response BEFORE opening SSE stream.

Response: SSE stream (`Content-Type: text/event-stream`). Client reads via `fetch()` + `ReadableStream`.

### SSE Event Sequence
ALL valid intents follow the same sequence: `meta` (first, always) → `sources` (optional) → `token*` (zero or more) → exactly one terminal: `done` OR `error`.

```
event: meta
data: {"state": "started", "trace_id": "...", "corpus_version": ...,
       "release_fingerprint": "...", "as_of_date": "...", "event_date": "...",
       "reference_date": "...", "legal_time_basis": "event|detected|current",
       "violation_state": "...", "streaming_mode": "streaming|buffered"}

event: sources
data: [{"citation": "Điểm a Khoản 3 Điều 6 NĐ 168/2024/NĐ-CP",
        "title": "...", "official_url": "...",
        "provision_path": ["Điều 6", "Khoản 3", "Điểm a"],
        "selected_interval": {"from": "2025-01-01", "to": null, "state": "in_force"},
        "applicability_label": "applicable",
        "snippet": "..."}]

event: token
data: {"text": "..."}

event: done
data: {"trace_id": "...", "outcome": "full|degraded|cached|clarified|coverage_unsupported",
       "degraded_components": [], "warnings": []}

event: error
data: {"code": "...", "message": "...", "trace_id": "..."}
```

Notes:
- `meta` contains ONLY immutable request/release fields and `state:"started"`. No outcome/warnings.
- `sources[]` items carry SELECTED interval/basis computed fresh per request (NOT a static stored value). Include applicability_label.
- `done` carries final outcome, warnings, degraded_components. Outcome includes `clarified` and `coverage_unsupported`.
- Client disconnect: server cancels generation + logs. Does NOT emit `done` after disconnect.
- `streaming_mode`: "streaming" or "buffered". Buffered mode emits exactly one `token` event with full text.
- Validation errors / 429 / 503 BEFORE opening stream → HTTP problem response. After stream opens → SSE `error` event.
- Clarify intent: emit one deterministic `token` event (clarification text) then `done` with outcome="clarified".

### Cache Write Timing (D34)
Answer cache written ONLY after terminal `done` success. NOT written on:
- Pre-token failure (validation, routing error)
- Mid-stream error (generator failure after some tokens)
- Client disconnect

Retrieval cache written after successful retrieval+rerank completion (before gen), but only if that stage completed without error.

### POST /v1/feedback
```typescript
{
  trace_id: string
  rating: "helpful" | "not_helpful"
  comment?: string          // max 1000 chars
  idempotency_key: string   // Client-generated UUID
}
```
Validates trace ownership. Returns 200 on success, 409 if idempotency_key already submitted.

No admin HTTP cache-invalidation endpoint in v1. Cache invalidation via CLI/local-only command only.

### Health Endpoints
- `GET /livez`: Process alive. No dependencies. 200 immediately.
- `GET /readyz`: Checks active reconciled release (descriptor loadable), Qdrant collection exists + count matches, Neo4j release_id snapshot + constraints + fulltext readiness, manifest_hash matches. Fresh volume → 503 `corpus_not_ready`. NO remote LLM call.
- `GET /metrics`: Prometheus exposition format.

---

## 14. Caching (Pillar 4)

Three-layer Redis cache in `cache:` namespace. Keys include full dependency fingerprint. Two distinct lookup points. Cache write only after terminal success.

### Cache Lookup Ordering
1. **Answer cache**: After route + rewrite (key depends on rewritten query + intent + facts).
2. **Retrieval cache**: After decompose, before search/rerank.

Both lookups occur AFTER release descriptor is pinned and dates/facts resolved.

### Cache Key Design
Keys capture everything affecting the cached value:
- **Answer cache**: `cache:ans:{sha256(rewritten_query + intent + filters_canonical + as_of_date + event_date + reference_date + legal_time_basis + violation_state + occurred_at + detected_date + release_descriptor.fingerprint + embedding_version + reranker_version + model_name + prompt_wrapper_version + policy_version)}`
- **Retrieval cache**: `cache:ret:{sha256(decomposed_queries_sorted + filters_canonical + reference_date + violation_state + release_descriptor.fingerprint + embedding_version + reranker_version + retrieval_policy_fingerprint)}`
- **Context cache**: `cache:ctx:{provision_version_id}:{release_descriptor.fingerprint}` (raw text; per provision_version + release)
- **Amends cache**: `cache:amends:{provision_uid}:{release_descriptor.fingerprint}` (version/lineage-aware; resolved per request date)

Normalization for hashing: case-fold + whitespace collapse + unicode NFC normalize. Canonical JSON serialization for filters (sorted keys). NO stopword stripping.

`filters_canonical` = canonical JSON serialization of filter params (sorted keys, no whitespace). Ensures same filters produce same key regardless of param order.

`retrieval_policy_fingerprint` (D50) = SHA256(canonical JSON of {aggregate, bm25_enabled, rerank_model_version, fetch_k, rerank_top, top_k, heuristic_enabled, code_schema_version}). Captures all retrieval config knobs.

### Layer 1: Answer Cache
- Value: full answer + sources (with selected intervals/basis) + intent.
- TTL: long (e.g., 24h). Invalidated by release_descriptor.fingerprint in key.
- Written ONLY after terminal done success.
- HIT: skip retrieval + generation. New trace created (cache_origin_trace_id recorded internally).

### Layer 2: Retrieval Cache
- Value: top_k chunk references (provision_version_id + label + score). Stores provision_version_id, NOT just provision_uid.
- TTL: medium (e.g., 6h).
- Written after successful retrieval+rerank completion.
- HIT: skip search + rerank. Still run generation.

### Layer 3: Context/Amends Cache
- Context: raw text per (provision_version_id, release). No date-dependent labels. Decorated per-request.
- Amends: per (provision_uid, release). Resolved per request date.
- Invalidated by release_descriptor.fingerprint in key.

### Invalidation Strategy
Version-based. Release descriptor fingerprint embedded in keys means stale entries naturally miss on release change. Old entries expire via TTL. Negative caching avoided or given very short TTL.

### Feedback Interaction
User 👎 does not immediately invalidate cache. Logged to Langfuse. Manual cache invalidation via CLI only.

### Cache Metrics (Prometheus)
`cache_hit_ratio` per layer, `cache_miss_latency` vs `cache_hit_latency`. High-cardinality fields NOT used as Prometheus label values.

### Redis Role Separation (D28)
- Cache (cache:) failure → fail-open, skip cache. Optimization loss.
- Descriptor (descriptor:) failure → durable local fallback (§8). Different semantics.
- Broker (arq:) failure → ingestion stops. Serving unaffected.

---

## 15. Graph Schema + Validity

### Node Labels (Hierarchy)
```
Document → Part → Chapter → Section → Article → Clause → Point
```
Levels optional in real documents. Parser attaches to nearest present ancestor.

Metadata nodes: DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field. Fields multi-value.

### Identity (see §4)
Neo4j node physical key = snapshot_node_id (UUIDv5(RELEASE_NAMESPACE, release_id:provision_version_id)). Properties: provision_version_id (canonical content ID), provision_uid (lineage), legal_document_key, release_id, validity_intervals (stored immutable rule states), etc.

Multiple versions per provision_uid allowed within a release (historical versions retained). Multiple releases coexist (active + previous). Nodes distinguished by snapshot_node_id.

### Relationships
```
Hierarchy : HAS_PART / HAS_CHAPTER / HAS_SECTION / HAS_ARTICLE / HAS_CLAUSE / HAS_POINT
Metadata  : BELONGS_TO_GROUP / HAS_TYPE / HAS_STATUS / ISSUED_BY / SIGNED_BY / IN_FIELD
Cross-doc : RELATED_TO  (ExternalDocument stub keyed by source_document_id if outside corpus)
          : AMENDS {operation, evidence_locator, effective_date,
                    extractor_version, prompt_version, confidence, review_status, validity_basis}
```

ALL relationships connect nodes within the SAME release. Cross-release relationships are not permitted. Queries, expansion, reconciliation, and GC all scope by release_id.

### Constraints
Uniqueness constraint on snapshot_node_id for all hierarchy nodes. Composite uniqueness for metadata nodes. All MERGE-based, idempotent.

### External References (D25)
`docListOther` entries outside corpus → `ExternalDocument` node keyed by source_document_id (docGUID). Lightweight stub for citation only. NOT retrievable, NOT validity-bearing.

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
  "violation_state": "completed",
  "resolved_provision_uids": ["168/2024/NĐ-CP::article::6::clause::3::point::a"],
  "reference_answer": "...",
  "notes": "..."
}
```
Labels resolved against specific corpus release. Held-out 80/20 split stratified by legal_document_key. Versioned.

**Retrieval quality**: Recall@8, Precision@k, MRR, nDCG. Eval retrieval_k separate from serving context_k. Token budget assertion. Run fingerprint includes release_descriptor.fingerprint.

**Reproducibility**: Router/rewrite/decompose are provider-neutral LLM calls; temperature=0 alone insufficient. CI replays versioned query-plan fixtures OR records canonical outputs per run. Run fingerprint enforced. Resume must not mix outputs from different fingerprints.

**AMENDS-aware relevance**: Successor UIDs marked acceptable for queries with event_date after amendment effective date.

**Answer quality**: LLM-as-judge scores 5 criteria. ROUGE/BERTScore supplementary. LLM-judge nightly/manual.

**Quality gate**: Recall@8 must not drop more than 2% vs established baseline (absolute threshold). nDCG must not drop more than 3%. Gates promotion/deploy, NOT every PR. Baseline artifact pinned. Split membership fixed. Threshold semantics absolute. Promotion rule documented.

### B. Online Observability
Langfuse (self-hosted) + Prometheus + Grafana. Trace every request. Attach feedback. Infrastructure metrics. Telemetry redaction/retention policy. NO high-cardinality labels.

### C. Data Drift Detection
Compare ingestion_manifest across runs. Flag unexpected drift. On model/segmenter change: offline eval BEFORE swapping production index. Deploy only if quality gate passes.

---

## 17. Error Handling

Philosophy: fail gracefully, never crash request path. Honest degradation.

### Request Path (Online)
| Component failure | Behavior |
|---|---|
| Release descriptor unavailable (Redis + local fallback both fail) | 503. Cannot determine active release. |
| Router LLM | Fallback to intent `retrieve` |
| Rewriter | Use original query, skip rewrite |
| Decomposer | Fallback to `[{"query": original}]` |
| Qdrant unavailable | Degraded error. Cannot claim functional retrieval. |
| Neo4j unavailable | Qdrant payload context_text + citation projection. Degraded but functional. BM25/expansion/Text2Cypher unavailable. Tag in SSE done. |
| Both stores unavailable | Friendly "system temporarily unavailable" message. |
| Reranker | Use ANN ordering, skip rerank. |
| Generator LLM (pre-token) | Retry once. After exhaustion: friendly error. |
| Generator LLM (mid-stream) | NO retry. Stream truncated. Emit error event. |
| Redis cache (cache:) | Fail-open, skip cache. |
| Redis descriptor (descriptor:) | Durable local fallback. |
| Langfuse/Prometheus | Fail-open. Log locally. |
| Missing facts for sanction/transitional | Clarify intent. No retrieval. |
| Reference_date outside coverage | historical_coverage_unsupported. |
| Admission queue saturated | 429/503 + Retry-After. |

### Ingestion Path (Offline)
See §9 error table. Key: AMENDS extraction failure quarantines edge, does not block document ingest. Parse-quality gate prevents data loss. Reconcile failure blocks publish.

### Cross-Cutting
- Structured JSON logging with `trace_id`.
- Timeouts on all LLM calls and DB queries.
- Bounded executor with bounded admission queue. GPU semaphore within API process. Deadline propagation.
- OOM: VRAM monitoring, graceful degradation.
- Cancellation: propagate through stages; release resources on disconnect.

---

## 18. Testing Strategy

### Unit Tests (No Network, No Services)
- Parser hierarchy: regex including alphanumeric clause numbers (2a, 18a). Footer split. Quote-block tracking. NFC normalization. Optional-level handling. Fixture-based.
- Identity: legal_document_key derivation, provision_uid building, provision_version_id from provision_text_hash, snapshot_node_id derivation, UUIDv5 determinism, round-trip mapping.
- AMENDS discriminated union: valid/invalid schemas per operation type. ADD without target = valid. REPLACE without base hash = quarantine. Patch non-deterministic on base = quarantine. Enum normalizer. EvidenceLocator verification.
- Cache key construction: all dependency dimensions (including release_descriptor.fingerprint, facts, retrieval_policy_fingerprint). Same inputs → same key. Version bump → key change.
- Eligibility predicate: various validity_intervals combinations, reference_date boundary cases, half-open [from,to) semantics, nested interval evaluation (same-element), conditional/unknown excluded by default.
- Per-request applicability computation: same stored intervals, different reference_dates → different applicability_labels.
- RRF aggregation.
- Heuristic rerank: recency calculation, post-eligibility application.
- Query normalization for cache (no stopwords, canonical JSON filters).
- Text2Cypher TemplateCall: valid param extraction, invalid kind → route to retrieval. Strict discriminator.
- Subquery dedup + max_subqueries enforcement.
- Token budget enforcement in context building.
- SSE event serialization.
- requires_clarification classifier.
- Prompt wrapper: reference_date + legal_time_basis correctly injected.
- ReleaseDescriptor.fingerprint deterministic derivation.
- Bounded admission queue: saturation → 429/503.

### Integration Tests (Single Docker-Compose Test Profile)
- Import parsed JSON → verify node/relationship counts + constraints hold.
- Batch embed → verify Qdrant point count + payload filter works + context_text populated + citation projection populated.
- End-to-end retrieval on fixture document: query → expected top_k provision_version_ids match reference.
- Dual-write reconciliation: simulate partial failure → verify no release published. Verify relationship digest checked.
- Tombstone: REVIEWED REPEAL → validity interval closed (to set), text/evidence preserved.
- Cache invalidation: publish new release → verify old keys miss.
- Atomic publish: verify serving sees consistent state after flip.
- Partial parse: verify last-good kept, no tombstone.
- AMENDS cascade: REVIEWED REPEAL of Article → Clauses/Points intervals closed.
- Eligibility predicate: event_date before/after amendment → different results.
- Staging N doesn't affect serving N-1 (snapshot isolation).
- Unchanged corpus carry-forward (no-op cron doesn't bump version).
- Rollback: flip back to previous release → serving reverts correctly. No cross-release leakage.
- Same-day replacement boundary ([from, to) half-open semantics).
- Multi-interval predicate (nested evaluation, no cross-filtering).
- NĐ 168 exceptions (general 2025-01-01, exception 2026-01-01, conditional, time-of-act).
- ADD_PROVISION successor creation.
- REPLACE_TEXT successor-before-predecessor-close.
- Metadata-only revision (different source_metadata_revision_hash, same raw_blob_hash).
- Normalizer revision (different normalized_text_hash + normalizer_version).
- Degraded Qdrant citations (Neo4j down → sources[] from payload).
- Redis descriptor outage → durable fallback serves last pinned release.
- Fresh volume bootstrap/restore → /readyz returns 503 corpus_not_ready.
- Two-container GPU contention (API rerank GPU + worker embed CPU → no contention).
- BM25 release isolation: N-1 results don't crowd out N results when both coexist.
- Fulltext index readiness checked in reconcile.
- Expansion applies eligibility predicate + final cap (≤8 + token budget).
- Generator prompt wrapper: historical reference_date produces historically-grounded answer.
- Future-effective act ingested before effective date → intervals with future from; eligibility predicate excludes for current reference_date.
- Historical cache/context no collision between two provision versions (keys include provision_version_id + release fingerprint).
- Completed vs ongoing violation/detected_date → different reference_date derivation.
- Interval before/on/after repeal boundary.
- Failed/disconnected stream NOT cached.
- Graph relationship reconcile catches missing relationships.
- Clarify intent: missing facts → one deterministic token + done, no retrieval.
- Coverage unsupported: reference_date outside window → distinct response.

Never touch production data. Isolated test database/collection. Placeholder credentials only. Demo/live-service scripts NOT in unit suite.

### Acceptance Tests (Concrete Scenarios)
| Scenario | Expected |
|---|---|
| Before/after amendment | Same legal_locator with event_date before/after → different applicable provision |
| Event_date transition | NĐ 100 vs NĐ 168 for same violation → correct provision by event_date |
| Amendment adds 2a | ADD_PROVISION with numbering `2a` → retrievable, cited correctly |
| Repeal cascades to children | REVIEWED REPEAL of Article → Clauses/Points intervals closed |
| Partial-ingest not exposed | Truncated/bad fetch → last-good kept, no tombstone |
| Stale payload after amendment | Payload/context hash recomputed, vector reused if text same |
| Truncated parser no tombstone | Parse-quality gate fails → last-good retained |
| SSE disconnect | Server cancels, no done emitted, resources freed, NOT cached |
| Cache-hit fresh trace | New trace_id, cache_origin_trace_id recorded |
| Fresh-volume bootstrap | Bootstrap from backup or re-ingest produces consistent serving |
| Staging isolation | During ingest N, serving continues on N-1 |
| Rollback no cross-release | Flip back → all reads revert to N-1 |
| Same-day boundary | Event_date = effective_date → [from,to) selects NEW provision |
| Multi-interval | Disjoint active intervals → predicate evaluates each independently |
| NĐ 168 conditional | conditional state → excluded, warning returned |
| Degraded citations | Neo4j down → sources[] from Qdrant payload |
| Redis descriptor outage | Serve last pinned from local fallback |
| Metadata-only revision | New metadata hash, same blob hash → manifest updated |
| Normalizer revision | New normalized artifact, evidence offsets bound correctly |
| Two-container GPU | API GPU + worker CPU → no contention |
| N+N+1 coexist | Both releases queryable independently |
| Historical cache no collision | Two provision versions → distinct cache keys |
| Completed vs ongoing | Different reference_date derivation |
| Future-effective ingest | Ingested before effective date; excluded by predicate until effective |
| Clarify path | Missing facts → one token + done, no retrieval |
| Coverage unsupported | Outside window → distinct response |
| BM25 release isolation | N-1 doesn't crowd out N |
| Expansion final cap | Expand + dedupe → ≤8 contexts + token budget |
| Generator historical ref | Historical reference_date → historically-grounded answer |
| Failed stream not cached | Mid-stream error → answer NOT written to cache |
| Relationship reconcile | Missing relationships detected, publish blocked |
| Fulltext readiness | Fulltext index not ready → reconcile fails |

### Evaluation
See §16A. Recall@8, nDCG, token-budget assertion, bounded retries/deadline, run fingerprint, replay fixtures.

### CI
- `pytest` unit tests on every PR (no network).
- Integration tests on parser/importer/retrieval/security/identity/release changes.
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
├── .gitmodules
├── NLP-LegalQA/                         # Git submodule at pinned SHA (READ-ONLY REFERENCE)
├── src/
│   ├── legal_rag/
│   │   ├── identity.py                  # Identity model (§4)
│   │   ├── temporal.py                  # Temporal model (§5)
│   │   ├── release.py                   # Release descriptor + publish protocol (§8)
│   │   ├── ingestion/                   # OWNERSHIP: serving corpus parser/adapter (D53)
│   │   │   ├── parser.py                # Alphanumeric clause, appendix preserve
│   │   │   ├── effectivity.py           # Effectivity ledger extraction
│   │   │   └── amends.py               # Discriminated union extraction
│   │   ├── embeddings/                  # Embedding jobs (§11)
│   │   ├── retrieval/                   # Retrieval pipeline (§12)
│   │   ├── templates/                   # Text2Cypher parameterized templates (§12)
│   │   ├── cache/                       # Cache layers (§14)
│   │   ├── api/                         # FastAPI endpoints (§13)
│   │   ├── sse/                         # SSE contract (§13)
│   │   ├── prompts/                     # Versioned prompt wrapper (§12/D47)
│   │   └── config/                      # Allow-list, constants, namespaces
│   └── ...
├── frontend/                            # Next.js React app
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/
├── eval/                                # Eval harness + datasets + fixtures
├── scripts/                             # CLI tools (cache invalidation, bootstrap, backup)
├── docker-compose.yml
├── docker-compose.test.yml
├── Dockerfile                           # Shared image (API + worker entrypoints)
├── uv.lock
├── package-lock.json
└── docs/superpowers/specs/
```

### Pinning
Everything pinned for reproducibility:
- Python version (`.python-version`)
- Node version (`.nvmrc`)
- Base Docker image tags (exact digest)
- All Docker service image tags (Neo4j, Qdrant, Redis, Langfuse, Prometheus, Grafana)
- HuggingFace model revisions (exact commit hashes)
- NLP-LegalQA submodule SHA

All fingerprints recorded in corpus manifest + eval manifest + release descriptor.

### Volumes, Bootstrap, Backup (D42)
| Volume | Content | Backup Strategy |
|---|---|---|
| `neo4j_data` | Graph + constraints | neo4j-admin dump/restore |
| `qdrant_storage` | Per-release collections | Qdrant snapshot API per collection |
| `redis_data` | Cache + arq queue + descriptor | Redis RDB/AOF backup |
| `raw_corpus` | Immutable provenance-separated raw store | File-level backup |
| `manifest_store` | Ingestion manifests + release manifests + active_release.json + review_log | File-level backup |
| `quarantine_store` | Quarantined AMENDS edges | File-level backup |

One common `backup_manifest`. Restore = consistent release set. Cache/arq ephemeral.

Bootstrap procedure documented step-by-step.

### GPU Profile (D27)
Two containers (API + arq-worker). Python semaphore does NOT serialize GPU across processes.

**Default profile**: API reranker on GPU. Ingestion embedder on CPU. No contention. Slower ingestion but safe and simple.

**Optional off-peak GPU**: Worker scheduled during low API load periods with GPU access. Requires operational discipline.

OOM protection: VRAM monitoring in API. Graceful degradation to CPU if GPU OOM. Timeout/cancellation propagated.

---

## 20. Open Items (Environment/Data Dependent Only)

These are NOT design gaps. They are data discovery or environment setup tasks. None affect spec correctness.

1. **Exact allow-list IDs**: docGroup/field IDs for traffic law need live API discovery. Seeded from NLP-LegalQA's search params, refined via facet exploration.
2. **Langfuse resource budget**: Official self-host compose recommends minimum resources. Actual allocation depends on host machine.
3. **HuggingFace model download**: First-time download of weights. Cached in volume.
4. **UpdDateTime capability probe result**: Whether API supports updDateTime filtering. Probed at first ingestion; fallback activated if unsupported.

All design decisions affecting correctness are finalized in this spec.

---

## Changelog: v4 → v5

| Area | v4 Issue | v5 Fix |
|---|---|---|
| Neo4j snapshot isolation | Global unique constraint on provision_version_id breaks when active+previous coexist | snapshot_node_id = UUIDv5(RELEASE_NAMESPACE, release_id:provision_version_id); constraint on snapshot_node_id |
| Neo4j reconciliation | Only checked node/point set + count | Added relationship digest/count + fulltext-index readiness |
| BM25 release isolation | No explicit release_id filter in BM25 before LIMIT | BM25 MUST filter release_id + validity BEFORE final LIMIT |
| Temporal facts | Single event_date insufficient for sanction/transitional | Added violation_state, occurred_at, detected_date; clarification before retrieval |
| Temporal penalty answer | Could give specific X/Y before retrieval | Must retrieval+citations for each branch; no specific penalty before retrieval |
| Stored validity states | Used active/inactive (reference_date-dependent) | Changed to in_force/conditional/unknown (immutable rule state); applicability computed fresh per request |
| Static applicability_basis | Stored in Qdrant payload | Removed; SSE sources carry selected interval/basis computed fresh per request |
| Coverage unsupported | "Not found" for out-of-window dates | Distinct historical_coverage_unsupported response |
| Future-effective ingestion | Not explicitly addressed | Ingest promulgated-but-not-yet-effective; future intervals self-handle boundary |
| Effectivity ledger | Described but no schema | Full schema with scope_locator, rule_kind, precedence, condition, evidence_locator, validity_basis |
| ReleaseDescriptor.fingerprint | Not explicitly defined | Canonical deterministic derivation specified (D54) |
| Retrieval cache | Stored provision_uid | Changed to provision_version_id |
| Context/amends cache | Keyed by provision_uid | Keyed by provision_version_id + release fingerprint; amends version/lineage-aware |
| Retrieval policy fingerprint | Not in cache key | Added to retrieval cache key (D50) |
| Cache write timing | Not restricted | Write ONLY after terminal done success; no cache on failure/disconnect |
| Clarify intent | Not defined | Explicit clarify intent/path: one deterministic token then done |
| Request limits | Body/history/filter cardinality unlimited | Bounded; executor has bounded admission queue (D51) |
| AMENDS discriminated union | Simple schema | Full discriminated union: AddProvisionOp, ReplaceTextOp, RepealOp, AmendOp with operation-specific fields |
| REPLACE base requirement | Not specified | Requires base_provision_version_id or base_text_hash + verbatim successor or verified patch |
| Patch materialization | Not specified | Only if deterministic on exact base hash; else annotation/quarantined |
| EvidenceLocator IDs | Used undefined source_revision_id | Changed to defined IDs: source_document_id, source_metadata_revision_hash, normalized_text_hash, normalizer_version |
| Review log | Not specified | Versioned review-log/override artifact in release manifest (D52) |
| Raw provenance | Flat raw/{hash}/ directory | Separated: blobs/, metadata/, normalized/, manifests/ per source-revision (D49) |
| Evidence offset binding | Bound to ambiguous path | Bound to specific (normalized_text_hash, normalizer_version) |
| Parser ownership | Implied submodule usage | Explicit: src/legal_rag/ingestion owns serving corpus; submodule is read-only reference (D53) |
| Generator prompt | Legacy prompt reuse implied | Versioned prompt wrapper receiving reference_date, legal_time_basis, selected intervals/basis, citations; NOT legacy prompt verbatim (D47) |
| Text2Cypher security | Read-only mentioned but not detailed | Enum TemplateCall + typed params + server-owned parameterized templates + hard LIMIT/timeout/result schema/citation mapping (D18) |
| Neo4j principals | Not separated | API read-only principal; worker write/schema/publish principal (D48) |
| Tests | Missing scenarios | Added 18+ new acceptance scenarios covering all fixes |
