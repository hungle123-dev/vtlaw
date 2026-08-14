# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-15  
**Status:** Draft v7 (post-Codex round-6 revision — executable temporal payload with governing_date_kind per interval, review_pending warning mechanism, publish_fence in CAS, source_revision_identity manifest path, per-context legal_time_basis)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (vendored as git submodule at pinned SHA, read-only reference). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Code and infrastructure are production-shape; deployable to cloud if desired.
- Right-sized for a bounded corpus (Vietnamese traffic-law documents, thousands of chunks). No fake scale claims.
- Legally correct validity model with **executable** temporal semantics: every validity_interval carries `governing_date_kind` enum so Qdrant/Neo4j filters know which fact-derived date to use; static general and exception intervals do not overlap; per-context legal_time_basis.
- **True snapshot isolation**: every serving request pins exactly one immutable release descriptor; no half-visible state during ingest. Active + previous releases coexist safely.
- **Legal history preserved**: tombstone closes validity intervals only; never physical-deletes history due to reparse success or snapshot retention.
- **Publish integrity**: fencing token (`publish_fence`, strictly monotonic) + fingerprint CAS; no stale publishes; rollback allocates new fence.
- **Executable correctness**: AMENDS discriminated union with validated effective_rule_id; generator prompt versioned; cache write only after terminal success; vector reuse only on full embedding_input_fingerprint match; review_pending produces distinct outcome + warning.

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters. Out of scope for this corpus.
- Fine-tuning pipeline (QLoRA/Unsloth). Eval harness supports evaluating fine-tuned models but training is not shipped.
- Public deployment or multi-tenant auth in v1. Single-user demo.
- OCR. Source API returns structured HTML/JSON, not scanned PDFs.
- LLM-based legal validation of amendments. Pydantic validates structure, not legal meaning.
- Free-form Cypher from LLM in v1. Text2Cypher uses server-owned parameterized templates only.
- Free-text condition evaluation. Fact-dependent rules use finite reviewed enum only.
- Per-document/per-regime custom temporal evaluator in v1. If a provision requires a temporal evaluator outside the supported RuleKind enum → return `temporal_rule_unsupported` / clarification. Do not silently fall back to query_time.

---

## 2. Decisions Log

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Reality bar | Self-host docker-compose, production-shape | Right-sized for portfolio |
| D2 | Feature scope | Core RAG + amendment graph + chat/routing/template-Cypher + eval/monitoring. No fine-tune | Covers all 5 pillars |
| D3 | Data store | Neo4j (graph + BM25) + Qdrant (ANN + metadata + context snapshot) | Separation matches "metadata-first + dedicated ANN" |
| D4 | LLM provider | Provider-neutral OpenAI-compatible for ALL LLM calls including AMENDS extraction and template selection | Same factory everywhere |
| D5 | Monitoring | Langfuse (self-host compose with closure, volumes, backup, resource profile) + Prometheus + Grafana | Complementary concerns; concrete deployment spec |
| D6 | UI | Next.js/React with SSE streaming via fetch() ReadableStream | POST JSON body incompatible with native EventSource |
| D7 | Ingestion queue | arq (async Redis queue) | Async-native, built-in cron + retry, right-sized vs Celery |
| D8 | Cache | Redis (answer + retrieval + context), key includes full dependency fingerprint | Cost + latency controller |
| D9 | Service topology | Modular monolith (API container, single worker process) + separate arq-worker container + infra sidecars | GPU contention avoided via profile; same codebase |
| D10 | Vector storage | Qdrant only; drop Neo4j vector index | Avoids duplicate embed |
| D11 | BM25 placement | Neo4j fulltext (Lucene) as baseline hypothesis | Validated through quality gate, not assumed superior |
| D12 | Corpus scope | Versioned allow-list; NO effect_status gate; dependency closure for amending docs | NĐ 168 = "Hết Hiệu lực một phần" must be included |
| D13 | TLS verification | Default ON. `INSECURE_TLS=true` causes production startup FAIL; only allowed in dev profile | Production default verifies credentials; dev override explicit |
| D14 | Chunk limit | Max 8 serving contexts AND token budget (default 4096); expansion counts toward budget | Dual cap; eval_k separate from serving_k |
| D15 | Identity scheme | Eight-tier identity (§4); content hash NOT in stable legal keys; snapshot_node_id for Neo4j physical key | docIdentity collides; multiple releases coexist |
| D16 | Qdrant point ID | UUIDv5(POINT_NAMESPACE, provision_version_id) per-release collection | Snapshot isolation via separate collections |
| D17 | Temporal model | Half-open `[from, to)`; stored states {in_force, conditional, unknown}; each validity_interval carries governing_date_kind enum {query_time, conduct_time, detection_time}; effectivity ledger with RuleKind enum; eligibility ordering defined; no overlapping applicable intervals per provision/regime without materialized precedence | Executable per-interval filtering; static general+exception don't overlap |
| D18 | Text2Cypher v1 | Typed discriminated TemplateCall per template kind (extra="forbid"); hard LIMIT/timeout/result schema/citation mapping; output provision_version_id + CitationContext; resolve via legal_document_key (not doc_identity); no free-form Cypher | Eliminates injection risk; ambiguity rejection |
| D19 | SSE contract | meta(state:started) → sources? → token* → done\|error. Buffered mode = one token event. HTTP problem before stream open. Clarify intent = one deterministic token then done. Cache write only after terminal done success. Per-context governing_date_kind in sources[]. | Explicit protocol; consistent across intents |
| D20 | Secrets policy | Env vars only, `.env` gitignored, test placeholders, docs `<placeholder>` | Credential safety |
| D21 | Release/snapshot isolation | Immutable release_descriptor per publish; Qdrant per-release collection; Neo4j snapshot_node_id per (release_id, provision_version_id); single pointer (descriptor); durable local fallback | True snapshot isolation; active+previous coexist safely |
| D22 | AMENDS operations | Discriminated union: AddProvisionOp / ReplaceTextOp / RepealOp / AmendOp; LegalLocator output; EvidenceLocator + effective_rule_id required (validated); auto = annotation only; reviewed = eligibility change; conservative review_pending for unresolved after planned effective date with distinct outcome code | Safe executable semantics |
| D23 | Clause numbering | Parser accepts alphanumeric `\d+[a-z]?` (e.g., `2a`, `18a`) | Raw data contains alphanumeric clauses; legacy parser misses them |
| D24 | Appendix handling | Default preserve when no reviewed rule; exclude only via verifiable rule + test fixture | Phụ lục can contain binding legal rules |
| D25 | External references | ExternalDocument stubs keyed by source_document_id (docGUID), not doc_identity. RELATED_TO is global-reference exception with canonical edge tuple, reconcile, backup semantics. | Unambiguous internal key |
| D26 | Health endpoints | /livez (no deps), /readyz (checks active reconciled release + stores + fulltext readiness + init lock absent), /metrics. No remote LLM call in probes | Fast cheap probes |
| D27 | GPU profile | API reranker GPU, ingestion embedder CPU default (optional off-peak GPU). V1: single API worker process so in-process semaphore suffices. Multi-worker requires Redis distributed lease (out of v1 scope). | Two containers; single worker simplifies GPU coordination |
| D28 | Redis roles | Three namespaces: arq:, cache:, descriptor:. Descriptor has durable local fallback. Broker outage ≠ snapshot failure. | Clear separation of concerns |
| D29 | Historical versions | Each release contains ALL provision versions within temporal support window; snapshot_node_id per (release, provision_version); tombstone = interval close only | Legal history preserved; multiple versions per lineage coexist across releases |
| D30 | Provision version granularity | provision_version_id based on provision_text_hash (not whole-doc hash) | Amending Điều 1 doesn't churn other provisions' point IDs |
| D31 | Reconcile manifest | Publish requires manifest match: node tuple set, edge digest/endpoints (including RELATED_TO external edges), fulltext-index readiness, Qdrant point/config fingerprint. Not just count. | Detects silent corruption, relationship loss, stale indexes |
| D32 | GC grace | Delete old release snapshots only after grace > max request/SSE deadline. Never delete provision version history within temporal support window. | In-flight requests complete safely |
| D33 | Delivery | Git submodule at pinned SHA; commit .gitmodules + gitlink; CI checkout recursive + assert SHA | Reproducible vendoring |
| D34 | Cache lookups | Two distinct points: (1) answer cache after route/rewrite; (2) retrieval cache after decompose, before search/rerank. Write timing locked per layer. | Correct ordering; no partial-failure cache pollution |
| D35 | Context cache | Store raw context text (no date-dependent labels); decorate validity per-request based on governing_dates | Date-independent caching |
| D36 | Retry generation | Only before first token streamed; once streaming started, no retry | Post-stream retry produces garbled/duplicate output |
| D37 | Effect status not scope gate | Allow-list does NOT filter by document effect_status | NĐ 168 = "Hết Hiệu lực một phần"; gating excludes it incorrectly |
| D38 | Dependency closure | Documents amending/repealing in-scope provisions ingested fully regardless of keyword/field match | Amendment coverage |
| D39 | Search strategy | Literal search_queries sent upstream; regex/local matching post-download; no wildcard API keywords; updDateTime used only after capability probe | Respect API capabilities |
| D40 | Conditional eligibility | conditional/unknown NOT eligible until reviewer materializes in_force interval with evidence/basis | Prevents serving unverified provisions as effective law |
| D41 | Factual inputs | Canonical field = occurred_at; event_date deprecated alias (normalized at validation; both present with different values → 422); ongoing requires detected_date (NO fallback to as_of_date); ended_at for completed-lasting; after normalization only canonical fields in cache/SSE/eval/tests | Unambiguous factual contract |
| D42 | Backup manifest | One common backup_manifest covering Neo4j dump, Qdrant collections, raw corpus, manifests, quarantine. Cache/arq jobs ephemeral, NOT restored as serving state. | Consistent restore |
| D43 | Citation projection | Qdrant payload includes title, official_url, human_citation, provision_path, validity_intervals (with governing_date_kind per interval), snippet. SSE sources carry selected interval + governing_date_kind computed fresh per request. | Neo4j-down still renders sources[]; applicability is per-request per-context |
| D44 | Stored validity vs applicability | Stored: in_force/conditional/unknown (immutable rule state) with governing_date_kind per interval. Applicability computed fresh per request using governing_dates[governing_date_kind]. | Clean separation; no static applicability_basis |
| D45 | Future-effective ingestion | Ingest promulgated-but-not-yet-effective documents; future intervals self-handle boundary. Don't wait for effective date. | Pre-loading; boundary handled by eligibility predicate |
| D46 | Coverage unsupported response | governing_dates outside verified temporal_support_window/coverage → historical_coverage_unsupported (distinct from "not found") | Honest signal |
| D47 | Generator prompt wrapper | Versioned prompt wrapper receiving governing_dates, per-context legal_time_basis (governing_date_kind + governing_date + rule_kind + effective_rule_id), selected intervals/basis, immutable citations; answer ONLY from supplied eligible contexts. NOT legacy prompt verbatim. | Temporally grounded per-context; no "current date" bias |
| D48 | Neo4j principals | API read principal (read-only). Worker ingestion principal (write/schema/publish). Separate credentials. | Least privilege |
| D49 | Provenance separation | HTML blob by raw_blob_hash; metadata by source_metadata_revision_hash; normalized artifact by normalized_text_hash + normalizer_version; per-source-revision manifest at path derived from full source_revision_identity. Evidence offsets bind to specific normalized artifact. | Supports metadata-only update + normalizer revision without collision |
| D50 | Retrieval policy fingerprint | retrieval_policy_fingerprint = canonical hash of RRF/BM25/rerank/fetch_k/top_k/heuristic/code-schema knobs. Included in retrieval cache key. | Cache invalidation on config change |
| D51 | Bounded admission | Executor has bounded admission queue (not unbounded ThreadPoolExecutor queue). On saturation → 429/503 + Retry-After. Streaming accumulator bounded. | Backpressure; OOM protection |
| D52 | Review log artifact | Versioned review-log/override artifact in release manifest. Records human-reviewed AMENDS decisions. | Audit trail for legality changes |
| D53 | Serving corpus owner | src/legal_rag/ingestion parser/adapter owns serving corpus guarantees (alphanumeric clause, appendix). Submodule NLP-LegalQA is read-only reference only. | Legacy parser lacks these guarantees |
| D54 | ReleaseDescriptor.fingerprint | Canonical deterministic derivation: SHA256(canonical JSON of descriptor fields excl. fingerprint). Used consistently in cache/eval/pinning. | Deterministic cache keys |
| D55 | Canonical factual inputs | occurred_at canonical; event_date deprecated alias (normalized at validation; both present with different values → 422); ongoing requires detected_date (NO fallback to as_of_date); ended_at for completed-lasting; after normalization only canonical fields in cache/SSE/eval/tests | Unambiguous factual contract |
| D56 | EffectivityLedgerEntry structured | Required fields: entry_id, scope_locator, rule_kind (enum), effective_from/to OR fact-dependent evaluator, precedence, regime_id, evidence_locator, validity_basis, review_status. No free-text condition evaluator. | Executable materialization |
| D57 | Conditional materialization | conditional/unknown eligible ONLY after reviewer materializes in_force interval with evidence/basis. Predicate matches state=in_force; conversion happens via review materialization. | Consistent predicate; no prose-predicate contradiction |
| D58 | Eligibility ordering | Reviewed AMENDS-materialized intervals (highest) > fact-dependent transition rules > static effectivity intervals. Defined precedence resolves conflicts. No overlapping applicable intervals per provision/regime without materialized precedence. | Deterministic eligibility |
| D59 | AMENDS effective_rule_id | Every op requires EvidenceLocator + effective_rule_id (validated: exists, correct source/release, reviewed, scope covers target). Materialize only with known effective date. Pydantic validators for one-of/at-least-one. Edge ID includes evidence identity. | Safe executable AMENDS |
| D60 | Conservative review_pending | Auto/unresolved AMENDS → pending-amendment lookup detects targets after planned effective date; response outcome = legal_status_review_pending; warning served; not confident old-law answer; no release block. | Safety without blocking; distinct from "not found" |
| D61 | Publish CAS/fencing | Serialized publisher with fencing token (publish_fence, strictly monotonic, separate from corpus_version). Build records base_fingerprint + base_fence. Publish via atomic CAS checking BOTH fingerprint AND fence. Stale → discard/rebase. Rollback allocates new fence. Redis authority (Hash); local file durable cache. Exclusive bootstrap init. Audited rollback. | No stale publish; no stale re-publish after rollback |
| D62 | Metadata/ExternalDocument scope | Immutable global reference (not release-scoped, not GC'd with releases). Content + AMENDS nodes release-scoped. RELATED_TO to ExternalDocument is global-reference exception with canonical edge tuple. Separate manifest section. Constraints/traversal/GC consistent. | Clear scoping |
| D63 | Deterministic edge reconciliation | Canonical edge tuple per relationship type (including RELATED_TO external). Reconcile checks node tuple (snapshot_node_id, release_id, provision_version_id), exact edge digest/endpoints, Qdrant point/config fingerprint. | Full integrity verification |
| D64 | Embedding reuse fingerprint | Reuse vector only when embedding_input_fingerprint matches: model_revision + segmenter/tokenizer + preprocessing config + dimension + provision_text. Pipeline change → re-embed. | Model change invalidates cached vectors |
| D65 | Cache write semantics | Answer cache: written only after terminal done success; includes retrieval_policy_fingerprint + runtime_policy_fingerprint. Retrieval cache: written immediately after retrieval+rerank success (even if generator later fails). | Locked semantics; no ambiguity |
| D66 | Source revision identity | Composite: (source_document_id, raw_blob_hash, source_metadata_revision_hash, normalized_text_hash, normalizer_version, parser_version). Manifest path derived from full identity. Parse idempotency key = full identity. Manifest immutable per revision. | Changes when raw/parser/normalizer changes; no overwrite |
| D67 | Eval label resolution | Labels have resolved_provision_version_ids + release fingerprint + canonical facts (occurred_at, ended_at, detected_date, violation_state) + governing dates/rule kind. Dedup + group-split by legal_document_key before 80/20 split. | No leakage; reproducible |
| D68 | Typed TemplateCall | Discriminated request models per template kind (extra="forbid"). Resolve via legal_document_key (not doc_identity) with ambiguity rejection. Output provision_version_id + CitationContext. | Type-safe; no free-form params; collision-safe |
| D69 | Per-context legal_time_basis | legal_time_basis is per selected source/context (not global): {governing_date_kind, governing_date, rule_kind, effective_rule_id}. SSE sources carry governing_date_kind per item. Prompt wrapper receives per-context basis. | Different contexts may use different governing dates |
| D70 | No overlapping applicable intervals | Same provision/regime cannot have two overlapping applicable in_force intervals unless explicit materialized precedence rule. Static general truncated/split at exception dates. | Deterministic eligibility; no ambiguity |
| D71 | Regime-aware transition rules | Fact-dependent transition rules apply to legal regime/lineage (regime_id), not just source document. Transition from NĐ 168 → NĐ 100 traces predecessor + successor. | Correct cross-regime temporal reasoning |
| D72 | Unsupported temporal rule | If provision requires temporal evaluator outside supported RuleKind enum → return temporal_rule_unsupported / clarification. Do not silently fall back to query_time. | Honest about limitations |
| D73 | Pending-amendment lookup | Executable mechanism for review_pending: index of auto/quarantined AMENDS edges with planned effective_date <= governing_date. Checked during retrieval. Hit → legal_status_review_pending outcome + warning. | review_pending produces warning, not "not found" |
| D74 | AMENDS edge identity | edge_id = SHA256(operation + amending_snapshot_node_id + target_snapshot_node_id + effective_rule_id + evidence_locator.excerpt_hash). Two different amendments cannot MERGE into one edge. | Distinct edges for distinct amendments |
| D75 | Langfuse self-host spec | Compose closure with pinned versions, persistent volumes (postgres, clickhouse, minio), backup/restore procedure, resource profile (min RAM/CPU/disk). Not just external link. | Concrete deployment |
| D76 | INSECURE_TLS enforcement | INSECURE_TLS=true causes production startup to FAIL. Only allowed in dev/test profile. Environment variable checked at boot. | Security by default |
| D77 | Cache amends value | Value is date-resolved result (amends applicable at canonical governing_dates). Key MUST contain canonical governing_dates. | Date-aware amends cache |

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
│  ├─ api             : FastAPI (single worker process)                │
│  │                   validate+normalize→trace→facts                  │
│  │                   →pin release→route→rewrite→cache(1)            │
│  │                   →decompose→cache(2)→retrieve                    │
│  │                   →pending-amendment lookup                       │
│  │                   →rerank(GPU, in-process semaphore)              │
│  │                   →gen(prompt wrapper, per-context basis)         │
│  │                   Endpoints: POST /v1/chat, POST /v1/feedback,    │
│  │                   GET /livez, GET /readyz, GET /metrics           │
│  ├─ arq-worker      : ingestion offline (scrape→parse→effectivity    │
│  │                   →amends→embed(CPU)→build→reconcile→publish-CAS)│
│  │                   Same codebase, different entrypoint             │
│  │                   Uses write/schema/publish Neo4j principal       │
│                                                                      │
│  DATA STORES                                                         │
│  ├─ neo4j           : graph hierarchy + AMENDS + Lucene BM25         │
│  │                   Nodes keyed by snapshot_node_id                │
│  │                   Filtered by release_id per snapshot            │
│  │                   Metadata/ExternalDoc = immutable global        │
│  │                   API read-only principal                        │
│  ├─ qdrant          : Per-release collections (legal_v{N})          │
│  │                   HNSW dense + payload (intervals w/             │
│  │                   governing_date_kind) + context + citation      │
│  └─ redis           : arq broker (arq:*)                            │
│                       cache (cache:*)                               │
│                       descriptor authority (descriptor:active Hash)  │
│                       publish_fence counter (descriptor:fence_ctr)   │
│                       Config: AOF ON, noeviction                    │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : LLM trace, feedback, eval experiment          │
│  │                   (self-host compose with closure, volumes,       │
│  │                    backup, resource profile — see §19)            │
│  ├─ prometheus      : infra metrics, scrapes GET /metrics           │
│  └─ grafana         : dashboard                                     │
└──────────────────────────────────────────────────────────────────────┘
```

### INVAR 1 (Snapshot Isolation)
Every serving request pins exactly one release_descriptor at ingress. All reads use that pinned descriptor. No read crosses releases. Active + previous releases coexist safely.

### Request Path (Hot Path, Pillar 3)
```
POST /v1/chat {query, chat_history?, as_of_date?, occurred_at?,
               ended_at?, detected_date?, violation_state?, filters?}
  │
  ▼ VALIDATE + NORMALIZE:
     • Schema, lengths, ISO dates, timezone Asia/Ho_Chi_Minh
     • event_date (deprecated alias) → normalize to occurred_at
     • If both event_date AND occurred_at present with DIFFERENT values → 422
     • After normalization, only canonical fields used downstream
     • Filter cardinality limits, history length limits
  ▼ create fresh trace (new trace_id for EVERY request)
  ▼ PIN RELEASE DESCRIPTOR (read once from Redis Hash authority; fallback local file)
     If unavailable and no fallback: 503
  ▼ DERIVE GOVERNING DATES from facts:
     • completed → conduct_time = ended_at if provided else occurred_at
     • ongoing → detection_time = detected_date (REQUIRED; NO fallback)
     • unknown/no facts → clarification path (for sanction/transitional queries)
     • query_time = as_of_date (default today)
     • governing_dates = {conduct_time?, detection_time?, query_time}
  ▼ requires_clarification check:
     • Sanction/transitional question + missing required facts → clarify intent
       (one deterministic token then done, BEFORE retrieval)
     • governing_dates outside verified coverage → historical_coverage_unsupported
  ▼ route (LLM, temp=0, max_tokens=64)
     intent ∈ {greeting, cypher_query, retrieve, reject, clarify}
  │
  ├─ reject → template refusal, return done
  ├─ greeting → deterministic response, return done
  ├─ clarify → one deterministic clarification token, return done
  ├─ cypher_query → rewrite if history → TemplateCall selection (typed discriminated,
  │   extra="forbid", resolve via legal_document_key)
  │   → execute parameterized template (pinned release + validity predicate
  │     + LIMIT + timeout + result schema + citation mapping)
  │   → gen natural language → tee tokens + bounded accumulator
  │   → on done SUCCESS: write answer cache + trace → stream
  │
  └─ retrieve:
       ▼ rewrite (multi-turn; skip if no history)
       ▼ CACHE LOOKUP (1) — ANSWER CACHE (after route+rewrite):
          key = sha256(rewritten_query + intent + filters_canonical
              + as_of_date + occurred_at + ended_at + detected_date
              + violation_state + governing_dates_canonical
              + release_descriptor.fingerprint
              + embedding_version + reranker_version
              + model_name + prompt_wrapper_version
              + retrieval_policy_fingerprint + runtime_policy_fingerprint)
          HIT → new trace (cache_origin_trace_id recorded), skip everything below
          MISS → continue
       ▼ decompose (code enforces max_subqueries=6 + dedup)
       ▼ CACHE LOOKUP (2) — RETRIEVAL CACHE (after decompose, before search):
          key = sha256(decomposed_queries_sorted + filters_canonical
              + governing_dates_canonical + violation_state
              + release_descriptor.fingerprint
              + embedding_version + reranker_version
              + retrieval_policy_fingerprint)
          HIT → skip search+rerank, continue to context build
          MISS → search+rerank below
       ▼ multi_search (sub-queries parallel, PINNED release):
          Each sub-query:
            • Qdrant: eligibility predicate using governing_dates[interval.governing_date_kind]
              (nested interval evaluation on SAME element: from, to, state, governing_date_kind)
              + metadata filter → HNSW within eligible set
            • Neo4j: BM25 fulltext with SAME eligibility predicate
              filtered by release_id + validity BEFORE final LIMIT
          Fuse RRF per sub-query → aggregate across sub-queries
       ▼ fetch context from Qdrant payload context_text (primary)
          or Neo4j hierarchy traversal (fallback, pinned release_id)
       ▼ expand context (same eligibility predicate; expand + dedupe;
          final-select ≤8 contexts AND token budget)
       ▼ PENDING-AMENDMENT LOOKUP (D73):
          Check each candidate against pending-amendment index
          (auto/quarantined AMENDS edges with planned effective_date <= governing_date).
          Hit → mark context review_pending flag + amendment info.
       ▼ compute per-request applicability per context: for each result, select the
          interval whose [from,to) covers governing_dates[interval.governing_date_kind]
          with state=in_force; derive applicability_label + legal_time_basis per context:
          {governing_date_kind, governing_date, rule_kind, effective_rule_id}
       ▼ cross-encoder rerank (GPU, FP16, batch_size small)
       ▼ heuristic rerank (recency bonus AFTER eligibility, eval-gated)
       ▼ final-select top serving contexts (max 8 AND token budget 4096)
       ▼ build context_str with per-request applicability labels + per-context
          legal_time_basis (NOT cached — computed fresh). Tag [ĐÃ BÃI BỎ]/[SỬA ĐỔI]
          per governing_dates. Amends note. Auto-review annotations.
          review_pending targets flagged with warning context.
       ▼ WRITE RETRIEVAL CACHE (after successful retrieval+rerank,
          even if generator later fails)
       ▼ gen (versioned prompt wrapper: governing_dates + per-context
          legal_time_basis + selected intervals/basis + immutable citations
          + eligible contexts + review_pending flags;
          answer ONLY from supplied eligible contexts; caveated if review_pending)
          Retry ONLY before first token. Once streaming starts, no retry.
       ▼ tee tokens to client + bounded accumulator
       ▼ on terminal done SUCCESS: write ANSWER CACHE
          NOT written on pre-token failure, mid-stream error, disconnect
       ▼ stream SSE events: meta(state:started) → sources → token* → done|error
          (sources carry governing_date_kind per item; done carries outcome
          legal_status_review_pending if any context flagged)
       ▼ write trace to Langfuse + metrics to Prometheus
```

### Ingestion Path (Offline, Pillar 1) — Separate Container
```
cron/manual trigger → arq enqueue (arq: namespace)
  ▼ ALLOCATE FENCING TOKEN: Redis INCR descriptor:fence_ctr
    → monotonic publish_fence (strictly increasing, separate from corpus_version)
    Record base_fingerprint + base_fence (current descriptor's values at build start).
  ▼ scrape phapluat.gov.vn (versioned allow-list, NO effect_status gate,
    dependency closure, delta via updDateTime + content_hash + overlap window,
    capability-probed). TLS verify ON (INSECURE_TLS=true fails production startup, D76).
    Throttle 0.5–1s. Save raw immutable with provenance separation (§7).
    Ingest promulgated-but-not-yet-effective documents (D45).
  ▼ parse hierarchy using src/legal_rag/ingestion parser/adapter (owner
    of serving corpus guarantees, D53). Regex \d+[a-z]?. Footer split
    with appendix default-preserve. Quote-block tracking. NFC normalize.
    Idempotent keyed by source revision identity (D66). Multi-value field.
  ▼ parse-quality gate: raw length sanity, anchor presence, hierarchy
    count/coverage ratio. If partial/bad → keep last-good, do NOT tombstone.
  ▼ effectivity_extract: extract "Hiệu lực thi hành" + "Điều khoản chuyển tiếp"
    into effectivity ledger entries (§5). Structured RuleKind enum + regime_id.
    Reviewed entries materialize intervals with governing_date_kind.
    Static general truncated/split at exception dates (D70).
    Cross-regime transition rules traced via regime_id (D71).
    Unsupported temporal evaluator → temporal_rule_unsupported annotation (D72).
  ▼ amends_extract: LLM outputs LegalLocator format (§10). Discriminated
    union operation. EvidenceLocator + effective_rule_id required (validated:
    exists, correct source/release, reviewed, scope covers target).
    Pydantic validators for one-of/at-least-one. Edge ID includes evidence
    identity (D74). Failed/ambiguous → quarantine edge. Does NOT block document ingest.
    Auto/unresolved after planned effective date → indexed in pending-amendment
    lookup (D73); target will receive review_pending warning at serve time.
  ▼ embed batch (vietnamese-bi-encoder + pyvi tokenize, CPU default).
    Record segmenter_version + model_revision.
    Reuse vector ONLY when embedding_input_fingerprint matches (D64):
    model_revision + segmenter/tokenizer + preprocessing + dimension + text.
    Pipeline change → re-embed. Recompute payload/context/derived-state hash.
  ▼ BUILD release N (STAGING):
    Create Qdrant collection legal_v{N}:
      - Carry forward unchanged provisions from N-1 (reuse vectors if
        embedding_input_fingerprint matches, update payload).
      - Embed changed/new provisions.
      - Full serving projection: all provisions + historical versions
        within temporal_support_window.
      - Payload validity_intervals include governing_date_kind per interval.
    Write Neo4j nodes/rels with release_id=N:
      - Node identity = snapshot_node_id (per release + provision_version).
      - Multiple versions per provision_uid allowed.
      - Relationships connect nodes within same release only.
      - RELATED_TO to ExternalDocument = global-reference exception (D62/D25).
      - Materialize validity from REVIEWED AMENDS:
        REPLACE creates successor version with new interval BEFORE closing predecessor.
        REPEAL cascades to children.
        ADD creates new provision (target need not pre-exist).
        Static general truncated at exception dates (D70).
        Record validity_basis + effective_rule_id + governing_date_kind per interval.
      - Tombstone = close validity interval (reviewed evidence only).
        NEVER physical-delete text/evidence.
    Build/update fulltext indexes scoped to release_id.
    Build pending-amendment index (auto/quarantined edges with planned effective_date).
    Record base_fingerprint + base_fence.
  ▼ RECONCILE: compare manifest against both stores:
      - Node tuple set: (snapshot_node_id, release_id, provision_version_id).
      - Edge digest: hash of sorted canonical edge tuples + endpoint verification
        (including RELATED_TO external edges).
      - Derived_state_hash per point.
      - Fulltext-index readiness.
      - Qdrant point set + config fingerprint.
    Any mismatch → DO NOT publish. Alert. Manual intervention.
  ▼ PUBLISH (CAS with fencing, D61):
    Construct release_descriptor with canonical fingerprint (D54):
      {corpus_version: N, publish_fence: <allocated>, qdrant_collection: "legal_v{N}",
       neo4j_release_id: N, embedding_version, schema_version,
       manifest_hash, published_at, temporal_support_window_from,
       fingerprint: SHA256(canonical JSON of above excl. fingerprint)}
    Atomic CAS via Redis Lua script (descriptor stored as Hash):
      Check: current.fingerprint == base_fingerprint
             AND current.publish_fence == base_fence
             AND new.publish_fence > current.publish_fence
      If all true: HSET all descriptor fields atomically. Return success.
      If any false: Return stale_detected. Discard build, rebase, rebuild.
    Bootstrap (descriptor nil): exclusive init via SETNX descriptor:init_lock
      with TTL. Only holder writes initial descriptor. Others wait/retry.
    On CAS success: write active_release.json to persistent volume
      (atomic: write-tmp + rename). Load into API in-memory cache.
    On interrupted publish (crash between CAS and local file write):
      restart loads from Redis Hash authority (consistent). Local file may lag
      but Redis is authoritative. Recovery: reload from Redis.
  ▼ GARBAGE COLLECT (after grace period > max request/SSE deadline):
    Delete Qdrant collections + Neo4j release_id nodes for releases older
    than retention. NEVER delete provision version history within temporal
    support window of active release.
  ▼ write ingestion_manifest + review_log artifact, emit metrics.
```

### INVAR 2 (Full Projection)
Each release is a FULL serving projection. Delta detection only optimizes compute/vector reuse.

### INVAR 3 (Single Pointer with CAS Authority)
Redis Hash `descriptor:active` is the CANONICAL AUTHORITY. Local `active_release.json` is DURABLE CACHE written AFTER successful CAS. Write order: CAS Redis → write local file. Restart/recovery: load from Redis authority; Redis down → local file fallback. Rollback: allocate new publish_fence (INCR), CAS to previous corpus_version content with new fence + new fingerprint. Audited. No plain SET without CAS/fencing.

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
| `legal_document_key` | Normalized docIdentity validated unique (§4.1) | Stable per legal act | Internal non-ambiguous key for provision UID construction; resolver uses this (not doc_identity) |
| `provision_uid` | `{legal_document_key}::article::{n}::clause::{n}::point::{letter}` | Stable logical lineage | Machine-stable key independent of content version. Citations. AMENDS target/source. QA label reference. |
| `provision_version_id` | `{provision_uid}::rev::{provision_text_hash[:12]}` | Per provision content version | Canonical content ID. Property in Neo4j. Used to derive snapshot_node_id. Qdrant point ID within per-release collection. |
| `snapshot_node_id` | UUIDv5(RELEASE_NAMESPACE, `{release_id}:{provision_version_id}`) | Per (release, provision_version) | Neo4j physical node key. Unique per release-version pair. Enables active+previous coexistence. |

### Source Revision Identity (D66)
Composite identity for parse idempotency and manifest path:
```
source_revision_identity = (source_document_id, raw_blob_hash,
    source_metadata_revision_hash, normalized_text_hash,
    normalizer_version, parser_version)
source_revision_identity_hash = SHA256(canonical serialization of above)
```
Manifest path: `manifests/{source_revision_identity_hash}/manifest.json`. Each transform revision = separate immutable manifest. Blobs dedupe by hash. Parse idempotency key = full identity. Changes when raw body OR parser OR normalizer changes. New revision = new manifest entry, never overwrite existing.

EvidenceLocator binds to specific `(normalized_text_hash, normalizer_version)` within the manifest identified by `source_revision_identity_hash`.

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
  Qdrant payload.provision_version_id ←→ Neo4j node.provision_version_id
  Both grouped by provision_uid for lineage queries

Resolution:
  legal_locator → legal_document_key (via resolver with ambiguity rejection)
  → provision_uid (via article/clause/point lookup)
  provision_uid + provision_text_hash → provision_version_id
```

POINT_NAMESPACE and RELEASE_NAMESPACE are fixed project-specific UUID constants committed to code.

### Resolver Ambiguity Rejection
LegalLocator resolution uses `legal_document_key` (not `doc_identity`). If a citation maps to multiple legal_document_keys (ambiguous) → reject/quarantine. Templates also resolve via legal_document_key with ambiguity rejection (D68).

### INVAR 6 (Provision Version Granularity)
provision_version_id is based on provision_text_hash, NOT whole-document hash.

### INVAR 5 (History Preserved)
Tombstone = close validity interval (set `to`). NEVER physical-delete text/evidence due to reparse success or snapshot retention. Physical deletion only after temporal support window expires + grace period.

---

## 5. Temporal Model

### Factual Inputs (Canonical Contract, D55)
```typescript
{
  as_of_date?: string,           // Query context date (default today)
  occurred_at?: string,          // CANONICAL: when violation/event happened
  ended_at?: string,             // For completed-lasting conduct
  detected_date?: string,        // REQUIRED if violation_state=ongoing
  violation_state?: "completed" | "ongoing" | "unknown"
}
```

**Validation + normalization:**
- `event_date` accepted as DEPRECATED ALIAS at validation only.
- If `event_date` present and `occurred_at` absent → normalize: `occurred_at = event_date`.
- If BOTH `event_date` AND `occurred_at` present with DIFFERENT values → **422 Conflict**.
- After normalization, ONLY canonical fields (`occurred_at`, `ended_at`, `detected_date`) used in cache keys, SSE meta, eval datasets, acceptance tests, prompt wrapper. `event_date` never appears downstream.

**Governing dates derivation:**
- completed → `conduct_time = ended_at if provided else occurred_at`
- ongoing → `detection_time = detected_date` (**REQUIRED**; NO fallback to as_of_date; missing → clarify)
- unknown/no facts → clarification path (for sanction/transitional queries)
- `query_time = as_of_date` (default today)
- `governing_dates = {conduct_time?, detection_time?, query_time}`

All dates resolved once at ingress to `Asia/Ho_Chi_Minh` timezone, validated as ISO 8601. Propagated through retrieval, Cypher/templates, generation, citations, cache, SSE.

### INVAR 41 (Clarification Before Retrieval)
Server-side classifier determines if a query requires factual inputs. When required facts are missing → clarify intent (one deterministic token then done, BEFORE retrieval). No specific penalty X/Y before retrieval. Multi-branch answers require retrieval + citations for EACH branch.

### INVAR 46 (Coverage Unsupported)
If governing_dates are outside the verified temporal_support_window/coverage → respond with `historical_coverage_unsupported` (distinct from "không tìm thấy").

### INVAR 7 (Half-Open Intervals)
Validity intervals use half-open notation: `[from, to)`.

### Validity Interval Payload (Executable, Per-Interval)
Each `validity_intervals[]` element carries all fields needed for Qdrant/Neo4j to filter independently:
```json
{
  "from": "2025-01-01",
  "to": null,
  "state": "in_force",
  "governing_date_kind": "query_time",
  "effective_rule_id": "ledger_entry_123",
  "basis": "static_general_effectivity",
  "regime_id": "traffic_sanctions_v2",
  "precedence": 10
}
```

**`governing_date_kind` enum (per interval):**
```python
class GoverningDateKind(str, Enum):
    QUERY_TIME = "query_time"        # Use governing_dates.query_time (as_of_date)
    CONDUCT_TIME = "conduct_time"    # Use governing_dates.conduct_time (occurred_at/ended_at)
    DETECTION_TIME = "detection_time" # Use governing_dates.detection_time
```

**Eligibility predicate (checks ALL fields on SAME nested interval element):**
```
ANY interval WHERE:
  interval.from <= governing_dates[interval.governing_date_kind]
  AND (interval.to IS NULL OR interval.to > governing_dates[interval.governing_date_kind])
  AND interval.state == 'in_force'
```

This ensures each interval declares WHICH governing date it's evaluated against. Static intervals use `query_time`; fact-dependent transitional intervals use `conduct_time` or `detection_time`.

### INVAR 70 (No Overlapping Applicable Intervals)
Same provision/regime cannot have two overlapping applicable `in_force` intervals unless an explicit materialized precedence rule exists. 

**Static general + exception handling (NĐ 168 example):**
- General provisions: interval `[2025-01-01, ∞)` in_force, governing_date_kind=query_time.
- Exception provisions (e.g., Điều 6 khoản 3 điểm m): interval `[2026-01-01, ∞)` in_force. The GENERAL interval for these exception provisions is TRUNCATED at 2026-01-01 during materialization, so they do NOT have `[2025-01-01, ∞)`. Result: exception provisions are only applicable from 2026-01-01, not from 2025-01-01. No overlap.

Materialization logic: when STATIC_EXCEPTION applies to specific provisions, those provisions' general interval is split/truncated at the exception date. The exception interval replaces the truncated portion.

### EffectivityLedgerEntry (Structured, Executable, D56)
```python
class RuleKind(str, Enum):
    STATIC_GENERAL = "static_general"
    STATIC_EXCEPTION = "static_exception"
    COMPLETED_CONDUCT_TIME = "completed_conduct_time"
    ONGOING_DETECTION_TIME = "ongoing_detection_time"
    REVIEW_PENDING = "review_pending"

@dataclass(frozen=True)
class EffectivityLedgerEntry:
    entry_id: str
    scope_locator: str               # provision_uid or document-level
    rule_kind: RuleKind
    effective_from: Optional[date]   # For static kinds
    effective_to: Optional[date]     # For static kinds
    precedence: int
    regime_id: str                   # Legal regime/lineage identifier
    evidence_locator: EvidenceLocator
    validity_basis: str
    review_status: str               # auto | reviewed
```

**No free-text `condition: str` evaluator.** Fact-dependent rules use `RuleKind` enum. Rules outside enum → REVIEW_PENDING. No inference. If provision requires temporal evaluator outside supported enum → `temporal_rule_unsupported` / clarification (D72).

**Materialization:**
- `static_general` / `static_exception` (reviewed) → materialize validity_interval `[effective_from, effective_to)` with state `in_force`, governing_date_kind=`query_time`.
- `completed_conduct_time` / `ongoing_detection_time` (reviewed) → materialize validity_interval with governing_date_kind=`conduct_time` or `detection_time`. The interval itself is static; the governing_date_kind tells the predicate which fact-derived date to use.
- `review_pending` → not materialized; not eligible. Served with warning/clarification.

**Example: NĐ 168/2024/NĐ-CP:**
- General effectivity: `STATIC_GENERAL, effective_from=2025-01-01, regime_id=traffic_sanctions_v2` → materializes `[2025-01-01, ∞)` in_force, governing_date_kind=query_time for most provisions.
- Exception (specific provisions): `STATIC_EXCEPTION, effective_from=2026-01-01, regime_id=traffic_sanctions_v2` → materializes `[2026-01-01, ∞)` in_force for those provisions. General interval truncated at 2026-01-01 for exception provisions (D70).
- Transitional rule §54.1: `COMPLETED_CONDUCT_TIME, regime_id=traffic_sanctions_v2` → for completed violations, governing_date_kind=conduct_time. Conduct before 2025-01-01 → NĐ 100 era provisions apply (predecessor regime traced via regime_id, D71). Conduct after → NĐ 168 provisions apply.
- Conditional provisions: `REVIEW_PENDING` until reviewer materializes in_force interval with evidence.

### Eligibility Ordering (D58)
Deterministic precedence for conflicting rules:
1. **Reviewed AMENDS-materialized intervals** (highest — encode explicit legality changes).
2. **Fact-dependent transition rules** (COMPLETED_CONDUCT_TIME / ONGOING_DETECTION_TIME) — use governing_dates[conduct_time] or governing_dates[detection_time].
3. **Static effectivity intervals** (STATIC_GENERAL / STATIC_EXCEPTION) — use governing_dates[query_time].

No overlapping applicable intervals per provision/regime without materialized precedence (D70).

### Regime-Aware Transition Rules (D71)
Fact-dependent transition rules apply to legal regime/lineage (identified by `regime_id`), not just the source document containing the ledger. When NĐ 168 transitions to NĐ 100, the rule traces both predecessor (NĐ 100, regime_id=traffic_sanctions_v1) and successor (NĐ 168, regime_id=traffic_sanctions_v2). The transition rule links regimes. During eligibility evaluation, the correct regime's provisions are selected based on governing_dates[conduct_time].

### Stored Validity States (Immutable Rule State, D44)
Stored in validity_intervals as immutable rule state:
- `in_force`: Provision is in force during this interval (subject to governing_date check).
- `conditional`: Effectiveness depends on external condition. NOT forced to a fake date.
- `unknown`: Insufficient information.

These are IMMUTABLE properties of the provision version, independent of any particular query.

### Per-Request Per-Context Applicability (Computed Fresh, D69)
Applicability computed fresh per request PER CONTEXT using governing_dates[interval.governing_date_kind]:
- An interval with state=`in_force` where `[from, to)` covers governing_dates[interval.governing_date_kind] → applicable.
- `conditional`/`unknown` intervals → NOT applicable unless reviewer materialized a concrete `in_force` interval with evidence/basis (D57).
- No interval covering the relevant governing_date → not_applicable.

Per-context `legal_time_basis`:
```python
@dataclass(frozen=True)
class LegalTimeBasis:
    governing_date_kind: GoverningDateKind
    governing_date: date
    rule_kind: RuleKind
    effective_rule_id: str
```
Carried per selected source/context in SSE sources[]. Prompt wrapper receives per-context basis. NOT a single global value.

### INVAR 40 (Conditional Materialization, D57)
`conditional` and `unknown` provisions become eligible ONLY when a reviewer materializes a concrete `in_force` interval with evidence/basis. The eligibility predicate matches `state=in_force`; conversion from conditional→in_force happens via review materialization.

### INVAR 9 (requires_clarification Rule)
Server-side classifier determines if a query requires factual inputs. Missing facts → clarify intent (see INVAR 41). Executable server-side rule, applied before retrieval begins.

### Future-Effective Ingestion (D45)
Promulgated-but-not-yet-effective documents ingested immediately. Their validity_intervals have `from` in the future. Eligibility predicate naturally excludes them for governing_dates before `from`.

---

## 6. Corpus Scope and Allow-List

See v6 §6 (unchanged). No effect_status gate. Dependency closure. Search strategy. Appendix default-preserve.

---

## 7. Raw Provenance

Immutable provenance with separation supporting metadata-only updates and normalizer revisions (D49, D66).

### Store Layout
```
raw/
  blobs/{raw_blob_hash}/original.html
  metadata/{source_metadata_revision_hash}/metadata.json
  normalized/{normalized_text_hash}/{normalizer_version}/normalized.txt
  manifests/{source_revision_identity_hash}/manifest.json
```

`source_revision_identity_hash = SHA256(canonical(source_document_id || raw_blob_hash || source_metadata_revision_hash || normalized_text_hash || normalizer_version || parser_version))`. Each transform revision = separate immutable manifest. Blobs dedupe by hash.

### Manifest Contents
```json
{
  "source_revision_identity_hash": "...",
  "source_document_id": "...",
  "raw_blob_hash": "...",
  "source_metadata_revision_hash": "...",
  "normalized_text_hash": "...",
  "normalizer_version": "...",
  "parser_version": "...",
  "official_url": "...",
  "fetched_at": "...",
  "evidence_offsets": {"start": N, "end": M}
}
```

Evidence offsets bind to specific `(normalized_text_hash, normalizer_version)` within the manifest identified by `source_revision_identity_hash`. Manifest immutable per source revision identity. New revision = new manifest entry, never overwrite.

### Owner of Serving Corpus Guarantees (D53)
`src/legal_rag/ingestion` parser/adapter owns serving corpus guarantees. Submodule `NLP-LegalQA` is READ-ONLY REFERENCE only.

---

## 8. Release and Snapshot Isolation

### Release Descriptor
```python
@dataclass(frozen=True)
class ReleaseDescriptor:
    corpus_version: int
    publish_fence: int           # Strictly monotonic, separate from corpus_version
    qdrant_collection: str
    neo4j_release_id: int
    embedding_version: str
    schema_version: str
    manifest_hash: str
    published_at: datetime
    temporal_support_window_from: date
    fingerprint: str             # SHA256(canonical JSON of above excl. fingerprint)
```

Fingerprint derivation (D54): canonical JSON serialization of all fields except `fingerprint`, sorted keys, no whitespace. SHA256. Deterministic.

### Storage + Authority (D61)
- **Authority**: Redis Hash `descriptor:active`. Fields accessed via HGET/HSET. Single canonical pointer.
- **Publish fence counter**: Redis key `descriptor:fence_ctr` (INCR for monotonic fencing tokens).
- **Bootstrap lock**: Redis key `descriptor:init_lock` (SETNX with TTL for exclusive initialization).
- **Durable cache**: Local file `active_release.json` on persistent volume, written AFTER successful CAS. For fallback on Redis outage.
- **API in-memory cache**: Loaded at startup from Redis; refreshed on each CAS success.

### Publish Protocol (CAS with Fencing, D61)
1. Allocate fencing token: `publish_fence = INCR descriptor:fence_ctr`. Strictly monotonic.
2. Build release N. Record `base_fingerprint` + `base_fence` = current descriptor's values at build start.
3. Reconcile (§8 INVAR 4).
4. **Atomic CAS** via Redis Lua script (descriptor stored as Hash):
   ```lua
   -- KEYS[1] = descriptor:active
   -- KEYS[2] = descriptor:init_lock (for bootstrap only)
   -- ARGV[1] = base_fence, ARGV[2] = base_fp, ARGV[3] = new_fence
   -- ARGV[4..N] = field_name, field_value pairs
   
   local cur_fence = tonumber(redis.call('HGET', KEYS[1], 'publish_fence'))
   local cur_fp = redis.call('HGET', KEYS[1], 'fingerprint')
   
   -- Bootstrap: descriptor absent
   if cur_fence == nil then
     if redis.call('EXISTS', KEYS[2]) == 1 then
       for i = 4, #ARGV, 2 do
         redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1])
       end
       return 1  -- initialized
     else
       return -1  -- bootstrap contention; another worker initializing
     end
   end
   
   -- Normal publish: check fingerprint AND fence
   if cur_fp == ARGV[2] 
      and cur_fence == tonumber(ARGV[1]) 
      and tonumber(ARGV[3]) > cur_fence then
     for i = 4, #ARGV, 2 do
       redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1])
     end
     return 1  -- published
   else
     return 0  -- stale detected
   end
   ```
   Note: uses HGET/HSET (Redis Hash), not json_extract. cjson.decode alternative available if storing as JSON string but Hash is preferred.
5. On CAS success (return 1): write `active_release.json` atomically. Notify API in-memory cache.
6. On stale_detected (return 0): discard build, log, rebase on current active descriptor (read current fingerprint + fence), rebuild.
7. On bootstrap contention (return -1): wait briefly, retry reading descriptor. If still nil, acquire init_lock and retry CAS.
8. On interrupted publish (crash between CAS and local file write): restart loads from Redis Hash authority (consistent). Local file may lag but Redis is authoritative.

**Rollback (Audited):**
Allocate NEW publish_fence (INCR). CAS to previous corpus_version content with new fence + new fingerprint. Fence always increases even on rollback → stale builds cannot re-publish after rollback because base_fence won't match. Audit log: who/when/from_fence/to_fence/from_corpus_version/to_corpus_version. Via CLI command.

**NOT plain Redis SET.** Always CAS with fencing. publish_fence checked alongside fingerprint.

### INVAR 1 (Request Pinning)
At ingress, API reads active release descriptor ONCE from Redis Hash authority (fallback: local file). Passes pinned descriptor to every downstream read.

### INVAR 2 (Full Projection)
Each release is a full serving projection. Delta only optimizes compute/vector reuse.

### INVAR 3 (Single Pointer with CAS Authority)
Redis Hash is authority. Local file is durable cache. Write order: CAS Redis → write local file. Recovery: reload from Redis.

### INVAR 4 (Reconcile by Manifest, D63)
Publish requires manifest match:
- Node tuple set: `(snapshot_node_id, release_id, provision_version_id)`.
- Edge digest: hash of sorted canonical edge tuples + endpoint verification (including RELATED_TO external edges).
- Derived_state_hash per point.
- Fulltext-index readiness.
- Qdrant point set + config fingerprint.

### Garbage Collection (D32)
Delete old releases only after grace period > max request/SSE deadline. Never delete provision version history within temporal support window.

### Redis Descriptor Outage (D28)
Serve from in-memory descriptor + local active_release.json fallback. If NO release ever published → 503 `corpus_not_ready`.

### /readyz Checks
Active reconciled release descriptor loadable. Qdrant collection exists + count matches. Neo4j release_id snapshot + constraints + fulltext readiness. manifest_hash matches. Init lock absent (bootstrap complete). Fresh volume → 503.

### Backup/Restore (D42)
One common `backup_manifest`. Restore consistent release set. Cache/arq ephemeral.

---

## 9. Ingestion Pipeline (Pillar 1)

Fully offline and async. Separate arq-worker container. Uses write/schema/publish Neo4j principal (D48).

### Stages
1. **allocate_fencing_token**: Redis INCR `descriptor:fence_ctr`. Record base_fingerprint + base_fence.
2. **scrape**: Allow-list, NO effect_status gate, dependency closure, delta detection, TLS verify ON (INSECURE_TLS=true fails production, D76), throttle. Save raw immutable with provenance separation (§7). Future-effective ingestion (D45).
3. **parse**: `src/legal_rag/ingestion` parser/adapter (D53). Alphanumeric clause. Appendix default-preserve. NFC normalize. Idempotent keyed by source revision identity (D66). Multi-value field.
4. **parse-quality gate**: Raw length, anchors, hierarchy coverage. Partial → keep last-good, no tombstone.
5. **effectivity_extract**: Extract "Hiệu lực thi hành" + "Điều khoản chuyển tiếp" into structured EffectivityLedgerEntry (§5). RuleKind enum + regime_id. Reviewed entries materialize intervals with governing_date_kind. Static general truncated at exception dates (D70). Cross-regime transition traced via regime_id (D71). Unsupported evaluator → temporal_rule_unsupported annotation (D72).
6. **amends_extract**: Discriminated union (§10). EvidenceLocator + effective_rule_id (validated). Pydantic validators. Edge ID includes evidence identity (D74). Quarantine failures. Auto/unresolved after planned effective date → indexed in pending-amendment lookup (D73).
7. **embed**: Batch embed. Reuse vector ONLY when embedding_input_fingerprint matches (D64). Recompute payload/context/derived-state hash.
8. **build_release**: Create staging. Full projection. Carry forward. Materialize validity from REVIEWED AMENDS. Static general truncated at exceptions (D70). Tombstone = interval close only. Fulltext indexes. Pending-amendment index built. Record base_fingerprint + base_fence.
9. **reconcile**: Compare manifest (§8 INVAR 4). Mismatch → DO NOT publish.
10. **publish**: CAS with fencing (§8). Stale → discard/rebase. Bootstrap → exclusive init.
11. **garbage_collect**: After grace period.
12. **report**: Metrics + review_log artifact.

### Error Handling (Ingestion)
| Failure | Behavior |
|---|---|
| Scrape HTTP 200 + non-null `error` | Upstream error, retry. NOT end-of-catalog. |
| Scrape `docs[]` empty + `error==null` | End-of-catalog, stop pagination. |
| Parse fail on one doc | Skip doc, log, continue batch. |
| Parse-quality gate fails | Keep last-good. Do NOT tombstone. |
| AMENDS extraction invalid schema | Quarantine edge. Document ingest continues. |
| AMENDS ADD_PROVISION target-not-found | Valid. Create target. |
| AMENDS REPLACE without base hash/text | Quarantine edge. |
| AMENDS patch not deterministic on base hash | Quarantine or require full successor. |
| AMENDS effective_rule_id validation fails | Quarantine edge. |
| AMENDS auto/unresolved after planned effective date | Indexed in pending-amendment lookup. Warning at serve time. No release block. |
| Embed/import fail | Retry. After N failures mark failed, alert. |
| Reconcile mismatch | DO NOT publish. Alert. |
| Stale publish (CAS returns 0) | Discard build, rebase, rebuild. |
| Bootstrap contention (CAS returns -1) | Wait, retry. |
| Interrupted publish (crash between CAS and local file) | Restart loads from Redis authority. Consistent. |
| INSECURE_TLS=true in production | Startup FAIL. |

### Secrets Policy
Env vars only. `.env` gitignored. Test placeholders. Worker uses write/schema/publish principal; API uses read-only principal (D48).

---

## 10. AMENDS Edge Model

### Discriminated Union Schemas (D22, D59)
Every operation requires EvidenceLocator + effective_rule_id (validated). Pydantic validators enforce one-of/at-least-one constraints. Edge ID includes evidence identity (D74).

```python
class AddProvisionOp(BaseModel):
    operation: Literal["ADD_PROVISION"]
    amending_locator: LegalLocator
    new_provision_locator: LegalLocator
    parent_locator: LegalLocator
    order: int
    evidence_locator: EvidenceLocator
    effective_rule_id: str          # Validated: exists, reviewed, scope covers target
    
    @model_validator(mode='after')
    def check_text_or_span(self):
        if not self.successor_text and not self.source_span:
            raise ValueError("At least one of successor_text or source_span required")
        return self
    successor_text: Optional[str] = None
    source_span: Optional[EvidenceLocator] = None

class ReplaceTextOp(BaseModel):
    operation: Literal["REPLACE_TEXT"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    evidence_locator: EvidenceLocator
    effective_rule_id: str
    
    @model_validator(mode='after')
    def check_base_and_successor(self):
        if not self.base_provision_version_id and not self.base_text_hash:
            raise ValueError("At least one of base_provision_version_id or base_text_hash required")
        if not self.successor_text and not self.exact_patch:
            raise ValueError("At least one of successor_text or exact_patch required")
        return self
    base_provision_version_id: Optional[str] = None
    base_text_hash: Optional[str] = None
    successor_text: Optional[str] = None
    exact_patch: Optional[PatchSpec] = None

class RepealOp(BaseModel):
    operation: Literal["REPEAL"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    evidence_locator: EvidenceLocator
    effective_rule_id: str
    cascade: bool = True

class AmendOp(BaseModel):
    operation: Literal["AMEND"]
    amending_locator: LegalLocator
    target_locator: LegalLocator
    evidence_locator: EvidenceLocator
    effective_rule_id: str
```

### PatchSpec Validation
```python
class PatchSpec(BaseModel):
    kind: Literal["replace_phrase", "delete_phrase"]
    target_phrase: str
    replacement_phrase: Optional[str]
    occurrence_rule: str
    expected_successor_hash: Optional[str]
    expected_successor_text: Optional[str]
    
    @model_validator(mode='after')
    def check_expected_successor(self):
        if not self.expected_successor_hash and not self.expected_successor_text:
            raise ValueError("At least one of expected_successor_hash or expected_successor_text required")
        return self
```

### effective_rule_id Validation
Before accepting an AMENDS edge, validate:
1. `effective_rule_id` exists in the effectivity ledger.
2. Belongs to correct source/release.
3. `review_status == reviewed`.
4. `scope_locator` covers the target provision.

Failure → quarantine edge.

### AMENDS Edge Identity (D74)
```
edge_id = SHA256(operation || amending_snapshot_node_id || target_snapshot_node_id 
          || effective_rule_id || evidence_locator.excerpt_hash)
```
Two different amendments targeting the same provision cannot MERGE into one edge.

### Operation Semantics
| Operation | Target Required? | Effective Rule Required? | Eligibility Change |
|---|---|---|---|
| ADD_PROVISION | No | Yes (validated) | Reviewed only |
| REPLACE_TEXT | Yes | Yes (validated) | Reviewed only; successor before close |
| REPEAL | Yes | Yes (validated) | Reviewed only; cascade |
| AMEND | Yes | Yes (validated) | Never until refined |

### Edge States
| State | Meaning | Eligibility Impact |
|---|---|---|
| `quarantined` | Invalid/unresolved/non-deterministic/validation-failed | None. Logged. |
| `auto` | Resolved, schema valid, not reviewed | Annotation only. |
| `review_pending` | Auto/unresolved but past planned effective date | Target marked conservative; warning served via pending-amendment lookup (D73). Not confident old-law answer. No release block. |
| `reviewed` | Human/corroborated verified | Changes eligibility. Materializes validity. Sets validity_basis. |

### INVAR 10 (AMENDS States)
Only `reviewed` edges change eligibility. `auto` = annotation. `review_pending` = warning via lookup (D73). `quarantined` = logged.

### EvidenceLocator (Uses Defined IDs Only)
```python
@dataclass(frozen=True)
class EvidenceLocator:
    source_document_id: str
    source_metadata_revision_hash: str
    normalized_text_hash: str
    normalizer_version: str
    start_offset: int
    end_offset: int
    excerpt_hash: str
```
All IDs defined in §4. Evidence offsets bind to specific normalized artifact identified by `(normalized_text_hash, normalizer_version)` within manifest identified by `source_revision_identity_hash`. Verified before review/apply. Tamper-evident.

### Versioned Review Log (D52)
`review_log` artifact in each release manifest. Records: edge ID, reviewer, timestamp, decision (approve/reject/refine), rationale. Immutable. Audit trail.

### Pending-Amendment Lookup (D73)
Index derived from AMENDS edges where `review_status in {auto, quarantined}` AND `planned_effective_date <= governing_date` (using the appropriate governing_date_kind for the target provision). Built during release build stage.

During retrieval, after candidates selected, check each candidate's provision_uid against pending-amendment index. If hit:
- Mark context with `review_pending` flag + pending amendment info.
- Set response outcome = `legal_status_review_pending`.
- Include warning in SSE done warnings array.
- Generator prompt wrapper receives review_pending flag → produces caveated answer (not confident).

If target/effective date can't be determined → clarification/unsupported, not treated as no amendment.

---

## 11. Embeddings + Indexing (Pillar 2)

Batch-first for documents. Query embedding online at request time. Metadata filtering first gate; vector search fallback.

### Document Embedding Job (offline, arq stage 7)
- Model: `bkai-foundation-models/vietnamese-bi-encoder`. Exact HuggingFace revision pinned.
- Mandatory: `pyvi.ViTokenizer.tokenize`.
- Batch size 32–64. CPU default (D27).
- Idempotent: reuse vector ONLY when `embedding_input_fingerprint` matches (D64):
  ```
  embedding_input_fingerprint = SHA256(model_revision || segmenter_version || 
      tokenizer_version || preprocessing_config || dimension || provision_text)
  ```
  Changing ANY component → fingerprint changes → re-embed.
- Recompute payload/context/derived-state hash for affected nodes even if vector reused.

### Qdrant Per-Release Collection Layout
```
Collection: "legal_v{N}"
  point id   = UUIDv5(POINT_NAMESPACE, provision_version_id)
  vector     = embedding 768d (HNSW, cosine)
  payload    = {
     provision_uid, provision_version_id,
     source_document_id, legal_document_key, doc_identity,
     doc_type,
     validity_intervals[],       // [{from, to, state, governing_date_kind,
                                 //   effective_rule_id, basis, regime_id, precedence}]
                                 // Per-interval governing_date_kind
     issue_date, upd_datetime,
     field[], organ[],
     label, number, parent_article, parent_clause,
     corpus_version, embedding_version,
     context_text,
     embedding_input_fingerprint,
     
     // Citation projection:
     title, official_url, human_citation, provision_path, snippet
     // NO static applicability_basis. Selected interval/governing_date_kind per request.
  }
```

### INVAR 7 (Eligibility Predicate)
Every Qdrant query includes eligibility predicate checking validity_intervals against governing_dates[interval.governing_date_kind]:
```
ANY interval WHERE:
  interval.from <= governing_dates[interval.governing_date_kind]
  AND (interval.to IS NULL OR interval.to > governing_dates[interval.governing_date_kind])
  AND interval.state == 'in_force'
```
Applied identically to Neo4j BM25 (scoped to release_id + validity BEFORE final LIMIT). Both filtered BEFORE RRF fusion. Nested interval evaluation ensures all fields checked on SAME element.

### BM25 Keyword — Baseline Hypothesis (D11)
BM25 lives in Neo4j fulltext. Baseline hypothesis, validated through quality gate. BM25 MUST filter release_id + validity BEFORE final LIMIT.

---

## 12. Retrieval + Generation (Pillar 3)

### Key Design Decisions
1. **Dual cap**: Max 8 serving contexts AND token budget (default 4096). Expansion counts toward budget.
2. **Decompose contract**: Code enforces max_subqueries=6 + dedup.
3. **Eligibility predicate mandatory**: Every query carries eligibility predicate with governing_dates (pinned release). Per-interval governing_date_kind.
4. **Early-exit patterns**: Cache hit, reject/greeting/clarify, empty retrieval, coverage unsupported.
5. **Context source**: Primary = Qdrant payload context_text (pinned release). Fallback = Neo4j hierarchy traversal (pinned release_id). Both apply same eligibility predicate.
6. **Expansion**: Same eligibility predicate. Expand + dedupe THEN final-select ≤8 + token budget.
7. **Pending-amendment lookup**: After candidates selected, check against pending-amendment index (D73). Hit → review_pending flag + warning + distinct outcome.
8. **Per-request per-context applicability**: Computed fresh using governing_dates[interval.governing_date_kind]. Per-context legal_time_basis carried in SSE sources[].
9. **Citation projection**: Neo4j-down → Qdrant payload + per-request selected interval/governing_date_kind.

### Execution Model (D51)
Bounded thread pool executor with BOUNDED admission queue. GPU semaphore within single API worker process (D27). Per-stage timeout/deadline. On saturation: 429/503 + Retry-After. OOM: graceful degradation to CPU. Cancellation propagated. Streaming accumulator bounded.

### Generator Prompt Wrapper (D47)
Versioned prompt wrapper receives:
- governing_dates dict
- Per-context legal_time_basis (governing_date_kind + governing_date + rule_kind + effective_rule_id)
- Selected intervals/basis per context
- Immutable citations (canonical UIDs + human citations)
- Eligible contexts (filtered by eligibility predicate)
- review_pending flags per context

Constraint: answer ONLY from supplied eligible contexts. Caveated if review_pending. NOT legacy prompt verbatim (legacy biases toward "Ngày hiện tại").

### Text2Cypher (D18, D68)
Typed discriminated request models per template kind with `extra="forbid"`:
```python
class CountArticlesByDocRequest(BaseModel):
    kind: Literal["count_articles_by_doc"]
    legal_document_key: str     # Resolve via legal_document_key, not doc_identity
    class Config:
        extra = "forbid"

class ListSignersOfDocRequest(BaseModel):
    kind: Literal["list_signers_of_doc"]
    legal_document_key: str
    class Config:
        extra = "forbid"

# ... extensible per template

TemplateCall = Annotated[
    Union[CountArticlesByDocRequest, ListSignersOfDocRequest, ...],
    Field(discriminator="kind")
]
```

Server-owned parameterized templates. Each template has:
- Fixed server-side Cypher query (never model-generated)
- Hard LIMIT clause
- Timeout
- Result schema definition
- **Output includes provision_version_id + CitationContext** for citation building
- Validity predicate built-in
- Scoped to pinned release_id
- Resolve via legal_document_key (not doc_identity) with ambiguity rejection

Invalid kind or params → route to retrieval. No legacy raw-Cypher generator reused.

---

## 13. API + SSE Contract

### POST /v1/chat

Request:
```typescript
{
  query: string              // max 2000 chars
  chat_history?: Array<{role: "user"|"assistant", content: string}>  // max 20 turns, each max 4000 chars
  as_of_date?: string        // ISO 8601, default today, Asia/Ho_Chi_Minh
  occurred_at?: string       // CANONICAL (event_date deprecated alias)
  ended_at?: string          // For completed-lasting conduct
  detected_date?: string     // REQUIRED if violation_state=ongoing
  violation_state?: "completed" | "ongoing" | "unknown"
  filters?: {
    doc_type?: string[]
    field?: string[]
    organ?: string[]
  }                          // Max cardinality enforced per field
}
```

Validation: event_date (deprecated alias) → normalized to occurred_at. Both present with different values → 422. After normalization, only canonical fields downstream.

Total request body/history/filter cardinality limited (D51). Validation errors returned as HTTP problem response BEFORE opening SSE stream.

Response: SSE stream via `fetch()` + `ReadableStream`.

### SSE Event Sequence
ALL valid intents follow the same sequence: `meta` → `sources`? → `token*` → exactly one terminal: `done` OR `error`.

```
event: meta
data: {"state": "started", "trace_id": "...", "corpus_version": ...,
       "release_fingerprint": "...", "publish_fence": ...,
       "as_of_date": "...", "occurred_at": "...", "ended_at": "...",
       "detected_date": "...", "violation_state": "...",
       "governing_dates": {...}, "streaming_mode": "streaming|buffered"}

event: sources
data: [{"citation": "Điểm a Khoản 3 Điều 6 NĐ 168/2024/NĐ-CP",
        "title": "...", "official_url": "...",
        "provision_path": ["Điều 6", "Khoản 3", "Điểm a"],
        "selected_interval": {"from": "2025-01-01", "to": null, "state": "in_force"},
        "governing_date_kind": "query_time",
        "applicability_label": "applicable",
        "review_pending": false,
        "snippet": "..."}]

event: token
data: {"text": "..."}

event: done
data: {"trace_id": "...",
       "outcome": "full|degraded|cached|clarified|coverage_unsupported|legal_status_review_pending",
       "degraded_components": [], "warnings": []}

event: error
data: {"code": "...", "message": "...", "trace_id": "..."}
```

Notes:
- `meta` contains ONLY immutable request/release fields and `state:"started"`. Canonical factual fields only.
- `sources[]` items carry SELECTED interval + governing_date_kind computed fresh per request. Include applicability_label + review_pending flag.
- `done` carries final outcome. `legal_status_review_pending` when any context flagged.
- Client disconnect: server cancels + logs. Does NOT emit `done`.
- `streaming_mode`: "streaming" or "buffered". Buffered = exactly one `token` event with full text.
- Validation errors / 429 / 503 BEFORE stream → HTTP problem response. After stream → SSE `error`.
- Clarify intent: one deterministic `token` then `done` with outcome="clarified".

### Cache Write Timing (D34, D65)
- **Answer cache**: Written ONLY after terminal `done` success. Includes retrieval_policy_fingerprint + runtime_policy_fingerprint. NOT written on pre-token failure, mid-stream error, disconnect.
- **Retrieval cache**: Written immediately after retrieval+rerank success, EVEN IF generator later fails.

### POST /v1/feedback
Validates trace ownership. Idempotency key. Returns 200/409. No admin HTTP cache-invalidation endpoint in v1.

### Health Endpoints
/livez (no deps), /readyz (checks active reconciled release + stores + fulltext readiness + init lock absent), /metrics. NO remote LLM call.

---

## 14. Caching (Pillar 4)

Three-layer Redis cache in `cache:` namespace. Keys include full dependency fingerprint. Two distinct lookup points. Write timing locked per layer (D65).

### Cache Lookup Ordering
1. **Answer cache**: After route + rewrite.
2. **Retrieval cache**: After decompose, before search/rerank.

Both AFTER release descriptor pinned and facts normalized.

### Cache Key Design
- **Answer cache**: `cache:ans:{sha256(rewritten_query + intent + filters_canonical + as_of_date + occurred_at + ended_at + detected_date + violation_state + governing_dates_canonical + release_descriptor.fingerprint + embedding_version + reranker_version + model_name + prompt_wrapper_version + retrieval_policy_fingerprint + runtime_policy_fingerprint)}`
- **Retrieval cache**: `cache:ret:{sha256(decomposed_queries_sorted + filters_canonical + governing_dates_canonical + violation_state + release_descriptor.fingerprint + embedding_version + reranker_version + retrieval_policy_fingerprint)}`
- **Context cache**: `cache:ctx:{provision_version_id}:{release_descriptor.fingerprint}` (raw text; per provision_version + release)
- **Amends cache**: `cache:amends:{provision_uid}:{release_descriptor.fingerprint}:{governing_dates_canonical}` (date-resolved result; key includes canonical governing_dates per D77)

Normalization: case-fold + whitespace collapse + unicode NFC. Canonical JSON for filters (sorted keys). NO stopword stripping.

### Layer Semantics (D65)
| Layer | Written When | Contains | TTL |
|---|---|---|---|
| Answer cache | After terminal done SUCCESS only | Answer + sources + intent + retrieval_policy_fingerprint + runtime_policy_fingerprint | Long (24h) |
| Retrieval cache | After retrieval+rerank SUCCESS (even if gen fails) | provision_version_id refs + label + score | Medium (6h) |
| Context cache | After context build success | Raw text per (provision_version_id, release) | Medium |
| Amends cache | After amends resolve (date-resolved) | Per (provision_uid, release, governing_dates) | Medium |

### Invalidation Strategy
Version-based via release_descriptor.fingerprint. Old entries expire via TTL. Negative caching avoided.

### Redis Role Separation (D28)
Cache (cache:) failure → fail-open. Descriptor (descriptor:) failure → durable local fallback. Broker (arq:) failure → ingestion stops.

---

## 15. Graph Schema + Validity

### Node Labels (Hierarchy)
```
Document → Part → Chapter → Section → Article → Clause → Point
```
Levels optional. Parser attaches to nearest present ancestor.

Metadata nodes: DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field. Fields multi-value.

### Node Scoping (D62)
- **Immutable global reference**: Metadata nodes + ExternalDocument. Not release-scoped. Not GC'd with releases. Shared across releases. Separate manifest section.
- **Release-scoped**: Content hierarchy nodes + AMENDS nodes. Keyed by snapshot_node_id. GC'd with release retention.

Relationships from release-scoped content nodes to global metadata nodes are release-scoped. RELATED_TO to ExternalDocument is global-reference exception with canonical edge tuple.

### Identity (see §4)
Neo4j node physical key = snapshot_node_id. Properties: provision_version_id, provision_uid, legal_document_key, release_id, validity_intervals (with governing_date_kind per interval), etc.

Multiple versions per provision_uid allowed within a release. Multiple releases coexist. Nodes distinguished by snapshot_node_id.

### Relationships
```
Hierarchy : HAS_PART / HAS_CHAPTER / HAS_SECTION / HAS_ARTICLE / HAS_CLAUSE / HAS_POINT
Metadata  : BELONGS_TO_GROUP / HAS_TYPE / HAS_STATUS / ISSUED_BY / SIGNED_BY / IN_FIELD
Cross-doc : RELATED_TO  (ExternalDocument stub keyed by source_document_id; global-reference exception)
          : AMENDS {operation, evidence_locator, effective_rule_id, edge_id,
                    extractor_version, prompt_version, confidence, review_status, validity_basis}
```

ALL relationships connect nodes within the SAME release (except content→metadata which connects release-scoped to global, and RELATED_TO to ExternalDocument which is global-reference exception).

### Deterministic Edge Identification (D63, D74)
Canonical edge tuple per relationship type:
- Hierarchy: `(release_id, rel_type, from_snapshot_node_id, to_snapshot_node_id)`
- AMENDS: `(release_id, operation, amending_snapshot_node_id, target_snapshot_node_id, effective_rule_id, edge_id)`
- Metadata: `(release_id, rel_type, from_snapshot_node_id, to_global_id)`
- RELATED_TO ExternalDocument: `(release_id, "RELATED_TO", from_snapshot_node_id, external_source_document_id)` — global-reference exception with canonical tuple for reconcile

Edge digest = SHA256(sorted canonical edge tuples). Reconcile checks node tuples + edge digest + endpoints.

### Constraints
Uniqueness constraint on snapshot_node_id for release-scoped nodes. Uniqueness on (id) for global metadata nodes. All MERGE-based, idempotent.

---

## 16. Monitoring + Evaluation (Pillar 5)

### A. Evaluation Harness (Offline)
Dataset: frozen versioned dataset (`eval_dataset_v1.json`). Each label (D67):
```json
{
  "question": "...",
  "legal_locators": ["Điều 6 Khoản 3 Điểm a NĐ 168/2024/NĐ-CP"],
  "as_of_date": "2026-08-14",
  "occurred_at": "2026-08-14",
  "ended_at": null,
  "detected_date": null,
  "violation_state": "completed",
  "governing_dates": {"conduct_time": "2026-08-14", "query_time": "2026-08-14"},
  "rule_kind": "completed_conduct_time",
  "resolved_provision_version_ids": ["..."],
  "release_fingerprint": "...",
  "reference_answer": "...",
  "notes": "..."
}
```
Labels have `resolved_provision_version_ids` + release fingerprint + canonical facts (occurred_at, ended_at, detected_date, violation_state) + governing dates/rule kind. Deduplicated and group-split by legal_document_key BEFORE 80/20 split.

**Retrieval quality**: Recall@8, Precision@k, MRR, nDCG. Eval retrieval_k separate from serving context_k. Token budget assertion. Run fingerprint includes release_descriptor.fingerprint. Eval relevance uses governing dates/rule kind, not just occurred_at.

**Quality gate**: Recall@8 must not drop more than 2% vs established baseline (absolute threshold). nDCG must not drop more than 3%. Gates promotion/deploy, NOT every PR.

### B. Online Observability
Langfuse (self-hosted, D75) + Prometheus + Grafana. Trace every request. Attach feedback. Infrastructure metrics. Telemetry redaction/retention policy. NO high-cardinality labels.

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
| Neo4j unavailable | Qdrant payload context_text + citation projection. Degraded but functional. |
| Both stores unavailable | Friendly "system temporarily unavailable". |
| Reranker | Use ANN ordering, skip rerank. |
| Generator LLM (pre-token) | Retry once. After exhaustion: friendly error. |
| Generator LLM (mid-stream) | NO retry. Stream truncated. Emit error event. |
| Redis cache (cache:) | Fail-open, skip cache. |
| Redis descriptor (descriptor:) | Durable local fallback. |
| Langfuse/Prometheus | Fail-open. Log locally. |
| Missing facts for sanction/transitional | Clarify intent. No retrieval. |
| Governing dates outside coverage | historical_coverage_unsupported. |
| Admission queue saturated | 429/503 + Retry-After. |
| event_date + occurred_at conflict | 422 Conflict. |
| Pending-amendment lookup hit | legal_status_review_pending outcome + warning. |
| Unsupported temporal evaluator | temporal_rule_unsupported / clarification. |
| INSECURE_TLS=true in production | Startup FAIL. |

### Ingestion Path (Offline)
See §9 error table.

---

## 18. Testing Strategy

### Unit Tests (No Network, No Services)
- Parser hierarchy: alphanumeric clause numbers (2a, 18a). Footer split. Quote-block. NFC. Fixture-based.
- Identity: legal_document_key derivation, provision_uid building, provision_version_id from provision_text_hash, snapshot_node_id derivation, UUIDv5 determinism, round-trip mapping. Resolver ambiguity rejection.
- Source revision identity: parse idempotency changes with raw/parser/normalizer. Manifest path derived from full identity.
- AMENDS discriminated union: valid/invalid schemas per operation type. ADD without target = valid. REPLACE without base hash = quarantine. Patch non-deterministic on base = quarantine. Pydantic one-of/at-least-one validators. EvidenceLocator verification. effective_rule_id validation (exists, reviewed, scope covers target). Edge ID includes evidence identity.
- Validity interval payload: governing_date_kind present per interval. No overlapping applicable intervals per provision/regime. Static general truncated at exception.
- Cache key construction: all dependency dimensions (including release_descriptor.fingerprint, canonical facts, governing_dates, retrieval_policy_fingerprint, runtime_policy_fingerprint). Same inputs → same key. Version bump → key change.
- Eligibility predicate: various validity_intervals combinations, governing_dates[interval.governing_date_kind] boundary cases, half-open [from,to) semantics, nested interval evaluation (all fields on same element), conditional/unknown excluded by default.
- Per-request per-context applicability computation: same stored intervals, different governing_dates → different applicability_labels + legal_time_basis.
- Conditional materialization: conditional → reviewer materializes in_force interval → becomes eligible.
- RRF aggregation.
- Heuristic rerank: recency calculation, post-eligibility application.
- Query normalization for cache (no stopwords, canonical JSON filters).
- Text2Cypher typed TemplateCall: valid param extraction, invalid kind → route to retrieval. Strict discriminator. extra="forbid". Output provision_version_id + CitationContext. Resolve via legal_document_key (not doc_identity).
- Subquery dedup + max_subqueries enforcement.
- Token budget enforcement.
- SSE event serialization. Per-context governing_date_kind in sources.
- requires_clarification classifier.
- Prompt wrapper: governing_dates + per-context legal_time_basis correctly injected.
- ReleaseDescriptor.fingerprint deterministic derivation. publish_fence separate from corpus_version.
- Bounded admission queue: saturation → 429/503.
- event_date normalization: alias → occurred_at. Conflict → 422.
- Embedding_input_fingerprint: model/segmenter/preprocess/dimension change → fingerprint changes.
- Pending-amendment lookup: auto amendment after planned effective date → hit.
- INSECURE_TLS=true in production → startup fail.
- Amends cache key includes governing_dates.

### Integration Tests (Single Docker-Compose Test Profile)
- Import parsed JSON → verify node/relationship counts + constraints hold.
- Batch embed → verify Qdrant point count + payload filter works + context_text populated + citation projection populated + governing_date_kind per interval.
- End-to-end retrieval on fixture: query → expected top_k provision_version_ids match reference.
- Dual-write reconciliation: simulate partial failure → verify no release published. Verify relationship digest + edge endpoints checked.
- Tombstone: REVIEWED REPEAL → validity interval closed, text/evidence preserved.
- Cache invalidation: publish new release → verify old keys miss.
- Atomic publish: verify serving sees consistent state after CAS flip.
- Partial parse: verify last-good kept, no tombstone.
- AMENDS cascade: REVIEWED REPEAL of Article → Clauses/Points intervals closed.
- Eligibility predicate: occurred_at before/after amendment → different results.
- Staging N doesn't affect serving N-1 (snapshot isolation).
- Unchanged corpus carry-forward (no-op cron doesn't bump version).
- Rollback: CAS back to previous release with new publish_fence → serving reverts correctly. No cross-release leakage. Stale build cannot re-publish after rollback (fence moved forward).
- Same-day replacement boundary ([from, to) half-open semantics).
- Multi-interval predicate (nested evaluation, all fields on same element, no cross-filtering).
- NĐ 168: general effective 2025-01-01, exception 2026-01-01 (truncated general), conditional, completed-vs-ongoing/detected-date.
- Static exception no overlap: exception provisions do NOT have general [2025-01-01,∞) interval.
- Predecessor/successor temporal regime: NĐ 168 transition to NĐ 100 traced via regime_id.
- ADD_PROVISION successor creation.
- REPLACE_TEXT successor-before-predecessor-close.
- Metadata-only revision (different source_metadata_revision_hash, same raw_blob_hash). New manifest created.
- Normalizer revision (different normalized_text_hash + normalizer_version). New manifest. Evidence offsets bound correctly.
- Raw body changed but metadata same → new source_revision_identity → new manifest.
- Degraded Qdrant citations (Neo4j down → sources[] from payload with governing_date_kind).
- Redis descriptor outage → durable fallback serves last pinned release.
- Fresh volume bootstrap/restore → /readyz returns 503 corpus_not_ready. Bootstrap exclusive init (only one worker initializes).
- Two-container GPU contention (API rerank GPU + worker embed CPU → no contention). Single API worker process confirmed.
- BM25 release isolation: N-1 results don't crowd out N results.
- Fulltext index readiness checked in reconcile.
- Expansion applies eligibility predicate + final cap (≤8 + token budget).
- Generator prompt wrapper: historical governing_dates + per-context legal_time_basis produces historically-grounded answer.
- Future-effective act ingested before effective date → intervals with future from; eligibility excludes for current governing_dates.
- Historical cache/context no collision between two provision versions (keys include provision_version_id + release fingerprint).
- Completed vs ongoing violation/detected_date → different governing_dates derivation.
- Interval before/on/after repeal boundary.
- Failed/disconnected stream NOT cached.
- Graph relationship reconcile catches missing relationships + edge digest mismatch.
- Clarify intent: missing facts → one deterministic token + done, no retrieval.
- Coverage unsupported: governing_dates outside window → distinct response.
- Concurrent build: two workers build simultaneously → only one CAS succeeds; other detects stale, discards.
- Stale publish: build with old base_fingerprint/base_fence → CAS returns 0, discarded.
- Stale build after rollback: old fingerprint matches but fence doesn't → CAS returns 0.
- Equal-base fingerprint: concurrent builds with same base → only higher fence wins.
- Interrupted CAS: crash between CAS and local file → restart from Redis authority.
- Redis restart: descriptor survives (AOF/RDB). Local file fallback if Redis down during restart.
- Manual rollback: CAS to previous with new fence, audit log recorded.
- Model revision change → embedding_input_fingerprint changes → vectors re-embedded, not reused.
- Auto amendment after planned effective date → pending-amendment lookup hit → legal_status_review_pending outcome + warning, not "not found" or confident old-law answer.
- Metadata/external graph scope: metadata nodes immutable global, not GC'd. RELATED_TO external edge reconcile + backup.
- Edge-digest reconciliation: corrupted edge detected, publish blocked.
- Answer cache includes retrieval_policy_fingerprint + runtime_policy_fingerprint.
- Retrieval cache written after retrieval+rerank success despite gen failure.
- Parser/normalizer/raw revision: new source revision identity → new manifest, never overwrite.
- event_date alias/conflict validation: alias normalized; conflict → 422.
- Conditional reviewed materializes in_force interval: conditional → eligible after review.
- AMENDS effective_rule vs doc metadata: effective_rule_id links to ledger, not doc effectDate. Validation checks existence, review status, scope.
- Per-context governing_date_kind: different contexts use different governing dates; SSE sources carry per-item governing_date_kind.
- Amends cache date-resolved: key includes governing_dates.
- Temporal rule unsupported: provision requiring evaluator outside enum → temporal_rule_unsupported / clarification.
- INSECURE_TLS=true in production → startup fail; dev profile allows.

Never touch production data. Isolated test database/collection. Placeholder credentials only. Demo/live-service scripts NOT in unit suite.

### Acceptance Tests (Concrete Scenarios)
| Scenario | Expected |
|---|---|
| Before/after amendment | Same legal_locator with occurred_at before/after → different applicable provision |
| Event_date transition | NĐ 100 vs NĐ 168 for same violation → correct provision by occurred_at |
| Amendment adds 2a | ADD_PROVISION with numbering `2a` → retrievable, cited correctly |
| Repeal cascades to children | REVIEWED REPEAL of Article → Clauses/Points intervals closed |
| Partial-ingest not exposed | Truncated/bad fetch → last-good kept, no tombstone |
| Stale payload after amendment | Payload/context hash recomputed, vector reused if embedding_input_fingerprint matches |
| Truncated parser no tombstone | Parse-quality gate fails → last-good retained |
| SSE disconnect | Server cancels, no done emitted, resources freed, NOT cached |
| Cache-hit fresh trace | New trace_id, cache_origin_trace_id recorded |
| Fresh-volume bootstrap | Bootstrap from backup or re-ingest produces consistent serving; exclusive init |
| Staging isolation | During ingest N, serving continues on N-1 |
| Rollback no cross-release | CAS back with new fence → all reads revert; stale build can't re-publish |
| Same-day boundary | occurred_at = effective_date → [from,to) selects NEW provision |
| Multi-interval | Disjoint active intervals → predicate evaluates each independently on same element |
| NĐ 168 conditional | conditional state → excluded, warning returned |
| NĐ 168 exception 2026 | STATIC_EXCEPTION effective_from=2026-01-01; general truncated; no overlap |
| NĐ 168 completed vs ongoing | completed → conduct_time; ongoing → detection_time |
| NĐ 168 predecessor/successor regime | Transition traced via regime_id |
| Degraded citations | Neo4j down → sources[] from Qdrant payload with governing_date_kind |
| Redis descriptor outage | Serve last pinned from local fallback |
| Metadata-only revision | New metadata hash, same blob hash → new manifest |
| Normalizer revision | New normalized artifact, evidence offsets bound correctly |
| Raw body changed, metadata same | New source_revision_identity → new manifest |
| Two-container GPU | API GPU + worker CPU → no contention; single worker process |
| N+N+1 coexist | Both releases queryable independently |
| Historical cache no collision | Two provision versions → distinct cache keys |
| Completed vs ongoing | Different governing_dates derivation |
| Future-effective ingest | Ingested before effective date; excluded by predicate until effective |
| Clarify path | Missing facts → one token + done, no retrieval |
| Coverage unsupported | Outside window → distinct response |
| BM25 release isolation | N-1 doesn't crowd out N |
| Expansion final cap | Expand + dedupe → ≤8 contexts + token budget |
| Generator historical governing_dates | Historically-grounded answer with per-context basis |
| Failed stream not cached | Mid-stream error → answer NOT written to cache |
| Relationship reconcile | Missing relationships detected, publish blocked |
| Fulltext readiness | Fulltext index not ready → reconcile fails |
| Concurrent build | Only one CAS succeeds; other stale-detected |
| Stale publish | Old base_fingerprint/base_fence → CAS returns 0, discarded |
| Stale build after rollback | Old fingerprint matches but fence doesn't → CAS returns 0 |
| Equal-base fingerprint | Higher fence wins |
| Interrupted CAS | Crash between CAS and local file → restart from Redis authority |
| Redis restart | Descriptor survives; local fallback if Redis down |
| Manual rollback | CAS to previous with new fence, audit log |
| Model revision change → re-embed | embedding_input_fingerprint changes → no vector reuse |
| Auto amendment after effective date | Pending-amendment lookup hit → legal_status_review_pending + warning |
| Metadata/external scope | Immutable global, not GC'd; RELATED_TO external reconcile |
| Edge-digest reconcile | Corrupted edge detected |
| Answer cache policy fingerprint | Includes retrieval + runtime policy fingerprint |
| Retrieval cache on gen failure | Written after retrieval+rerank success despite gen failure |
| Parser/normalizer revision | New source revision identity → new manifest |
| event_date alias/conflict | Alias normalized; conflict → 422 |
| Conditional reviewed materializes | in_force interval created → becomes eligible |
| AMENDS effective_rule vs doc metadata | effective_rule_id validated; links to ledger |
| Per-context governing_date_kind | Different contexts → different governing dates in SSE sources |
| Amends cache date-resolved | Key includes governing_dates |
| Temporal rule unsupported | Outside enum → temporal_rule_unsupported / clarification |
| INSECURE_TLS production fail | Production startup fails; dev profile allows |
| Static exception no overlap | Exception provisions don't have general interval |

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
Root repo vendors `NLP-LegalQA` as git submodule at pinned SHA. Read-only reference.

Delivery acceptance criteria: `.gitmodules` committed. Gitlink committed. CI checkout recursive. CI asserts SHA.

```
Agentic/
├── .gitmodules
├── NLP-LegalQA/                         # Git submodule (READ-ONLY REFERENCE)
├── src/legal_rag/
│   ├── identity.py                      # §4
│   ├── temporal.py                      # §5
│   ├── release.py                       # §8 (descriptor + CAS publish)
│   ├── ingestion/                       # OWNERSHIP: serving corpus (D53)
│   │   ├── parser.py                    # Alphanumeric clause, appendix preserve
│   │   ├── effectivity.py               # Effectivity ledger extraction
│   │   └── amends.py                    # Discriminated union extraction
│   ├── embeddings/                      # §11
│   ├── retrieval/                       # §12
│   ├── templates/                       # §12 (typed TemplateCall)
│   ├── cache/                           # §14
│   ├── api/                             # §13
│   ├── sse/                             # §13
│   ├── prompts/                         # §12 (versioned prompt wrapper)
│   └── config/                          # Allow-list, constants, namespaces
├── frontend/
├── tests/{unit,integration,acceptance}/
├── eval/
├── scripts/
├── docker-compose.yml
├── docker-compose.dev.yml               # Dev profile (allows INSECURE_TLS=true)
├── docker-compose.test.yml
├── Dockerfile
├── uv.lock
├── package-lock.json
└── docs/superpowers/specs/
```

### Pinning
Everything pinned: Python version, Node version, base Docker image digests, service image tags, HuggingFace model revisions, submodule SHA. All fingerprints in corpus manifest + eval manifest + release descriptor.

### Langfuse Self-Host Spec (D75)
Deployed via dedicated compose file or profile with:
- **Pinned versions**: langfuse image tag, postgres, clickhouse, minio all pinned.
- **Persistent volumes**: postgres_data, clickhouse_data, minio_data. Named volumes, backed up.
- **Backup/restore**: pg_dump for postgres, clickhouse backup tool, minio mc mirror. Documented procedure.
- **Resource profile**: Min 4GB RAM, 2 CPU cores, 20GB disk for Langfuse stack. Documented.
- **Secrets**: via env file, not hardcoded. NEXTAUTH_SECRET, SALT, encryption keys generated.
- **Network**: internal network, not exposed publicly in v1.

Not just a link to external documentation. Concrete deployment specification.

### INSECURE_TLS Enforcement (D76)
Application startup checks `INSECURE_TLS` environment variable. If `true` AND running in production profile → startup FAILS with clear error message. Only allowed in dev/test profile (docker-compose.dev.yml). This prevents accidental insecure connections in production.

### GPU Profile (D27)
Two containers (API + arq-worker). V1: **single API worker process** (uvicorn workers=1) so in-process GPU semaphore suffices for coordinating reranker access. If scaling to multiple API workers in future, require Redis distributed lease for GPU coordination (out of v1 scope, documented as upgrade path).

Default: API reranker GPU, ingestion embedder CPU. No contention. Optional off-peak GPU scheduling for worker. OOM: graceful degradation to CPU.

### Volumes, Bootstrap, Backup (D42)
| Volume | Content | Backup Strategy |
|---|---|---|
| `neo4j_data` | Graph + constraints | neo4j-admin dump/restore |
| `qdrant_storage` | Per-release collections | Qdrant snapshot API per collection |
| `redis_data` | Cache + arq queue + descriptor Hash + fence counter | Redis RDB/AOF backup |
| `raw_corpus` | Immutable provenance-separated raw store | File-level backup |
| `manifest_store` | Ingestion manifests + release manifests + active_release.json + review_log | File-level backup |
| `quarantine_store` | Quarantined AMENDS edges | File-level backup |
| `langfuse_postgres` | Langfuse metadata | pg_dump |
| `langfuse_clickhouse` | Langfuse analytics | clickhouse backup |
| `langfuse_minio` | Langfuse blob storage | minio mc mirror |

One common `backup_manifest`. Restore consistent release set. Cache/arq ephemeral.

### Open Items
1. **Exact allow-list IDs**: Live API discovery. Seeded from NLP-LegalQA search params.
2. **Langfuse resource budget**: Specific profile documented; actual allocation depends on host machine.
3. **HuggingFace model download**: First-time setup. Cached in volume.
4. **UpdDateTime capability probe**: Probed at first ingestion; fallback activated if unsupported.
5. **Multi-worker GPU coordination**: Out of v1 scope. Requires Redis distributed lease when scaling beyond single API worker.

Note: Not all design decisions are finalized until all new invariants are tested via acceptance tests. This spec represents the current design target; implementation may reveal additional refinements.

---

## Changelog: v6 → v7

| Area | v6 Issue | v7 Fix |
|---|---|---|
| Validity interval payload | Missing governing_date_kind per interval | Added governing_date_kind enum (query_time/conduct_time/detection_time) to each interval element. Predicate checks all fields on SAME nested element. |
| Static general + exception overlap | Could create overlapping in_force intervals | Static general truncated/split at exception dates during materialization (D70). No overlap invariant. |
| Per-context legal_time_basis | Single global legal_time_basis for entire request | Per-context legal_time_basis: {governing_date_kind, governing_date, rule_kind, effective_rule_id}. SSE sources carry governing_date_kind per item. Prompt wrapper receives per-context basis. (D69) |
| review_pending warning | Excluded from eligibility → "not found" instead of warning | Pending-amendment lookup (D73): index of auto/quarantined edges with planned effective_date <= governing_date. Checked during retrieval. Hit → legal_status_review_pending outcome + warning. Not confident old-law answer. |
| Publish fencing | CAS only checked fingerprint, not fence | Separated corpus_version from publish_fence (strictly monotonic). CAS checks BOTH fingerprint AND fence. New fence > current fence required. (D61) |
| Stale re-publish after rollback | Old fingerprint could match after rollback | Rollback allocates new publish_fence. Stale builds can't re-publish because base_fence won't match. |
| CAS implementation | Used json_extract (may not exist in standard Redis) | Redis Hash (HGET/HSET) for descriptor storage. Lua script uses HGET. cjson.decode noted as alternative. |
| Bootstrap race | Two workers could publish simultaneously | Exclusive init via SETNX descriptor:init_lock with TTL. Only holder writes initial descriptor. CAS returns -1 on contention. |
| Manifest path | Missing transform identity components | Path derived from full source_revision_identity_hash (includes raw_blob_hash, normalizer_version, parser_version). Each transform revision = separate immutable manifest. (D66) |
| effective_rule_id validation | Not validated | Validator checks: exists in ledger, correct source/release, reviewed, scope covers target. (D59) |
| PatchSpec validation | expected_successor not required | model_validator requires expected_successor_hash OR expected_successor_text. |
| Template resolve | Could use doc_identity (collides) | Resolve via legal_document_key with ambiguity rejection. Template models extra="forbid". (D68) |
| AMENDS edge identity | Two amendments could MERGE | edge_id = SHA256(operation + endpoints + effective_rule_id + excerpt_hash). Distinct edges. (D74) |
| RELATED_TO external | Not explicitly scoped | Global-reference exception with canonical edge tuple, reconcile, backup semantics. (D62/D25) |
| Cache amends | Value type ambiguous | Date-resolved result. Key includes canonical governing_dates. (D77) |
| Eval labels | Missing facts/governing dates | Labels include occurred_at, ended_at, detected_date, violation_state, governing_dates, rule_kind. (D67) |
| INSECURE_TLS | Dev override could leak to production | Production startup FAILS if INSECURE_TLS=true. Only dev profile allows. (D76) |
| GPU profile | Semaphore insufficient for multiple workers | V1: single API worker process. Multi-worker requires Redis distributed lease (documented as upgrade path). (D27) |
| Langfuse self-host | Just external link | Compose closure, persistent volumes, backup/restore, resource profile specified. (D75) |
| Finality claim | "All design decisions finalized" | Removed. Not all invariants tested yet. |
| Coherence pass | Various remnants | Removed: ANY interval without governing_date_kind; static exception overlapping general; review_pending only annotation; publish_fence unchecked; manifest path missing transform identity; template resolve by doc_identity; legal_time_basis without schema. |
