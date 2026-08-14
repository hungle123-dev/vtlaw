# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-14  
**Status:** Draft v3 (post-Codex round-2 revision)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (vendored as git submodule at pinned SHA). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Code and infrastructure are production-shape; deployable to cloud if desired.
- Right-sized for a bounded corpus (Vietnamese traffic-law documents, thousands of chunks). No fake scale claims.
- Legally sound validity model at provision level with explicit temporal semantics (`as_of_date`, `event_date`).
- Atomic publish of corpus snapshots — serving path never sees partial state.

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
| D1 | Reality bar | Self-host docker-compose, production-shape | Right-sized for portfolio; no cloud cost required |
| D2 | Feature scope | Core RAG + amendment graph + chat/routing/template-Cypher + eval/monitoring. No fine-tune | Covers all 5 pillars without research-artifact bloat |
| D3 | Data store | Neo4j (graph + BM25) + Qdrant (ANN + metadata + context snapshot) | Separation matches "metadata-first + dedicated ANN"; each store does one job |
| D4 | LLM provider | Provider-neutral OpenAI-compatible (`base_url`/`model`/`api_key`) for ALL LLM calls including AMENDS extraction and Text2Cypher template selection | Works with OpenRouter/local vLLM/Ollama/Gemini; same factory everywhere |
| D5 | Monitoring | Langfuse (self-hosted per official compose) + Prometheus + Grafana | Langfuse = LLM trace + feedback + eval experiment; Prometheus/Grafana = infra metrics + alerting |
| D6 | UI | Next.js/React with SSE streaming via fetch() ReadableStream | Production-feel frontend; POST JSON body incompatible with native EventSource |
| D7 | Ingestion queue | arq (async Redis queue) | Async-native, built-in cron + retry, single worker. Right-sized vs Celery |
| D8 | Cache | Redis (answer + retrieval + context), key includes full dependency fingerprint | Pillar 4 controller for cost and latency |
| D9 | Service topology | Modular monolith + infra sidecars | One FastAPI app with clear modules + separate containers for stores/observability/UI |
| D10 | Vector storage | Qdrant only; drop Neo4j vector index | Avoids duplicate embed; Neo4j becomes pure graph + keyword |
| D11 | BM25 placement | Neo4j fulltext (Lucene) as baseline hypothesis | Baseline config validated through quality gate, not assumed superior |
| D12 | Corpus scope | Versioned allow-list with specific IDs/criteria; seeded from NLP-LegalQA params | Bounded, well-defined, auditable |
| D13 | TLS verification | Default ON. Dev-only insecure override via `INSECURE_TLS=true` env flag | Production default must verify credentials |
| D14 | Chunk limit | Max 8 serving contexts AND token budget (default 4096); expansion counts toward budget | Dual cap prevents overflow; eval_k separate from serving_k |
| D15 | Identity scheme | Five-tier identity (see §4); content hash NOT in stable legal locator/provision UID | docIdentity collides; stable legal keys needed for AMENDS/citations/QA labels |
| D16 | Qdrant point ID | UUIDv5(fixed_namespace, provision_version_id) | Qdrant requires uint64/UUID; canonical UID preserved in payload |
| D17 | Temporal model | `as_of_date` + `event_date` + `reference_date`; provision-level validity states {active, inactive, conditional, unknown} | Penalty questions require event_date; conditional provisions don't get fake dates |
| D18 | Text2Cypher v1 | Server-owned read templates only; LLM selects operation + params. Mandatory LIMIT, validity predicate, canonical UID return | Eliminates Cypher injection risk; ungrounded free Cypher blocked |
| D19 | SSE contract | Event types: meta, sources, token, done, error. Exactly one terminal. Via fetch() ReadableStream | Explicit protocol; disconnect = cancel/log, no post-disconnect emit |
| D20 | Secrets policy | Env vars only, `.env` gitignored, test placeholders, docs use `<placeholder>` | Credential safety |
| D21 | Atomic publish | Staging writes to both stores with corpus_version=N, reconcile, then atomic pointer flip in Redis. Rollback = flip back. GC after retention. | No half-visible state during ingest |
| D22 | AMENDS operations | Minimal set: ADD_PROVISION, REPLACE_TEXT, REPEAL, AMEND. LegalLocator output from LLM, server resolver maps to internal IDs. review_status=auto = annotation only. | Safe semantics; auto edges don't change eligibility |
| D23 | Clause numbering | Parser accepts alphanumeric clause numbers (`\d+[a-z]?` e.g., `2a`, `18a`) | Raw data contains alphanumeric clauses; pure-numeric regex loses them |
| D24 | Appendix handling | Index if contains normative legal content; exclude if purely administrative | Phụ lục can contain binding legal rules (penalty schedules, lists) |
| D25 | External references | docListOther entries outside corpus → ExternalDocument citation stubs (not retrievable, not validity-bearing) | Prevents treating referenced-but-unscraped docs as corpus members |
| D26 | Health endpoints | Split /livez (no deps), /readyz (deps reachable, no LLM), /metrics. No remote LLM call in probes. | Probes fast + cheap; don't burn LLM quota |
| Caching ordering | Route + rewrite BEFORE cache lookup; direct_answer only for deterministic greeting/help | Cache key depends on rewritten query + intent; prevent ungrounded legal answers |
| Context cache | Store raw context text; decorate validity labels per-request based on reference_date | Date-dependent labels invalidate cache otherwise |
| Retry generation | Only before first token streamed; once streaming started, no retry | Post-stream retry produces garbled/duplicate output |
| GPU concurrency | Semaphore serializes reranker + ingestion embedder; CPU fallback baseline | Single GPU shared safely between API + worker |
| Redis dual-role | Single instance, persistence AOF ON, maxmemory-policy=noeviction, separate key namespaces (arq:/cache:) | Broker outage ≠ cache miss; noeviction protects arq queue integrity |
| Eval dataset | Frozen versioned dataset with legal_locator + dates + resolved UIDs per corpus release. Held-out 80/20 stratified. | Reproducible eval tied to corpus releases |
| Quality gate | Recall@8 + nDCG baseline + tolerance (e.g., ≤2% drop). Blocks deploy, not every PR. LLM-judge nightly/manual. | Practical gate that catches regressions without blocking iteration |

---

## 3. Architecture Overview

### Service Map
```
┌─────────────────────────── docker-compose ───────────────────────────┐
│                                                                      │
│  FRONTEND                                                            │
│  ├─ nextjs          : React chat UI, dark theme,                     │
│  │                   SSE via fetch() ReadableStream,                  │
│  │                   citation panel, feedback button                 │
│                                                                      │
│  APP (modular monolith, 1 Python image)                              │
│  ├─ api             : FastAPI — validate→trace→dates→route→rewrite   │
│  │                   →cache→retrieve→rerank→gen                      │
│  │                   Endpoints: POST /v1/chat, POST /v1/feedback,     │
│  │                   GET /livez, GET /readyz, GET /metrics            │
│  └─ arq-worker      : ingestion offline (scrape→parse→amends→embed   │
│                      →import→reconcile→publish). Same image.         │
│                                                                      │
│  DATA STORES                                                         │
│  ├─ neo4j           : graph hierarchy Điều→Khoản→Điểm + AMENDS        │
│  │                   + Lucene fulltext BM25                           │
│  │                   + Text2Cypher read-only principal                │
│  ├─ qdrant          : ANN dense vector + payload metadata filter     │
│  │                   + context-ready text snapshot                    │
│  └─ redis           : arq broker (arq:* namespace) +                 │
│                       cache (cache:* namespace) +                     │
│                       active_corpus_snapshot pointer                  │
│                       Config: AOF ON, noeviction                     │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : LLM trace, feedback score, eval experiment     │
│  │                   (official self-host compose profile)             │
│  ├─ prometheus      : infra metrics, scrapes GET /metrics            │
│  └─ grafana         : dashboard latency, cache hit, DB health        │
└──────────────────────────────────────────────────────────────────────┘
```

Langfuse deployed per https://langfuse.com/self-hosting/ with pinned versions, persistent volumes, secrets via env. Resource budget documented.

### Request Path (Hot Path, Pillar 3)
```
POST /v1/chat {query, chat_history?, as_of_date?, event_date?, filters?}
  │
  ▼ validate input (schema, lengths, date format ISO, timezone Asia/Ho_Chi_Minh)
  ▼ create fresh trace (new trace_id for EVERY request, including cache hits)
  ▼ resolve dates:
     reference_date = event_date if provided else as_of_date (default today)
     legal_time_basis = "event" | "current"
     If penalty question lacks event_date and answer may differ by date
       → respond conditionally or ask clarification
  ▼ route (LLM, temp=0, max_tokens=64)
     intent ∈ {greeting, cypher_query, retrieve, reject}
  │
  ├─ reject → template refusal, write trace, return
  ├─ greeting → deterministic response (no LLM, no retrieval), write trace, return
  ├─ cypher_query → rewrite if history → select template + params (LLM)
  │   → execute read template (read-only principal, LIMIT, validity predicate)
  │   → gen natural language from template results → write cache + trace → stream
  │
  └─ retrieve:
       ▼ rewrite (multi-turn; skip if no history)
       ▼ CACHE LOOKUP (after route+rewrite):
          key = sha256(rewritten_query + intent + filters + reference_date
              + corpus_version + embedding_version + reranker_version
              + model_name + prompt_version)
          HIT → new trace (cache_origin_trace_id recorded), skip search+rerank+gen
          MISS → continue below
       ▼ decompose (LLM, JSON sub-queries, append original;
          code enforces max_subqueries=6 + dedup by normalized string)
       ▼ multi_search (sub-queries parallel):
          Each sub-query:
            • Qdrant: eligibility predicate (reference_date) + metadata filter
              → HNSW within eligible set
            • Neo4j: BM25 fulltext with same eligibility predicate
          Fuse RRF per sub-query → aggregate across sub-queries
       ▼ fetch context from Qdrant payload context_text (primary)
          or Neo4j hierarchy traversal (fallback)
       ▼ cross-encoder rerank (GPU semaphore, FP16, batch_size small)
       ▼ heuristic rerank (recency bonus AFTER eligibility gate, eval-gated)
       ▼ select top serving contexts (max 8 AND token budget 4096)
       ▼ expand context (optional: sibling Points, Article children;
          expansion counts toward token budget)
       ▼ decorate validity labels per-request using reference_date
          (NOT cached — computed fresh from eligibility state)
       ▼ build context_str (tag [ĐÃ BÃI BỎ]/[SỬA ĐỔI] per reference_date,
          amends note, provisional annotations for auto-review-status edges)
       ▼ gen (LLM, _QA_FEW_SHOT_SYSTEM_PROMPT)
          Retry ONLY before first token. Once streaming starts, no retry.
       ▼ write back cache (answer + retrieval)
       ▼ stream SSE events: meta → sources → token* → done|error
       ▼ write trace to Langfuse + metrics to Prometheus
```

### Ingestion Path (Offline, Pillar 1) — Fully Separate from Request Path
```
cron/manual trigger → arq enqueue
  ▼ scrape phapluat.gov.vn (versioned allow-list, delta via updDateTime
    + content_hash + overlap window). TLS verify ON (dev override available).
    Throttle 0.5–1s between requests.
    Save raw immutable: HTML + JSON + normalized text + official URL
    + fetched_at + parser/normalizer version + evidence offsets.
    Content-addressed store (path = content_hash). Never overwrite existing.
  ▼ parse hierarchy (regex accepts alphanumeric clause numbers \d+[a-z]?)
    + metadata + NFC normalize. Parse-quality gate:
    raw length sanity, anchor presence (Điều markers), hierarchy count/coverage.
    If partial/bad → keep last-good revision, do NOT tombstone.
    Appendix indexed if normative legal content; excluded if administrative.
  ▼ amends extract (LLM outputs LegalLocator, NOT internal source_key).
    Server resolver maps LegalLocator → provision_uid.
    Schema-validate structure (Pydantic). Enum normalizer for amend_type.
    ADD_PROVISION target-not-found = valid (not quarantine).
    Other resolution failures → quarantine edge (annotation, review_status=auto).
    Document ingest continues regardless of AMENDS outcome.
  ▼ embed batch (vietnamese-bi-encoder + pyvi tokenize, GPU semaphore).
    Record segmenter_version + model_revision. Skip unchanged content-hash.
    Reuse vector if text unchanged, recompute payload/context/derived-state hash.
  ▼ import dual-write:
    Neo4j: MERGE nodes keyed by (provision_uid, corpus_version=N).
    Relationships within version N.
    Qdrant: upsert points (UUIDv5(ns, provision_version_id)) with full payload
    + context_text snapshot.
    Materialize validity: REVIEWED AMENDS cascade REPEAL to children.
    Validity basis recorded per provision.
  ▼ reconcile: verify both stores have expected node/point counts for version N.
    Detect discrepancies → DO NOT publish. Alert. Manual intervention.
  ▼ tombstone/delete: remove provisions absent after reparse
    (only if parse-quality gate passed — partial parse keeps last-good).
  ▼ publish: SET active_corpus_snapshot = N in Redis (atomic pointer flip).
    Only if reconcile passed AND changes exist.
    No-op cron (zero changes) does NOT bump version.
  ▼ garbage collect: delete corpus_version < N - retention (default retention=2).
  ▼ write ingestion_manifest, emit metrics.
```

---

## 4. Identity Model

Five distinct identifiers. Content hash NEVER appears in stable legal locators used by AMENDS, QA labels, or citations.

| Identifier | Format | Stability | Purpose |
|---|---|---|---|
| `source_document_id` | docGUId (UUID from API) | Immutable per source artifact | Identifies the downloaded document file. Never changes. |
| `source_revision_id` | SHA256(normalized_content) | Changes when content changes | Identifies a specific content version of a document. Used for delta detection, idempotent embed, audit trail. |
| `legal_locator` | Human citation: `"Điều 5 Khoản 2a Điểm b Nghị định 168/2024/NĐ-CP"` | Stable across content revisions | What humans and legal professionals cite. Used in LLM prompts, user-facing output. NOT machine-parseable key. |
| `provision_uid` | `{doc_identity}::article::{n}::clause::{n}::point::{letter}` | Stable across content revisions | Machine-stable key for a legal provision independent of content version. Neo4j node property. AMENDS target/source. QA label reference. Citations. Does NOT include content hash. |
| `provision_version_id` | `{provision_uid}::rev::{source_revision_id[:12]}` | Changes when provision text changes | Identifies a specific content version of a provision. Embedded into Qdrant. Qdrant point ID = UUIDv5(NAMESPACE, provision_version_id). |

### Mapping Round-Trip
```
provision_version_id  →  UUIDv5(NAMESPACE, provision_version_id)  →  Qdrant point id
Qdrant payload.provision_uid  ←→  Neo4j node.provision_uid
Neo4j node.provision_uid  →  legal_locator (via stored properties + parent traversal)
legal_locator  →  provision_uid (via resolver: doc_identity + article/clause/point lookup)
```

NAMESPACE is a fixed project-specific UUID constant committed to code (e.g., `uuid.UUID("a1b2c3d4-...")`).

### docIdentity Collision Handling
The source API's `docIdentity` can collide across different source versions/amendments (confirmed in NLP-LegalQA CLAUDE.md line 155 and raw data). Therefore:
- `source_document_id` (docGUId) is the true internal document key.
- `doc_identity` is a display/legal-citation property, NOT a primary key.
- Multiple source artifacts may share a doc_identity. They are distinguished by source_document_id and source_revision_id.
- Provision UIDs derive from doc_identity (for legal citation stability) but uniqueness is guaranteed because provisions within the same doc_identity but different source revisions represent genuinely different legal states (e.g., pre/post-amendment versions of the same provision).

---

## 5. Temporal Model

Three temporal parameters govern what the system serves. All dates resolved once at ingress to `Asia/Ho_Chi_Minh` timezone, validated as ISO 8601.

| Parameter | Meaning | Default |
|---|---|---|
| `as_of_date` | The point in time we're asking about the state of the law | Today |
| `event_date` | When the factual event (violation) occurred | None (optional) |
| `reference_date` | Which date drives eligibility: `event_date` if provided, else `as_of_date` | Derived |
| `legal_time_basis` | Annotation: `"event"` (user provided event_date) or `"current"` (using as_of_date) | Derived |

### Why event_date Matters
Penalty questions ("vượt đèn đỏ phạt bao nhiêu") depend on which law was in force WHEN THE VIOLATION OCCURRED. NĐ 100/2019 governed until 2024; NĐ 168/2024 took effect 2025. Asking about a 2023 violation under today's law gives wrong penalties. The system must apply the law in force at event_date.

If a penalty question lacks event_date and the answer differs by era, the system MUST either:
1. Ask for clarification ("Hành vi xảy ra khi nào?"), OR
2. Answer conditionally ("Theo NĐ 168/2024 (hiện hành): X. Nếu hành vi xảy ra trước 01/01/2025 thì theo NĐ 100/2019: Y.")

Never silently assume current law for penalty questions without noting the assumption.

### Validity States
Each provision carries validity state:
- `active`: In force at reference_date. Served normally.
- `inactive`: Not in force at reference_date (expired, repealed, superseded). Excluded from retrieval unless explicitly queried historically.
- `conditional`: Effectiveness depends on an external condition ("có hiệu lực khi Chính phủ ban hành nghị định hướng dẫn"). NOT forced to a fake date. Served with annotation "[Hiệu lực có điều kiện — chưa xác định ngày cụ thể]" and included with caveat.
- `unknown`: Insufficient information to determine. Served with annotation.

Multiple intervals supported: a provision can be active 2020-01-01 to 2024-12-31, inactive, then re-enacted. Stored as list of `{from, to, status}` intervals. Outer bound from document `expireDate` (if non-null, all provisions inactive after expireDate).

### Eligibility Predicate
Applied uniformly to Qdrant (payload filter), Neo4j BM25 (WHERE clause), Text2Cypher templates (built-in), context expansion, and all child node queries:
```
provision has interval where:
  interval.from <= reference_date
  AND (interval.to IS NULL OR interval.to >= reference_date)
  AND interval.status IN ('active', 'conditional')
```
`conditional` provisions pass the gate but carry annotation. `inactive` and `unknown` excluded from standard retrieval.

### Propagation
Resolved `reference_date` and `legal_time_basis` flow through: retrieval eligibility, BM25 filter, Text2Cypher template execution, context building, generation prompt, citations (show validity window), cache key, SSE meta event.

---

## 6. Corpus Scope and Allow-List

### Versioned Allow-List
Corpus scope defined by `allowlist_v1.yaml` (committed, versioned):
```yaml
version: 1
source: phapluat.gov.vn
doc_group_ids: [...]        # Seeded from NLP-LegalQA search params
field_ids: [...]            # Giao thông đường bộ field IDs
keyword_patterns:
  - "giao thông"
  - "trật tự.*an toàn.*giao thông"
  - "xử phạt.*giao thông"
effect_status_include:
  - "Còn Hiệu lực"
  - "Hết Hiệu lực"          # Historical laws needed for temporal queries
discovery:
  pagination: rowAmount=100, iterate pageIndex
  delta_fallback: scan updDateTime range when search returns empty
inclusion_rules:
  - Must match at least one keyword_pattern AND one field_id
exclusion_rules:
  - Administrative circulars without normative content
```

Allow-list updated manually when corpus scope expands. Version tracked in ingestion manifest.

### Discovery and Pagination
Initial population: paginate through search API with allow-list filters. Delta detection: compare `updDateTime` + content_hash against last-ingested manifest + overlap window (catches retroactive updates). Fallback: if search returns zero results unexpectedly, scan by updDateTime range.

---

## 7. Raw Source Storage

Immutable content-addressed store. Path structure: `raw/{content_hash}/`.

Each raw entry contains:
- `original.html`: Full HTML from `/detail?tabName=noidung`
- `metadata.json`: Full JSON from `/detail?tabName=tomtat`
- `normalized.txt`: Cleaned text (BeautifulSoup + NFC)
- `manifest.json`: `{ source_document_id, content_hash, official_url, fetched_at, parser_version, normalizer_version, evidence_offsets }`

`evidence_offsets`: character offsets in normalized.txt for extracted spans (AMENDS mentions, hierarchy anchors). Enables auditing extraction against source.

Re-scrape behavior: if content_hash already exists in raw store → skip (already have this exact content). Never overwrite existing raw files. New content gets new hash directory.

---

## 8. Ingestion Pipeline (Pillar 1)

Ingestion is fully offline and async. Never touches request path.

### Stages (each an arq task with retry)
1. **scrape**: Call phapluat.gov.vn API with versioned allow-list. Delta detection via `updDateTime` + content_hash + overlap window. TLS verify ON (dev override `INSECURE_TLS=true`). Throttle 0.5–1s. Save raw immutable (§7). Rate-limit aware.
2. **parse**: ContentParser with FIXED regex: `_RE_CLAUSE = r"^(\d+[a-z]?)\.\s+(.*)"` accepting alphanumeric clause numbers (e.g., `2a`, `18a`). Footer split with configurable appendix decision (§6/D24). Quote-block tracking. NFC normalize. Idempotent keyed by source_document_id + source_revision_id. Multi-value field normalization.
3. **parse-quality gate**: Before accepting parsed output: check raw length ≥ minimum threshold, presence of Điều anchors, hierarchy count/coverage ratio vs previous revision. If parse is partial/truncated/degraded → keep last-good revision, log warning, do NOT tombstone existing provisions. This prevents data loss from a bad scrape.
4. **amends_extract**: LLM extracts amendments. LLM outputs LegalLocator (citation text + structured article/clause/point), NOT internal source_key. Server resolver maps LegalLocator → provision_uid. Operation classification: ADD_PROVISION | REPLACE_TEXT | REPEAL | AMEND. Enum normalizer: trim whitespace, lowercase, map variants to canonical form (`sửa đổi, bổ sung` with space). Numbering accepts `\d+[a-z]?`. Failed/ambiguous → quarantine edge (review_status=auto, logged, does NOT block document ingest).
5. **embed**: Batch embed via `bkai-foundation-models/vietnamese-bi-encoder` (768d, cosine-normalized). Pre-tokenize with `pyvi.ViTokenizer.tokenize`. Record segmenter_version + model_revision in manifest. Skip unchanged content-hash (reuse vector). Recompute payload/context/derived-state hash for all affected nodes even if vector reused. GPU semaphore serializes with API reranker.
6. **import**: Dual-write to staging (corpus_version=N):
   - Neo4j: MERGE nodes keyed by `(provision_uid, corpus_version)`. Relationships within version. Constraints per-version.
   - Qdrant: Upsert points (UUIDv5(ns, provision_version_id)). Payload includes all metadata + context_text snapshot.
   - Materialize validity from REVIEWED AMENDS: REPEAL cascades to children (Clause, Point inherit inactive from parent Article). Record `validity_basis` per provision ("document_effectStatus", "amends_edge_X_reviewed", "cascade_from_article_Y").
7. **reconcile**: Verify both stores contain expected node/point counts for version N. Check no orphan nodes. Detect discrepancies → DO NOT publish. Alert. Manual intervention required.
8. **tombstone/delete**: Remove provisions absent after reparse ONLY if parse-quality gate passed. If parse was partial/degraded, keep last-good (do not delete).
9. **publish**: `SET active_corpus_snapshot N` in Redis (atomic pointer flip). ONLY if reconcile passed AND at least one change detected. No-op cron (zero changes) does NOT bump version. Write ingestion_manifest (doc count, model_revision, segmenter_version, timestamp, corpus_version, content hashes, updDateTime watermarks, allow-list version).
10. **garbage_collect**: Delete corpus_version < N − retention (default retention = 2 versions). Removes old Neo4j nodes/edges and Qdrant points for expired versions.
11. **report**: Emit metrics (docs processed, points created/deleted, reconcile result, duration).

### Error Handling (Ingestion)
| Failure | Behavior |
|---|---|
| Scrape HTTP 200 + non-null `error` | Upstream error, retry. NOT end-of-catalog. |
| Scrape `docs[]` empty + `error==null` | End-of-catalog, stop pagination. |
| Parse fail on one doc | Skip doc, log, continue batch. |
| Parse-quality gate fails | Keep last-good revision. Log warning. Do NOT tombstone. |
| AMENDS extraction invalid schema | Quarantine that edge (review_status=auto, logged). Document ingest continues. |
| AMENDS target resolution fails for ADD_PROVISION | Valid — new provision being added. Create target. |
| AMENDS target resolution fails for REPEAL/REPLACE | Quarantine edge. Document ingest continues. |
| Embed/import fail | Retry with backoff. After N failures mark failed, alert. |
| Reconcile mismatch | DO NOT publish corpus_version. Alert. Manual intervention. |

### Re-Ingest Special Case
Nghị định 238/2026/NĐ-CP amending Nghị định 168/2024/NĐ-CP, effective 2026-08-15. On/after that date, re-ingest picks up the amendment. The materialized validity updates affected provisions' eligibility windows. Queries with event_date before 2026-08-15 correctly serve NĐ 168 pre-amendment provisions; queries with event_date after serve amended versions.

### Secrets Policy
All credentials loaded from environment variables. `.env` gitignored. Tests use placeholder credentials. Documentation uses `<placeholder>` syntax.

---

## 9. AMENDS Edge Model

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

Server resolver maps LegalLocator → provision_uid via doc_identity + article/clause/point lookup in Neo4j. Resolution failure for ADD_PROVISION = valid (new provision). Resolution failure for REPEAL/REPLACE = quarantine.

### Operation Semantics
| Operation | Meaning | Target Required? | Cascade |
|---|---|---|---|
| ADD_PROVISION | Creates new provision (e.g., bổ sung khoản 2a) | No (target doesn't exist yet) | N/A |
| REPLACE_TEXT | Replaces content of existing provision | Yes | N/A |
| REPEAL | Bãi bỏ provision entirely | Yes | Cascades to children (Clause→Point) |
| AMEND | Generic amendment when specific op unclear | Yes | Reviewer refines to specific op |

Numbering: `\d+[a-z]?` accepted throughout. Parser, schema, resolver all support `2a`, `18a`, etc.

### Edge Schema (Neo4j Relationship Properties)
```python
class AmendEdge:
    operation: Literal["ADD_PROVISION", "REPLACE_TEXT", "REPEAL", "AMEND"]
    amending_provision_uid: str
    target_provision_uid: str
    effective_date: Optional[date]
    evidence_span: str               # Excerpt from source supporting this edge
    source_content_hash: str          # Hash of amending document at extraction time
    extractor_version: str            # Code version of extraction logic
    prompt_version: str               # Version of LLM prompt
    confidence: float                 # 0.0–1.0
    review_status: Literal["auto", "reviewed", "quarantined"]
    validity_basis: Optional[str]     # Set when reviewed: "amends_edge_{id}"
```

Enum normalizer: trim whitespace, lowercase, map common variants:
- `sửa đổi,bổ sung` → `sửa đổi, bổ sung`
- `Sửa Đổi` → `sửa đổi`
- `bo sung` → `bổ sung` (accent restoration attempted)

### review_status Semantics
- `auto`: LLM-extracted, unverified. ANNOTATION ONLY. Does NOT change provision eligibility. Does NOT set effective_to. Context decorated with "[CÓ THỂ đã được sửa đổi bởi ... — chờ xác minh]".
- `reviewed`: Human or corroborated-evidence verified. CHANGES eligibility. Materializes validity (sets effective_to on replaced/repealed provision, creates new provision for ADD_PROVISION). Sets validity_basis.
- `quarantined`: Extraction failed or ambiguous. Logged for manual review. Not applied.

This prevents LLM hallucination from automatically repealing legal provisions. Auto edges surface warnings but don't alter served law.

### Validity Materialization and Cascade
When a REVIEWED REPEAL targets an Article:
1. Mark Article as `inactive` from amendment effective_date. Set `validity_basis = "amends_edge_{id}_reviewed"`.
2. Cascade to all Clauses and Points under that Article: mark `inactive`, set `validity_basis = "cascade_from_article_{uid}"`.
3. Update Qdrant payloads for all affected nodes (eligibility predicate will now exclude them).
4. If vector text unchanged, reuse vector. Always recompute payload + context_text + derived-state hash.

### Real Example: NĐ 168/2024/NĐ-CP
- Bổ sung khoản 2a Điều 28 → ADD_PROVISION, numbering `2a`, target didn't exist before
- Bãi bỏ Điều 5–11 → REPEAL cascade: Article 5–11 + all their Clauses/Points marked inactive
- Hiệu lực có điều kiện → some provisions marked `conditional`, not forced to fake date
- Điều khoản chuyển tiếp → transitional provisions with specific applicability windows

---

## 10. Embeddings + Indexing (Pillar 2)

Batch-first for documents. Query embedding happens online at request time (required for ANN). Metadata filtering is the first gate; vector search is the fallback.

### Document Embedding Job (offline, arq stage 5)
- Model: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base-v2, 768-dim, cosine-normalized). Exact HuggingFace revision pinned in manifest.
- Mandatory: `pyvi.ViTokenizer.tokenize` applied to every text before embedding, both index and query. Segmenter + model revision recorded in manifest and serve as reuse-keys.
- Batch size 32–64 texts. GPU behind semaphore (serialized with API reranker). CPU fallback baseline documented.
- Idempotent: skip nodes already embedded at same content-hash. Reuse vector if text unchanged.
- Documents embedded in batch offline. Queries embedded online at request time (single query, low latency).

### Qdrant Index Layout
```
Collection: "legal_chunks"
  point id   = UUIDv5(NAMESPACE, provision_version_id)
  vector     = embedding 768d (HNSW, cosine)
  payload    = {
     provision_uid,              // stable legal key (no hash)
     provision_version_id,       // versioned key (has hash)
     source_document_id,         // docGUId
     source_revision_id,         // content hash
     doc_identity,               // legal citation (display)
     doc_type, effect_status,    // normalized enums
     validity_intervals[],       // [{from, to, status}] per §5
     issue_date, upd_datetime,
     field[], organ[],           // multi-value arrays
     label (Article/Clause/Point),
     number, parent_article, parent_clause,
     corpus_version,
     embedding_version,
     context_text,               // immutable context-ready text snapshot
     derived_state_hash          // hash of all payload fields except vector
  }
```
Payload indexes enabled on filter fields (keyword/integer/array). `context_text` stores formatted hierarchy + content so Qdrant can serve context even when Neo4j is unavailable.

### Eligibility Predicate (see §5)
Every Qdrant query includes the eligibility predicate checking `validity_intervals` against `reference_date`. Applied identically to BM25 (Neo4j WHERE clause). Both filtered BEFORE RRF fusion.

### BM25 Keyword — Baseline Hypothesis
BM25 lives in Neo4j fulltext (Lucene). This is a **baseline hypothesis**, not a claimed-superior configuration. The evaluation harness (§15) validates whether hybrid (dense+BM25 RRF) outperforms dense-only on this specific corpus. Configuration chosen through quality gate, not assumed.

App-layer RRF fuses dense (Qdrant) + keyword (Neo4j fulltext). Same eligibility predicate applied to both before fusion.

### Neo4j Role
Pure graph (hierarchy traversal, AMENDS, metadata) + Lucene fulltext keyword + provision-level validity storage + Text2Cypher template execution (read-only principal). No dense vector index.

### Index Health + Re-Embed
Metrics exposed: Qdrant point count, HNSW index size, payload index status → Prometheus. On model/segmenter change: bump `embedding_version`, run offline batch re-embed, then swap collection (blue-green or alias) for zero-downtime transition. Only publish new corpus_version after re-embed verified.

---

## 11. Retrieval + Generation (Pillar 3)

Tight request path. Dual cap: max 8 serving contexts AND token budget. Early-exit wherever possible.

### Key Design Decisions
1. **Dual cap**: Maximum 8 serving contexts AND token budget (default 4096 tokens, configurable). Expansion (sibling Points, Article children) counts toward the token budget. If expansion would exceed budget, truncate. Evaluation retrieval_k and serving context_k are separate parameters — eval may test recall@30 while serving caps at 8.
2. **Decompose contract**: Decomposer prompt decides sub-query count. Code enforces `max_subqueries=6` and deduplicates by normalized query string. Prompt alone insufficient; code contract authoritative.
3. **Eligibility predicate mandatory**: Every Qdrant and Neo4j BM25 query carries the eligibility predicate with `reference_date`. No raw unfiltered search ever.
4. **Early-exit patterns**:
   - Cache hit → skip search + rerank + gen (routing + rewriting already executed for correct cache key computation).
   - Intent = reject/greeting → skip retrieval entirely.
   - Empty retrieval result → return "không tìm thấy", skip gen.
5. **Context source**: Primary = Qdrant payload `context_text` (immutable snapshot from ingest time). Fallback = Neo4j hierarchy traversal (if Qdrant payload missing or stale). Ensures degraded-but-functional retrieval when Neo4j is down.
6. **Validity decoration**: Context text cached raw (without date-dependent labels). Validity labels (`[ĐÃ BÃI BỎ]`, `[SỬA ĐỔI]`, `[HIỆU LỰC CÓ ĐIỀU KIỆN]`) decorated per-request based on `reference_date` against provision validity_intervals. This prevents date-dependent cache invalidation.

### Cross-Encoder Reranker
Model: `AITeamVN/Vietnamese_Reranker`, FP16, max_length 2304. Runs on pool of `rerank_top` candidates (default 15). Batch processing with small batch_size for VRAM safety. GPU semaphore ensures serialization with ingestion embedder.

### Heuristic Post-Rerank
Heuristic scoring runs AFTER the eligibility predicate has already filtered invalid provisions. It is an ordering signal, NOT a validity mechanism. Must be eval-gated (§15):
- Recency bonus: `max(0, 2.0 − 0.3 × years_since_effective)` based on provision effective_from date. Applied only to eligible provisions.
- Amendment-tagged provisions carry contextual markers regardless of heuristic score.
- Abolished provisions excluded by eligibility predicate, not penalized by arithmetic.

### Execution Model
FastAPI is async but Neo4j/Qdrant drivers + reranker are blocking:
- Blocking DB calls: bounded thread pool executor (configurable max_workers).
- GPU reranker/embedder: concurrency semaphore (only 1 concurrent GPU operation). CPU fallback baseline when GPU busy.
- Per-stage timeout/deadline. Total request deadline.
- Overload: 429 (rate limited) or 503 with `Retry-After` header when executors saturated.

### Generation
Uses `_QA_FEW_SHOT_SYSTEM_PROMPT` from NLP-LegalQA. Retry generation ONLY before first token streamed. Once any token has been sent to client via SSE, no retry (would produce garbled/duplicate output). After exhaustion pre-token, return friendly error message.

If LLM provider doesn't support streaming/cancel: use buffered mode — generate full answer server-side then emit as token events (or single message event). Declare buffered mode in SSE meta event. Never fake token-by-token streaming from buffered output.

---

## 12. API + SSE Contract

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
Exactly: `meta` (first, always) → `sources` (optional) → `token*` (zero or more) → exactly one terminal: `done` OR `error`.

```
event: meta
data: {"trace_id": "...", "corpus_version": "...", "as_of_date": "...", "event_date": "...", "reference_date": "...", "legal_time_basis": "event|current", "outcome": "full|degraded|cached", "degraded_components": [], "warnings": [], "streaming_mode": "streaming|buffered"}

event: sources
data: [{"citation": "Điểm a Khoản 3 Điều 6 NĐ 168/2024/NĐ-CP", "title": "...", "official_url": "...", "provision_path": ["Điều 6", "Khoản 3", "Điểm a"], "validity_window": {"from": "2025-01-01", "to": null}, "snippet": "..."}]

event: token
data: {"text": "..."}

event: done
data: {"trace_id": "...", "corpus_version": "...", "reference_date": "..."}

event: error
data: {"code": "...", "message": "...", "trace_id": "..."}
```

Notes:
- `sources[]` items expose human citation, title, official_url, provision_path, validity_window, snippet. Internal identifiers (source_key, source_revision_id) NOT exposed.
- `meta` and `done` carry trace_id, resolved dates, corpus_version, outcome/warnings.
- Client disconnect: server cancels generation + logs. Does NOT emit `done` after disconnect (client gone, can't receive).
- `streaming_mode`: "streaming" if provider supports streaming, "buffered" if not. Buffered mode generates full answer server-side then emits events.

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
- `GET /readyz`: Dependencies reachable (Neo4j, Qdrant, Redis connectivity). NO remote LLM call. Returns 200 if all deps OK, 503 otherwise.
- `GET /metrics`: Prometheus exposition format. Infrastructure metrics only.

Health probes never call remote LLM (would burn quota + slow probes).

---

## 13. Caching (Pillar 4)

Three-layer Redis cache. Keys include full dependency fingerprint. Cache lookup AFTER route + rewrite (key depends on rewritten query + intent).

### Cache Key Design
Keys capture everything affecting the cached value:
- **Answer cache**: `cache:ans:{sha256(rewritten_query + intent + filter_params + reference_date + corpus_version + embedding_version + reranker_version + model_name + prompt_version)}`
- **Retrieval cache**: `cache:ret:{sha256(decomposed_queries_sorted + filter_params + reference_date + corpus_version + embedding_version + reranker_version)}`
- **Context cache**: `cache:ctx:{provision_uid}:{corpus_version}` (raw text only, no date-dependent labels)
- **Amends cache**: `cache:amends:{provision_uid}:{corpus_version}`

Normalization for hashing: case-fold + whitespace collapse + unicode NFC normalize. NO stopword stripping (can change meaning).

### Ordering
Flow: validate → trace → dates → route → rewrite → **THEN** cache lookup. Cache key depends on rewritten query and intent. Answer cache hit still creates a new trace (fresh trace_id); stores `cache_origin_trace_id` internally for debugging only.

### Layer 1: Query → Answer Cache
- Value: full answer + sources + intent.
- TTL: long (e.g., 24h). Invalidated by corpus_version in key.
- HIT: skip retrieval + generation. Routing + rewriting already ran.

### Layer 2: Query → Retrieval Cache
- Value: top_k chunk references (provision_uid + label + score).
- TTL: medium (e.g., 6h).
- HIT: skip search + rerank. Still run generation.

### Layer 3: Data/Index Cache
- `cache:ctx:{provision_uid}:{corpus_version}` → raw context text (no validity labels).
- `cache:amends:{provision_uid}:{corpus_version}` → amends list.
- Invalidated by corpus_version in key.

### Invalidation Strategy
Version-based. `corpus_version` published only after successful dual-write + reconcile. Version embedded in keys means stale entries naturally miss. Old entries expire via TTL.

Negative caching avoided or given very short TTL.

### Feedback Interaction
User 👎 does not immediately invalidate cache. Logged to Langfuse. Queries accumulating multiple 👎 flagged for review; manual cache invalidation via CLI only (no HTTP endpoint in v1).

### Cache Metrics (Prometheus)
`cache_hit_ratio` per layer, `cache_miss_latency` vs `cache_hit_latency`. High-cardinality fields (query text, UID, trace_id) NOT used as Prometheus label values — use enum/bucket labels only.

### Failure Mode
Redis down → fail-open (skip cache, run pipeline normally). BUT: arq broker also uses Redis. Redis outage means BOTH cache miss AND ingestion queue unavailable. These are different failure modes — monitor separately. Alert on Redis unavailability.

---

## 14. Graph Schema + Validity

### Node Labels (Hierarchy)
```
Document → Part → Chapter → Section → Article → Clause → Point
```
Levels optional in real documents. Parser attaches to nearest present ancestor; never invents placeholder levels.

Metadata nodes: DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field. Fields multi-value.

### Identity (see §4)
Neo4j node key: `(provision_uid, corpus_version)` composite. `doc_identity` stored as property. `source_document_id` stored as property.

### Relationships
```
Hierarchy : HAS_PART / HAS_CHAPTER / HAS_SECTION / HAS_ARTICLE / HAS_CLAUSE / HAS_POINT
Metadata  : BELONGS_TO_GROUP / HAS_TYPE / HAS_STATUS / ISSUED_BY / SIGNED_BY / IN_FIELD
Cross-doc : RELATED_TO  (from API docListOther, target is ExternalDocument stub if outside corpus)
          : AMENDS {operation, evidence_span, source_content_hash, effective_date,
                    extractor_version, prompt_version, confidence, review_status, validity_basis}
```

### External References
`docListOther` entries whose doc_identity is NOT in the corpus → `ExternalDocument` node: lightweight stub with doc_identity + name for citation only. NOT retrievable (no embedding), NOT validity-bearing (no effectStatus gating). Prevents treating referenced-but-unscraped docs as corpus members.

### Constraints
Composite uniqueness constraints: `(provision_uid, corpus_version)` for all hierarchy nodes. Single-field constraints for metadata nodes. All MERGE-based, idempotent.

---

## 15. Monitoring + Evaluation (Pillar 5)

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
Labels resolved against specific corpus release (not legacy UID/docIdentity). Held-out 80/20 split stratified by doc_identity. Versioned — new corpus release may require label re-resolution.

**Retrieval quality**: Recall@8, Precision@k, MRR, nDCG. Eval retrieval_k separate from serving context_k (eval tests recall@30 while serving caps at 8). Token budget assertion (generated context ≤ budget). Run whenever embedding model, reranker, filter logic, or decompose prompt changes.

**AMENDS-aware relevance**: Successor UIDs (post-amendment) marked as acceptable references for queries with event_date after amendment effective date. Answer citing successor provision is correct, not wrong.

**Answer quality**: LLM-as-judge scores 5 criteria. ROUGE/BERTScore supplementary. LLM-judge runs nightly/manual (expensive). Lexical/recall runs per-PR.

**Quality gate**: Recall@8 must not drop more than 2% vs established baseline. nDCG must not drop more than 3%. Gates promotion/deploy, NOT every PR. Run fingerprint (corpus_version + model revisions + prompt version + timestamp) recorded. Resume of eval run must not mix outputs from different fingerprints (fingerprint check on resume).

**Output**: JSON report + pushed to Langfuse dataset/experiment.

### B. Online Observability

**Langfuse (self-hosted)**: Deployed per official compose profile. Trace every request with spans. Attach feedback score. Include retrieval UIDs + scores.

**Prometheus + Grafana**: Latency p50/p95/p99 per-stage. Cache hit-rate per layer. Qdrant/Neo4j health. Ingestion metrics. Request rate by intent. Alert rules. Telemetry redaction/retention policy. NO high-cardinality labels (query text, UID, trace_id as label values forbidden).

### C. Data Drift Detection
Compare ingestion_manifest across runs. Flag unexpected drift. On model/segmenter change: offline eval BEFORE swapping production index. Deploy only if quality gate passes.

### D. Feedback Loop
```
user 👎 → Langfuse → identify problematic queries
  → review → fix (prompt/filter/data/AMENDS review) → re-evaluate offline
  → quality gate passes → deploy → monitor 👎 decrease
```

---

## 16. Error Handling

Philosophy: fail gracefully, never crash request path. Honest degradation, not false claims.

### Request Path (Online)
| Component failure | Behavior |
|---|---|
| Router LLM | Fallback to intent `retrieve` |
| Rewriter | Use original query, skip rewrite |
| Decomposer | Fallback to `[{"query": original}]` |
| Qdrant unavailable | Return degraded error. Cannot claim functional retrieval without ANN. |
| Neo4j unavailable | Retrieval uses Qdrant payload context_text (degraded but functional). BM25/expansion/Text2Cypher unavailable. Tag response as degraded in SSE meta. |
| Both stores unavailable | Friendly "system temporarily unavailable" message. |
| Reranker | Use ANN ordering, skip rerank. |
| Generator LLM (pre-token) | Retry once with exponential backoff. After exhaustion: friendly error. |
| Generator LLM (mid-stream) | NO retry. Stream truncated. Emit error event. |
| Redis cache | Fail-open, skip cache, run pipeline. Arq ingestion separately affected. |
| Langfuse/Prometheus | Fail-open. Log locally. |

### Ingestion Path (Offline)
See §8 error table. Key: AMENDS extraction failure quarantines edge, does not block document ingest. Parse-quality gate prevents data loss from bad scrape. Reconcile failure blocks publish.

### Cross-Cutting
- Structured JSON logging with `trace_id` spanning components.
- Timeouts on all LLM calls and DB queries. No hung requests.
- Bounded executor for blocking calls. GPU semaphore. Deadline propagation.

---

## 17. Testing Strategy

### Unit Tests (No Network, No Services)
- Parser hierarchy: regex matching including alphanumeric clause numbers (2a, 18a). Footer split. Quote-block tracking. NFC normalization. Optional-level handling. Fixture-based.
- Identity: source_document_id derivation, provision_uid building, UUIDv5 determinism, round-trip mapping.
- AMENDS schema validation: valid/invalid JSON, missing fields, unresolvable targets → quarantine path. Enum normalizer correctness. ADD_PROVISION with non-existent target = valid.
- Cache key construction: verify all dependency dimensions present. Same inputs → same key. Version bump → key change.
- Eligibility predicate: various validity_intervals combinations, reference_date boundary cases, conditional/unknown states.
- RRF aggregation.
- Heuristic rerank: recency calculation, post-eligibility application.
- Query normalization for cache (no stopwords).
- Text2Cypher template selection: valid param extraction, invalid query → route to retrieval.
- Subquery dedup + max_subqueries enforcement.
- Token budget enforcement in context building.
- SSE event serialization.

### Integration Tests (Single Docker-Compose Test Profile)
- Import parsed JSON → verify node/relationship counts + constraints hold.
- Batch embed → verify Qdrant point count + payload filter works + context_text populated.
- End-to-end retrieval on fixture document: query → expected top_k provision_uids match reference.
- Dual-write reconciliation: simulate partial failure → verify no corpus_version published.
- Tombstone: reparse with removed provision → verify deletion (only when quality gate passes).
- Cache invalidation: bump corpus_version → verify old keys miss.
- Atomic publish: verify serving sees consistent state after flip.
- Partial parse: verify last-good kept, no tombstone.
- AMENDS cascade: REVIEWED REPEAL of Article → Clauses/Points inactive.
- Eligibility predicate: event_date before/after amendment → different results.

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

### Evaluation
See §15A. Recall@8, nDCG, token-budget assertion, bounded retries/deadline, run fingerprint.

### CI
- `pytest` unit tests on every PR (no network).
- Integration tests on parser/importer/retrieval/security/identity changes (require docker-compose test profile).
- Acceptance tests on release candidates.
- Eval (lexical/recall) on retrieval-change PRs. LLM-judge nightly/manual.
- Security tests (template injection, credential leaks) on every PR.
- Quality gate on deploy promotion.

---

## 18. Delivery and Repository Layout

### Repository Structure
Root repo (`Agentic`) vendors `NLP-LegalQA` as a **git submodule pinned at a fixed commit SHA**. Read-only reference — no modifications to vendored code.

```
Agentic/
├── .gitmodules                          # Submodule pinning
├── NLP-LegalQA/                         # Git submodule at fixed SHA
├── src/                                 # Application source
│   ├── legal_rag/                       # Main package
│   │   ├── identity.py                  # Identity model (§4)
│   │   ├── temporal.py                  # Temporal model (§5)
│   │   ├── ingestion/                   # Ingestion pipeline (§8)
│   │   ├── amends/                      # AMENDS extraction + resolver (§9)
│   │   ├── embeddings/                  # Embedding jobs (§10)
│   │   ├── retrieval/                   # Retrieval pipeline (§11)
│   │   ├── templates/                   # Text2Cypher read templates (§11)
│   │   ├── cache/                       # Cache layers (§13)
│   │   ├── api/                         # FastAPI endpoints (§12)
│   │   ├── sse/                         # SSE contract (§12)
│   │   └── config/                      # Allow-list, prompts, constants
│   └── ...
├── frontend/                            # Next.js React app
├── tests/
│   ├── unit/                            # No-network unit tests
│   ├── integration/                     # Docker-compose test profile
│   └── acceptance/                      # Concrete scenario tests
├── eval/                                # Eval harness + datasets
├── scripts/                             # CLI tools (cache invalidation, bootstrap, backup)
├── docker-compose.yml                   # All services
├── docker-compose.test.yml              # Test profile
├── Dockerfile                           # App image
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

All fingerprints recorded in corpus manifest + eval manifest. An eval run is reproducible only if all fingerprints match.

### Volumes, Bootstrap, Backup
| Volume | Content | Backup Strategy |
|---|---|---|
| `neo4j_data` | Graph + constraints | neo4j-admin dump/restore |
| `qdrant_storage` | Vectors + payload | Qdrant snapshot API |
| `redis_data` | Cache + arq queue + active_corpus_snapshot | Redis RDB/AOF backup |
| `raw_corpus` | Immutable content-addressed raw store | File-level backup |
| `manifest_store` | Ingestion manifests + eval manifests | File-level backup |
| `quarantine_store` | Quarantined AMENDS edges | File-level backup |

Bootstrap: fresh volumes → restore from backup OR full re-ingest from scratch. Documented procedure.

### GPU Profile
- CUDA available: reranker + embedder share GPU via semaphore (serialized). 
- CPU baseline: both operate on CPU if no GPU. Slower but functional.
- Ingestion worker: can be configured to avoid GPU contention with API (schedule off-peak, or separate CUDA MPS).

---

## 19. Open Items (Genuinely Environment/Data Dependent)

These are NOT design gaps — they are data discovery or environment setup tasks that happen during implementation:

1. **Exact allow-list IDs**: docGroup/field IDs for traffic law need live API discovery. Seeded from NLP-LegalQA's search params, refined via facet exploration during first ingestion run.
2. **Langfuse resource budget**: Official self-host compose recommends minimum resources. Actual allocation depends on host machine. Documented sizing guide provided; user confirms hardware.
3. **HuggingFace model download**: First-time download of vietnamese-bi-encoder + Vietnamese_Reranker weights. Requires internet access during initial setup. Cached in volume for subsequent runs.

All other decisions are finalized in this spec.
