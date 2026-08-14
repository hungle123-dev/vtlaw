# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-14  
**Status:** Draft (post-Codex review revision)  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose on a single machine. Foundation: `n3sfan/NLP-LegalQA` (domain knowledge + data source + graph schema + prompts). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Code and infrastructure are production-shape; deployable to cloud if desired.
- Right-sized for a bounded corpus (Vietnamese traffic-law documents, thousands of chunks). No fake scale claims.
- Legally sound validity model at provision level with explicit `as_of_date`, not document-level hard filters.

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters. Out of scope for this corpus.
- Fine-tuning pipeline (QLoRA/Unsloth). Eval harness supports evaluating fine-tuned models but training is not shipped.
- Public deployment or multi-tenant auth in v1. Single-user demo.
- OCR. Source API returns structured HTML/JSON, not scanned PDFs.
- LLM-based legal validation of amendments. Pydantic validates structure, not legal meaning.

---

## 2. Decisions Log

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Reality bar | Self-host docker-compose, production-shape | Right-sized for portfolio; no cloud cost required |
| D2 | Feature scope | Core RAG + amendment graph + chat/routing/text2cypher + eval/monitoring. No fine-tune | Covers all 5 pillars without research-artifact bloat |
| D3 | Data store | Neo4j (graph) + Qdrant (ANN + metadata filter) | Separation matches "metadata-first + dedicated ANN"; each store does one job |
| D4 | LLM provider | Provider-neutral OpenAI-compatible (`base_url`/`model`/`api_key`) for ALL LLM calls including Text2Cypher | Works with OpenRouter/local vLLM/Ollama/Gemini; same factory everywhere |
| D5 | Monitoring | Langfuse (self-hosted per official compose) + Prometheus + Grafana | Langfuse = LLM trace + feedback + eval experiment; Prometheus/Grafana = infra metrics + alerting. Complementary, not redundant |
| D6 | UI | Next.js/React with SSE streaming contract | Production-feel frontend, explicit SSE protocol |
| D7 | Ingestion queue | arq (async Redis queue) | Async-native, built-in cron + retry, single worker. Right-sized vs Celery |
| D8 | Cache | Redis (answer + retrieval + context), key includes full dependency fingerprint | Pillar 4 controller for cost and latency |
| D9 | Service topology | Modular monolith + infra sidecars | One FastAPI app with clear modules + separate containers for stores/observability/UI. Right-sized for single-machine corpus |
| D10 | Vector storage | Qdrant only; drop Neo4j vector index | Avoids duplicate embed; Neo4j becomes pure graph + keyword |
| D11 | BM25 placement | Neo4j fulltext (Lucene) as baseline hypothesis | Proven on this domain/language; avoids unvalidated Vietnamese sparse tokenizer risk. Not claimed superior — validated through quality gate |
| D12 | Corpus scope | Scoped to traffic-law domain (same allow-list as NLP-LegalQA) | Bounded, well-defined, metadata filtering meaningful |
| D13 | TLS verification | Default ON. Dev-only insecure override via env flag | Production default must verify. Override available for development against self-signed sources |
| D14 | Chunk limit | Max 8 chunks AND token budget (e.g., 4096 tokens) to LLM context | More context ≠ better results; dual cap prevents expansion overflow |
| D15 | Identity scheme | Separate immutable source_key (docGUId + content_hash) from legal citation (doc_identity) | docIdentity can collide across source versions; provisions need stable unique keys |
| D16 | Qdrant point ID | Deterministic UUIDv5 from canonical provision UID | Qdrant requires uint64 or UUID; canonical UID lives in payload |
| D17 | Validity model | Provision-level valid_as_of (effective_from/effective_to), not document-level hard filter | Legal validity varies per provision via amendments; as_of_date query parameter |
| D18 | Text2Cypher security | Read-only Neo4j principal, mandatory LIMIT, parameterized queries, reject write/admin/unsafe CALL | Prevent injection/exfiltration via generated Cypher |
| D19 | SSE contract | Explicit event types: token, sources, done, error, trace_id; cancellation/disconnect handling | Frontend/backend contract clarity |
| D20 | Secrets policy | Env vars only, never in tests/docs/code. `.env` gitignored. Test fixtures use placeholders | Credential safety |

---

## 3. Architecture Overview

### Service Map
```
┌─────────────────────────── docker-compose ───────────────────────────┐
│                                                                      │
│  FRONTEND                                                            │
│  ├─ nextjs          : React chat UI, dark theme, SSE streaming,      │
│  │                   citation panel, feedback button                 │
│                                                                      │
│  APP (modular monolith, 1 Python image)                              │
│  ├─ api             : FastAPI — router→rewrite→decompose→retrieve    │
│  │                   →rerank→gen. Endpoint /chat /health /feedback   │
│  └─ arq-worker      : ingestion offline (scrape→parse→embed→import)  │
│                      + cron re-ingest. Same image, different entry   │
│                                                                      │
│  DATA STORES                                                         │
│  ├─ neo4j           : graph hierarchy Điều→Khoản→Điểm + AMENDS        │
│  │                   + Lucene fulltext BM25                           │
│  │                   (read-only principal for Text2Cypher)            │
│  ├─ qdrant          : ANN dense vector + payload metadata filter     │
│  │                   + context-ready text snapshot                    │
│  └─ redis           : cache (answer+retrieval+context) + arq broker  │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : LLM trace, feedback score, eval experiment     │
│  │                   (deployed per official self-host compose profile)│
│  ├─ prometheus      : infra metrics, scrapes /metrics from api       │
│  └─ grafana         : dashboard latency, cache hit, DB health        │
└──────────────────────────────────────────────────────────────────────┘
```

Langfuse self-host follows the official docker-compose profile with pinned versions, persistent volumes, secrets via env, and documented resource requirements. See https://langfuse.com/self-hosting/.

### Request Path (Hot Path, Pillar 3)
```
user query + optional chat_history + as_of_date (default: today)
  → [cache answer?] HIT (key includes rewritten query + history fingerprint
    + intent + filters + as_of_date + corpus/embedding/reranker/model/prompt versions)
    → return immediately                                (pillar 4)
  → router (LLM) classifies intent
     ├─ direct_answer → gen straight (no retrieval)
     ├─ reject        → template refusal
     ├─ cypher_query  → rewrite → text2cypher (read-only principal,
     │                   parameterized, LIMIT capped, timeout) → gen
     └─ retrieve:
        rewrite (multi-turn; skip if no history)
        → decompose (LLM, JSON sub-queries, append original;
          max_subqueries enforced by code contract + dedup)
        → [cache retrieval?] HIT → skip search
        → metadata filter (Qdrant payload) + eligibility predicate
          (valid_as_of covers as_of_date) + HNSW within filtered set
        → BM25 Neo4j fulltext with same eligibility predicate (parallel)
        → RRF fuse per sub-query → aggregate across sub-queries
        → fetch context from Qdrant payload (primary) or Neo4j hierarchy (fallback)
        → cross-encoder rerank (Vietnamese_Reranker FP16)
        → heuristic rerank (recency bonus AFTER validity gate, eval-gated)
        → select top_k serving contexts (max 8 AND token budget)
        → expand context (optional: sibling Points, Article children;
          expansion counts toward token budget)
        → build context_str (tag [ĐÃ BÃI BỎ]/[SỬA ĐỔI], amends note)
        → gen (LLM, _QA_FEW_SHOT_SYSTEM_PROMPT)
        → write back cache (answer + retrieval)
        → trace Langfuse + metrics Prometheus
        → stream SSE to frontend (token/sources/done/error/trace_id events)
```

### Ingestion Path (Offline, Pillar 1) — Fully Separate from Request Path
```
cron/manual trigger → arq enqueue
  → scrape phapluat.gov.vn (allow-list traffic-law,
    delta detection via updDateTime + content_hash + overlap window)
  → parse hierarchy + metadata + AMENDS (LLM extract, schema-validate)
    (AMENDS failure → quarantine edge, do NOT block document ingest)
  → batch embed (GPU/CPU, vietnamese-bi-encoder + pyvi tokenize)
  → dual-write Neo4j (graph MERGE) + Qdrant (vector+payload+context upsert)
  → reconciliation: verify both stores consistent
  → tombstone/delete provisions absent after reparse
  → publish corpus_version ONLY after both stores verified
    (no-op cron with zero changes does NOT bump version)
  → report metrics
```

Source key (`source_key = docGUId + content_hash`) is the immutable internal identifier linking both stores. Canonical provision UID derives from source_key, not doc_identity. Re-ingest bumps corpus_version only when actual changes published.

---

## 4. Identity Scheme

The source API's `docIdentity` field can collide across different source versions/amendments (confirmed in NLP-LegalQA CLAUDE.md and raw data). Using it as the sole internal key risks overwrites.

### Two-Level Identity
- **source_key**: Immutable internal identifier. For documents: `docGUId` (always unique per source artifact). For provisions: `{source_key}::{label}::{number}` where source_key ties to the specific source document version. Content hash included in manifest for change detection.
- **doc_identity**: Legal citation string (e.g., `168/2024/NĐ-CP`). Used for display, human-facing references, and legal reasoning. Stored as property, NOT as primary key. Multiple source artifacts may share a doc_identity (different versions/amendments).
- **canonical provision UID**: Derived from source_key: `{source_key}::article::{n}::clause::{n}::point::{letter}`. Unique per provision per source version. This is what Qdrant UUIDv5 derives from and Neo4j uses as node key.

### Mapping Round-Trip
```
canonical UID  →  UUIDv5(namespace, canonical_uid)  →  Qdrant point id
canonical UID  ←  Qdrant payload.canonical_uid
canonical UID  →  Neo4j node uid property
Neo4j node uid  ↔  Qdrant payload.canonical_uid  (same value)
```

UUIDv5 is deterministic: same input always produces same UUID. Namespace is a fixed project-specific UUID constant.

---

## 5. Ingestion Pipeline (Pillar 1)

Ingestion is fully offline and async. Never touches request path.

### Stages (each an arq task with retry)
1. **scrape**: Call phapluat.gov.vn API with traffic-law allow-list (docGroup/field filter + keywords). Delta detection uses `updDateTime` + content_hash comparison + overlap window (not just issueDate/effectDate). Save raw `.txt` + `.json`. TLS verification ON by default; dev-only override via `INSECURE_TLS=true` env flag. Throttle between requests (0.5–1s) to respect rate limits.
2. **parse**: ContentParser regex hierarchy (Phần/Chương/Mục/Điều/Khoản/Điểm) + footer split + quote-block tracking + NFC normalize. MetadataParser extracts entities + relationships. Idempotent keyed by source_key. Multi-value field normalization (a document can belong to multiple fields).
3. **amends**: LLM extracts amendments from preamble+content into `amends[]` schema. Schema-validate via Pydantic (structure only — NOT legal validation). Normalize amend_type enum: trim whitespace, lowercase, map variants to canonical form (`sửa đổi, bổ sung` with space, not `sửa đổi,bổ sung`). Failed/ambiguous extraction → quarantine individual edge, log reason, continue document ingest. Quarantined edges available for manual review/retry separately.
4. **embed**: Batch embed using `bkai-foundation-models/vietnamese-bi-encoder` (768d, cosine-normalized). Pre-tokenize with `pyvi.ViTokenizer.tokenize`. Record segmenter_version + model_revision in manifest. Skip nodes already embedded at same content-hash.
5. **import**: Dual-write Neo4j (graph MERGE + constraints) + Qdrant (vector + payload + context-ready text snapshot upsert). Both keyed by canonical UID (via UUIDv5 for Qdrant point id). Payload includes all metadata needed for filtering AND context rendering.
6. **reconcile**: Verify both stores contain expected nodes/points for this ingest run. Detect discrepancies. Tombstone/delete provisions that no longer exist after reparse (mark as deleted or remove). Report orphan count.
7. **finalize**: Publish `corpus_version` ONLY if reconcile passed AND at least one change was made. No-op cron (zero changes detected) does NOT bump version. Write ingestion_manifest (doc count, model_revision, segmenter_version, timestamp, corpus_version, content hashes). Emit metrics.

### Error Handling (Ingestion)
- Scrape HTTP 200 with non-null `error` field → treat as upstream error, retry. Do NOT treat as end-of-catalog.
- Scrape `docs[]` empty with `error == null` → end-of-catalog, stop pagination.
- Parse fail on one doc → skip that doc, log, continue batch.
- AMENDS LLM output invalid schema → quarantine that edge, do not import it, log reason. Document ingest continues. Quarantine queue available for manual review/retry.
- Embed/import fail → retry with backoff; after N failures mark doc/edge as failed, alert.
- Reconcile mismatch → do NOT publish corpus_version. Alert. Manual intervention required.

### Re-Ingest
- Cron job (arq scheduled task) runs periodically (daily/weekly). Delta detection: compare `updDateTime` + content_hash against last-ingested manifest + overlap window (to catch retroactive updates). Only re-ingests changed documents.
- Each successful run with actual changes publishes new `corpus_version`. Version lives inside cache keys so stale entries auto-miss.
- Special case: Nghị định 238/2026/NĐ-CP amending Nghị định 168/2024/NĐ-CP, effective 2026-08-15. Re-ingest on/after that date must pick up the amendment and re-evaluate affected provisions' validity windows. The as_of_date query parameter enables correct answers before and after the effective date.

### Manifest
Each ingestion run records: doc count, model_revision, segmenter_version, timestamp, corpus_version, per-doc content hashes, updDateTime watermarks. Used for cache invalidation, delta detection, monitoring data-drift detection, and reconciliation.

### Secrets Policy
All credentials (Neo4j password, LLM API keys, Redis password) loaded from environment variables only. `.env` is gitignored. Tests use placeholder credentials. Documentation uses `<placeholder>` syntax, never real values.

---

## 6. Embeddings + Indexing (Pillar 2)

Batch-first for documents. Query embedding happens online at request time (required for ANN). Metadata filtering is the first gate; vector search is the fallback.

### Document Embedding Job (offline, arq stage 4)
- Model: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base-v2, 768-dim, cosine-normalized). Matches NLP-LegalQA exactly.
- Mandatory: `pyvi.ViTokenizer.tokenize` applied to every text before embedding, both index and query. Segmenter + model revision recorded in manifest and serve as reuse-keys.
- Batch size 32–64 texts. GPU when available, CPU fallback.
- Idempotent: skip nodes already embedded at same content-hash.
- Documents are embedded in batch offline. Queries are embedded online at request time (single query, low latency).

### Qdrant Index Layout
```
Collection: "legal_chunks"
  point id   = UUIDv5(NAMESPACE, canonical_uid)
  vector     = embedding 768d (HNSW, cosine)
  payload    = {
     canonical_uid,              // round-trip back to legal UID
     source_key,                 // immutable internal key
     doc_identity,               // legal citation (display)
     doc_type, effect_status,    // normalized enums
     effective_from, effective_to, // provision-level validity
     issue_date, upd_datetime,
     field[], organ[],           // multi-value arrays
     label (Article/Clause/Point),
     number, parent_article, parent_clause,
     corpus_version,
     embedding_version,
     context_text                // immutable context-ready text snapshot
  }
```
Payload indexes enabled on filter fields (keyword/integer/array) for fast metadata filtering. `context_text` stores the formatted hierarchy + content so Qdrant can serve context even when Neo4j is unavailable.

### Eligibility Predicate (Replaces Hard Effect-Status Filter)
Instead of document-level `effect_status=còn hiệu lực`, every retrieval query applies a provision-level eligibility predicate:
```
effective_from <= as_of_date AND (effective_to IS NULL OR effective_to >= as_of_date)
```
This predicate is applied identically to both Qdrant (payload filter) and Neo4j BM25 (WHERE clause) BEFORE RRF fusion. The `as_of_date` parameter defaults to today but can be overridden by the user.

Effect status remains stored as metadata for display/filtering but is NOT the sole validity gate. A provision within a "còn hiệu lực" document may still be bãi bỏ/thay thế by amendment — the provision-level validity window captures this.

### Metadata-First Filtering
Every Qdrant query includes: eligibility predicate + optional field/doc_type/organ filters. ANN runs only within the eligible candidate set. This directly implements "metadata filtering is your first gate, vector search is the fallback."

### BM25 Keyword (Hybrid Half) — Baseline Hypothesis
BM25 lives in Neo4j fulltext (Lucene). This is a **baseline hypothesis**, not a claimed-superior configuration. The evaluation harness (§10) validates whether hybrid (dense+BM25 RRF) outperforms dense-only on this specific corpus. Configuration is chosen through the quality gate, not assumed.

App-layer RRF fuses dense (Qdrant) + keyword (Neo4j fulltext). Same eligibility predicate applied to both before fusion.

### Neo4j Role
Pure graph (hierarchy traversal, AMENDS, metadata) + Lucene fulltext keyword + provision-level validity storage. No dense vector index.

### Index Health + Re-Embed (Links to Pillar 5)
- Metrics exposed: Qdrant point count, HNSW index size, payload index status → Prometheus.
- On model/segmenter change: bump `embedding_version`, run offline batch re-embed, then swap collection (blue-green or alias) for zero-downtime transition. Only publish new corpus_version after re-embed verified.

---

## 7. Retrieval + Generation (Pillar 3)

Tight request path. Dual cap: max 8 chunks AND token budget. Early-exit wherever possible.

### Flow Detail (intent=retrieve)
See §3 request path diagram above. Key design decisions:

1. **Dual cap**: Maximum 8 serving contexts AND token budget (e.g., 4096 tokens). Expansion (sibling Points, Article children) counts toward the token budget. If expansion would exceed budget, truncate. Evaluation retrieval_k and serving context_k are separate parameters — eval may test recall@30 while serving caps at 8.
2. **Decompose contract**: The decomposer prompt decides sub-query count (NLP-LegalQA prompt instructs "câu đơn giản thì 1 subquery là đủ"). Code enforces `max_subqueries` (e.g., 6) and deduplicates by normalized query string. Prompt alone is insufficient; code contract is authoritative.
3. **Eligibility predicate mandatory**: Every Qdrant and Neo4j BM25 query carries the provision-level eligibility predicate with `as_of_date`. No raw unfiltered search ever.
4. **Early-exit patterns**:
   - Cache hit → skip search + rerank + gen (but routing/rewrite still run if chat history exists, because cache key depends on rewritten query).
   - Intent = reject/direct_answer → skip retrieval entirely.
   - Empty retrieval result → return "không tìm thấy", skip gen.
5. **Text2Cypher security**: Read-only Neo4j principal (separate from ingestion principal). Reject any generated Cypher containing write operations (CREATE, MERGE, DELETE, SET, REMOVE), admin commands, or unsafe CALL procedures. Mandatory LIMIT clause (server-enforced cap). Parameterized queries only (no string interpolation). Query timeout. Injection tests in CI. Provider-neutral LLM (same factory as other LLM calls, not hardcoded to Gemini/OpenRouter).
6. **Context source**: Primary = Qdrant payload `context_text` (immutable snapshot from ingest time). Fallback = Neo4j hierarchy fetch (if Qdrant payload missing or stale). This ensures degraded-but-functional retrieval when Neo4j is down.

### Cross-Encoder Reranker
Model: `AITeamVN/Vietnamese_Reranker`, FP16, max_length 2304. Runs on pool of `rerank_top` candidates (default 15). Batch processing with small batch_size for VRAM safety.

### Heuristic Post-Rerank
Heuristic scoring runs AFTER the eligibility predicate has already filtered invalid provisions. It is an ordering signal, NOT a validity mechanism. Must be eval-gated (§10):
- Recency bonus: `max(0, 2.0 − 0.3 × years_since_effective)` based on provision effective_from date. Applied only to eligible provisions.
- Amendment-tagged provisions carry contextual markers (`[LƯU Ý - NỘI DUNG SỬA ĐỔI]`) in context_text regardless of heuristic score.
- Abolished provisions (`bãi bỏ`/`thay thế`) are excluded by the eligibility predicate, not penalized by arithmetic. If an abolished provision appears in results due to as_of_date being before abolition, it is correctly eligible and should not be penalized.

### Generation Prompt
Uses `_QA_FEW_SHOT_SYSTEM_PROMPT` from NLP-LegalQA (10 principles including faithfulness to context, handling overlapping docs by effect_date, correct citation format, abolished-provision tagging). Few-shot examples included.

---

## 8. Caching (Pillar 4)

Three-layer Redis cache. Keys include full dependency fingerprint. Real path: `query → cache → (miss) retrieve+LLM → write back`.

### Cache Key Design
Keys must capture everything that affects the cached value:
- **Answer cache**: `ans:{sha256(rewritten_query + intent + filter_params + as_of_date + corpus_version + embedding_version + reranker_version + model_name + prompt_version)}`
- **Retrieval cache**: `ret:{sha256(decomposed_queries_sorted + filter_params + as_of_date + corpus_version + embedding_version + reranker_version)}`
- **Context cache**: `ctx:{canonical_uid}:{corpus_version}`
- **Amends cache**: `amends:{canonical_uid}:{corpus_version}`

Note: raw user query is NEVER used as cache key for multi-turn conversations. The rewritten query (which resolves pronouns/references from chat history) is used instead. If chat history exists, routing and rewriting MUST run before cache lookup (otherwise the cache key is wrong).

### Layer 1: Query → Answer Cache (FAQ/Repeat)
- Value: full answer + sources + intent + trace_id.
- TTL: long (e.g., 24h). Invalidation via corpus_version in key — re-ingest publishes new version, old keys auto-miss.
- HIT: skip retrieval + generation. Routing + rewriting still execute if chat history present (to compute correct cache key).

### Layer 2: Query → Retrieval Cache (Top Chunks)
- Value: top_k chunk references (canonical_uid + label + score). Context fetched fresh or from ctx cache.
- TTL: medium (e.g., 6h).
- HIT: skip search + rerank, but still run generation.

### Layer 3: Data/Index Cache (Hot Lookups)
- `ctx:{canonical_uid}:{corpus_version}` → formatted context text.
- `amends:{canonical_uid}:{corpus_version}` → amends list.
- Invalidated explicitly by corpus_version in key. When corpus_version bumps, old ctx/amends entries auto-miss.

### Query Normalization
Before hashing keys: lowercase, collapse whitespace, light stopword strip. Goal: near-duplicate questions map to the same key without losing semantic meaning.

### Invalidation Strategy
Version-based. `corpus_version` published only after successful dual-write + reconcile (§5). Version embedded in keys means stale entries naturally miss; no manual flush needed. Old entries expire via TTL.

Negative caching ("not found") avoided or given very short TTL to prevent stale emptiness after re-ingest adds data.

### Feedback Interaction
User 👎 does not immediately invalidate cache (one negative signal is insufficient). Logged to Langfuse. Queries accumulating multiple 👎 flagged for review; manual key invalidation available via admin endpoint.

### Cache Metrics (Prometheus)
- `cache_hit_ratio` per layer (answer/retrieval/context).
- `cache_miss_latency` vs `cache_hit_latency`.
- These metrics demonstrate cost/latency savings on CV.

### Failure Mode
Redis down → fail-open (skip cache, run pipeline normally). Cache is optimization, not a hard dependency. Log + metric alert raised.

---

## 9. Monitoring + Evaluation (Pillar 5)

Two distinct subsystems: offline evaluation and online observability.

### A. Evaluation Harness (Offline)
Dataset: labeled QA set derived from NLP-LegalQA's `qa_dataset/`. Requires migration: current CSV stores `reference` as comma-separated string; migrate to structured `reference[]` array with per-reference metadata (UID, label, relevance tier). Labels are versioned, deduplicated, and include a held-out test split. Each label carries `as_of_date` to handle temporal validity.

**Retrieval quality**: Recall@8, Precision@k, MRR, nDCG computed against reference UIDs. Evaluation retrieval_k is separate from serving context_k (eval may test recall@30 while serving caps at 8). Token budget enforcement validated during eval. Run whenever embedding model, reranker, filter logic, or decompose prompt changes. Enables A/B comparison.

**AMENDS-aware relevance**: When a provision has been amended, the "correct" answer depends on as_of_date. Successor UIDs (post-amendment) are marked as acceptable references for queries with as_of_date after the amendment effective date. Eval accounts for this — an answer citing the successor provision is correct, not wrong.

**Answer quality**: LLM-as-judge scores 5 criteria (legal_accuracy, citation correctness, completeness, hallucination absence, structure) using NLP-LegalQA's `_JUDGE_USER_PROMPT` framework. ROUGE/BERTScore as supplementary sanity checks only.

**Output**: JSON report + pushed to Langfuse dataset/experiment for version-over-version tracking.

**Trigger**: Manual CLI or CI on retrieval-change PRs. Never run inline during serving.

**Baseline validation**: Hybrid (dense+BM25 RRF) vs dense-only is tested through this harness. Configuration is chosen by quality gate results, not assumed.

### B. Online Observability

**Langfuse (LLM-specific, self-hosted)**:
- Deployed per official self-host docker-compose profile (https://langfuse.com/self-hosting/) with pinned versions, persistent volumes, secrets via env.
- Trace every request with spans: router / rewrite / decompose / retrieve / rerank / gen, each with latency + tokens + cost.
- Attach user feedback score (👍/👎 from UI) to trace.
- Include retrieval UIDs + scores for debugging.

**Prometheus + Grafana (Infrastructure)**:
- Latency: request p50/p95/p99, per-stage breakdown.
- Cache hit-rate per layer (§8).
- Qdrant/Neo4j health: point count, query latency, index size.
- Ingestion: job success/fail rate, re-ingest watermark, docs/sec throughput, reconcile pass/fail.
- Request rate by intent.
- Alert rules: cache hit-rate drop, latency p95 threshold breach, consecutive ingestion failures, reconcile failures.

### C. Data Drift Detection
Compare `ingestion_manifest` across runs: doc count delta, new/modified documents, updDateTime distribution. Flag unexpected drift.

On embedding model or segmenter change: run offline eval BEFORE swapping production index (§6). Deploy only if recall does not regress.

### D. Feedback Loop Closure
```
user 👎 → Langfuse score → identify problematic queries
  → review → fix (prompt/filter/data) → re-evaluate offline
  → deploy → monitor 👎 rate decrease
```

### Demonstrable Metrics for Portfolio
- "Recall@8 = X%, improved Y% after adding amendment-aware eligibility"
- "Answer-cache hit 60% → 60% reduction in LLM calls"
- "Latency p95 < Z ms"
- "Feedback-positive rate improved from A% to B% after iteration N"

---

## 10. Graph Schema + AMENDS

Domain differentiator. Schema largely matches NLP-LegalQA; validation upgraded to production grade.

### Node Labels (Hierarchy)
```
Document → Part → Chapter → Section → Article → Clause → Point
```
Levels are optional in real documents (decrees omit Part/Section). Parser attaches to nearest present ancestor; never invents placeholder levels.

Metadata nodes: DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field. Fields are multi-value (a document can belong to multiple fields).

### Identity (see §4)
- Neo4j node key = canonical UID (derived from source_key).
- doc_identity stored as property for display/legal citation.
- source_key stored as property for provenance.

### Relationships
```
Hierarchy : HAS_PART / HAS_CHAPTER / HAS_SECTION / HAS_ARTICLE / HAS_CLAUSE / HAS_POINT
Metadata  : BELONGS_TO_GROUP / HAS_TYPE / HAS_STATUS / ISSUED_BY / SIGNED_BY / IN_FIELD
Cross-doc : RELATED_TO  (from API docListOther)
          : AMENDS {type, evidence_span, source_hash, effective_date,
                    extractor_version, prompt_version, confidence, review_status}
```

### AMENDS Edge — Enhanced Schema
Source: LLM extraction from preamble + content using instruction prompt (derived from NLP-LegalQA's `data/amends/instruction.txt`). NOT derived from `docListOther`.

Schema per amend entry (Pydantic-validated for structure, NOT legal meaning):
```python
class AmendEdge(BaseModel):
    amending_source_key: str
    amending_article: str
    amending_clause: Optional[str]
    amending_point: Optional[str]
    amend_type: Literal["sửa đổi", "bổ sung", "bãi bỏ", "thay thế", "sửa đổi, bổ sung"]
    target_source_key: str
    target_article: str
    target_clause: Optional[str]
    target_point: Optional[str]
    evidence_span: str          # excerpt from source text supporting this edge
    source_content_hash: str    # hash of source document at extraction time
    effective_date: Optional[date]
    extractor_version: str      # code version of extraction logic
    prompt_version: str         # version of LLM prompt used
    confidence: float           # 0.0–1.0 from LLM or heuristic
    review_status: Literal["auto", "reviewed", "quarantined"]
```

Enum normalizer: trim whitespace, lowercase, map common variants (`sửa đổi,bổ sung` → `sửa đổi, bổ sung`; `Sửa Đổi` → `sửa đổi`).

Import resolves both endpoints to canonical UIDs and executes `MERGE (src)-[:AMENDS {props}]->(tgt)`. Failed resolution → quarantine edge (do NOT block document ingest). Quarantined edges logged with reason, available for manual review/retry.

Usage in retrieval (§7):
- Eligibility predicate uses AMENDS-derived effective_from/effective_to on provisions, not heuristic penalties.
- Context text tagged `[LƯU Ý - NỘI DUNG SỬA ĐỔI]` with amending content for provisions with active amendments.
- Abolished provisions excluded by eligibility predicate, not arithmetic penalty.

### Constraints
13 uniqueness constraints matching NLP-LegalQA pattern, updated to use canonical UID as key. All MERGE-based, idempotent.

---

## 11. Error Handling

Philosophy: fail gracefully, never crash the request path. Each layer has a defined fallback. Honest degradation, not false claims.

### Request Path (Online)
| Component failure | Behavior |
|---|---|
| Router LLM | Fallback to intent `retrieve` (runs full pipeline) |
| Rewriter | Use original query, skip rewrite |
| Decomposer | Fallback to `[{"query": original}]` |
| Qdrant unavailable | Return degraded error ("vector search unavailable"). Cannot claim functional retrieval without ANN. Log + alert |
| Neo4j unavailable | Retrieval uses Qdrant payload `context_text` for context (degraded but functional). Graph traversal/AMENDS unavailable. Tag response as degraded |
| Both stores unavailable | Return friendly "system temporarily unavailable" message, log + alert |
| Reranker | Use ANN ordering, skip rerank |
| Generator LLM | Retry once with exponential backoff (idempotent read-like operation). After exhaustion return friendly error message. Do NOT retry indefinitely |
| Redis cache | Fail-open, skip cache, run pipeline |
| Langfuse/Prometheus | Fail-open, no request impact. Log locally |

Principle: observability and cache are optimizations; their failure must not kill requests. Stores have honest degradation paths — Qdrant-down means no vector search (not "BM25-only functional"), Neo4j-down means degraded context (not "full functional").

### Ingestion Path (Offline)
See §5 error handling. Key points: scrape HTTP 200 with non-null `error` is treated as upstream error (retry), NOT as end-of-catalog. AMENDS extraction failure quarantines the edge, does not block document ingest.

### Cross-Cutting
- Structured JSON logging with `trace_id` spanning components → aligns with Langfuse traces.
- `/health` endpoint checks neo4j, qdrant, redis, llm reachability → returns per-component status. Scraped by Prometheus + consumed by UI.
- Timeouts on all LLM calls and DB queries. No hung requests.
- Generator LLM: retry once (bounded), not indefinitely. Rationale: generation is mostly idempotent (same prompt → similar output), one retry handles transient failures. Beyond that, return error rather than risk runaway costs.

---

## 12. Testing Strategy

Layered pyramid focused on highest-risk logic (parser, retrieval, cache invalidation, security).

### Unit Tests (Many, Fast)
- Parser hierarchy: regex matching for Điều/Khoản/Điểm, footer split, quote-block tracking, NFC normalization, optional-level handling. Fixture-based.
- Identity: source_key derivation, canonical UID building, UUIDv5 determinism, round-trip mapping.
- AMENDS schema validation: valid/invalid JSON, missing fields, unresolvable targets → quarantine path. Enum normalizer correctness.
- Cache key construction: verify all dependency dimensions present. Same inputs → same key. Version bump → key change.
- Eligibility predicate: provision with various effective_from/effective_to combinations, as_of_date boundary cases.
- RRF aggregation: fuse multiple sub-query result sets correctly.
- Heuristic rerank: recency calculation, post-eligibility application.
- Query normalization for cache.
- Text2Cypher security: reject write/admin/unsafe CALL patterns. Parameterization. LIMIT enforcement. Timeout.
- Subquery dedup + max_subqueries enforcement.
- Token budget enforcement in context building.

### Integration Tests (Fewer, Require Services)
Run in docker-compose test profile or testcontainers (neo4j + qdrant + redis):
- Import parsed JSON → verify node/relationship counts + constraints hold.
- Batch embed → verify Qdrant point count + payload filter works + context_text populated.
- End-to-end retrieval on fixture document: query → expected top_k UIDs match reference.
- Dual-write reconciliation: simulate partial failure → verify no corpus_version published.
- Tombstone: reparse with removed provision → verify deletion/tombstone.
- Cache invalidation: bump corpus_version → verify old keys miss.
- Text2Cypher injection tests: malicious input → rejected.

Never touch production data. Isolated test database/collection. Placeholder credentials only.

### Evaluation (Quality Gate, Not Pass/Fail Test)
Recall@8, Precision, MRR, nDCG + LLM-judge on labeled dataset (§9A). AMENDS-aware relevance. Token budget validated. Run manually or in CI. Produces report, does not block merge but tracks regression over time.

### Contract / Smoke Tests
- `/health` returns all component statuses.
- `/chat` with fixture query → 200, SSE stream with correct event types (token/sources/done/error/trace_id), answer present, sources present, Pydantic schema valid.
- SSE disconnect handling: server detects client disconnect, cancels generation.
- Frontend: smoke check SSE/streaming endpoint only.

### Explicitly Not Tested
- LLM output quality (non-deterministic) → handled by eval harness, not unit tests.
- Detailed UI behavior (Next.js) → smoke only.

### CI
- `pytest` unit tests on every PR.
- Integration tests on parser/importer/retrieval/security changes (require docker).
- Eval runs manually or nightly; reports pushed to Langfuse.
- Security tests (Text2Cypher injection, credential leaks) on every PR.

---

## 13. SSE Contract

Server-Sent Events protocol between API and frontend:

### Event Types
```
event: token
data: {"text": "..."}

event: sources
data: [{"uid": "...", "label": "...", "score": 0.95, "uid_formatted": "...", "context_snippet": "..."}]

event: done
data: {"trace_id": "...", "corpus_version": "..."}

event: error
data: {"code": "...", "message": "...", "trace_id": "..."}
```

### Behavior
- `token`: streamed incrementally as LLM generates.
- `sources`: sent once before tokens, contains retrieval results.
- `done`: final event, signals completion. Includes trace_id for Langfuse correlation.
- `error`: terminal event on failure. Includes trace_id if available.
- Client disconnect: server detects broken connection, cancels LLM generation + releases resources.
- Cancellation: client sends abort signal (disconnect or explicit cancel endpoint). Server stops generation, emits `done` with `cancelled: true`.
- Timeout: server-side max generation timeout. Emits `error` if exceeded.

---

## 14. Assumptions Requiring Confirmation

These assumptions were made during spec revision and should be confirmed before implementation planning:

1. **UUIDv5 namespace**: A fixed project-specific UUID will be generated and committed as a constant. Confirm this approach vs alternative (uint64 hash).
2. **Token budget value**: Spec uses 4096 as example. Actual value should be determined by the chosen LLM's context window and empirical testing.
3. **Provision-level effective_from/effective_to derivation**: Currently these are inferred from document effect_date + AMENDS edges. Confirm whether the source API provides provision-level dates directly or if derivation logic is needed.
4. **Quarantine storage**: Quarantined AMENDS edges are logged but storage location TBD (separate DB table? JSON file? Langfuse annotation?).
5. **Corpus version format**: Monotonic integer vs semantic version vs timestamp. Recommendation: monotonic integer incremented only on successful publish.
6. **Held-out test split ratio**: Evaluation dataset split ratio TBD (recommend 80/20 train/test, stratified by doc_identity).
7. **Langfuse self-host resource requirements**: Official compose recommends minimum 4GB RAM for Langfuse stack. Confirm single-machine has sufficient resources for all services simultaneously.
