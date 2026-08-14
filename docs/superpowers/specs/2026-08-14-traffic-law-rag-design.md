# Vietnamese Traffic-Law RAG — Production Design

**Date:** 2026-08-14  
**Status:** Draft  
**Scope:** Complete production-grade graph-RAG system for Vietnamese traffic law QA, self-hosted via docker-compose. Foundation: `n3sfan/NLP-LegalQA` (domain knowledge + data source + graph schema + prompts). Target: portfolio/CV showcase demonstrating real AI systems engineering beyond coursework.

---

## 1. Goals and Non-Goals

### Goals
- A complete, runnable production RAG system covering all five pillars: ingestion, embeddings/indexing, retrieval+generation, caching, monitoring.
- Domain-differentiated by hybrid graph-RAG with amendment-aware retrieval over a structured legal hierarchy.
- Self-hostable end-to-end on one machine via `docker-compose`. Code and infrastructure are production-shape; deployable to cloud if desired.
- Right-sized for a bounded corpus (Vietnamese traffic-law documents, thousands of chunks). No fake scale claims.

### Non-Goals
- Millions-of-PDFs scale, distributed sharding, multi-node ANN clusters. Out of scope for this corpus.
- Fine-tuning pipeline (QLoRA/Unsloth). Eval harness supports evaluating fine-tuned models but training is not shipped.
- Public deployment or multi-tenant auth in v1. Single-user demo.
- OCR. Source API returns structured HTML/JSON, not scanned PDFs.

---

## 2. Decisions Log

| # | Decision | Choice | Rationale |
|---|---|---|---|
| D1 | Reality bar | Self-host docker-compose, production-shape | Right-sized for portfolio; no cloud cost required |
| D2 | Feature scope | Core RAG + amendment graph + chat/routing/text2cypher + eval/monitoring. No fine-tune | Covers all 5 pillars without research-artifact bloat |
| D3 | Data store | Neo4j (graph) + Qdrant (ANN + metadata filter) | Separation matches "metadata-first + dedicated ANN"; each store does one job |
| D4 | LLM provider | Provider-neutral OpenAI-compatible (`base_url`/`model`/`api_key`) | Works with OpenRouter/local vLLM/Ollama; same factory everywhere |
| D5 | Monitoring | Langfuse + Prometheus + Grafana | Langfuse = LLM trace + feedback + eval experiment; Prometheus/Grafana = infra metrics + alerting. Complementary, not redundant |
| D6 | UI | Next.js/React | Production-feel frontend, SSE streaming, citation panel, feedback button |
| D7 | Ingestion queue | arq (async Redis queue) | Async-native, built-in cron + retry, single worker. Right-sized vs Celery |
| D8 | Cache | Redis (answer + retrieval + context) | Pillar 4 controller for cost and latency |
| D9 | Service topology | Modular monolith + infra sidecars | One FastAPI app with clear modules + separate containers for stores/observability/UI. Right-sized for single-machine corpus |
| D10 | Vector storage | Qdrant only; drop Neo4j vector index | Avoids duplicate embed; Neo4j becomes pure graph + keyword |
| D11 | BM25 placement | Neo4j fulltext (Lucene) | Proven on this domain/language; avoids unvalidated Vietnamese sparse tokenizer risk |
| D12 | Corpus scope | Scoped to traffic-law domain (same allow-list as NLP-LegalQA) | Bounded, well-defined, metadata filtering meaningful |
| D13 | TLS verification | `verify=False`, matching NLP-LegalQA | Pragmatic match to source behavior |
| D14 | Chunk limit | Max 8 chunks to LLM context | More context ≠ better results; pillar 3 constraint |

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
│  ├─ qdrant          : ANN dense vector + payload metadata filter     │
│  └─ redis           : cache (answer+retrieval+context) + arq broker  │
│                                                                      │
│  OBSERVABILITY                                                       │
│  ├─ langfuse        : LLM trace, feedback score, eval experiment     │
│  ├─ prometheus      : infra metrics, scrapes /metrics from api       │
│  └─ grafana         : dashboard latency, cache hit, DB health        │
└──────────────────────────────────────────────────────────────────────┘
```

### Request Path (Hot Path, Pillar 3)
```
user query
  → [cache answer?] HIT → return immediately            (pillar 4)
  → router (LLM) classifies intent
     ├─ direct_answer → gen straight (no retrieval)
     ├─ reject        → template refusal
     ├─ cypher_query  → rewrite → text2cypher → gen
     └─ retrieve:
        rewrite (multi-turn; skip if no history)
        → decompose (LLM, JSON sub-queries, append original)
        → [cache retrieval?] HIT → skip search
        → metadata filter (Qdrant payload) + HNSW within filtered set
        → BM25 Neo4j fulltext (parallel)
        → RRF fuse per sub-query → aggregate across sub-queries
        → fetch hierarchy context (Neo4j variable-length path)
        → cross-encoder rerank (Vietnamese_Reranker FP16)
        → heuristic rerank (abolished penalty + recency bonus)
        → select top_k (max 8)
        → expand context (optional: sibling Points, Article children)
        → build context_str (tag [ĐÃ BÃI BỎ]/[SỬA ĐỔI], amends note)
        → gen (LLM, _QA_FEW_SHOT_SYSTEM_PROMPT)
        → write back cache (answer + retrieval)
        → trace Langfuse + metrics Prometheus
        → stream SSE to frontend
```

### Ingestion Path (Offline, Pillar 1) — Fully Separate from Request Path
```
cron/manual trigger → arq enqueue
  → scrape phapluat.gov.vn (allow-list traffic-law)
  → parse hierarchy + metadata + AMENDS (LLM extract, schema-validate)
  → batch embed (GPU/CPU, vietnamese-bi-encoder + pyvi tokenize)
  → import Neo4j (graph MERGE) + Qdrant (vector+payload upsert)
  → bump ingestion_version (Redis)
  → report metrics
```

UID is the single source of truth linking both stores: `{doc_identity}::article::{n}::clause::{n}::point::{letter}` serves as Neo4j node key AND Qdrant point id. Re-ingest bumps version to invalidate caches.

---

## 4. Ingestion Pipeline (Pillar 1)

Ingestion is fully offline and async. Never touches request path.

### Stages (each an arq task with retry)
1. **scrape**: Call phapluat.gov.vn API with traffic-law allow-list (docGroup/field filter + keywords). Save raw `.txt` + `.json`. Use `verify=False` matching NLP-LegalQA. Throttle between requests (0.5–1s) to respect rate limits.
2. **parse**: ContentParser regex hierarchy (Phần/Chương/Mục/Điều/Khoản/Điểm) + footer split + quote-block tracking + NFC normalize. MetadataParser extracts entities + relationships. Idempotent keyed by doc_identity.
3. **amends**: LLM extracts amendments from preamble+content into `amends[]` schema (amending/target doc/article/clause/point + amend_type). Schema-validate before accepting; reject malformed output and log.
4. **embed**: Batch embed using `bkai-foundation-models/vietnamese-bi-encoder` (768d, cosine-normalized). Pre-tokenize with `pyvi.ViTokenizer.tokenize`. Record segmenter_version + model_revision in manifest. Skip nodes already embedded at same content-hash.
5. **import**: Neo4j MERGE (graph + constraints) + Qdrant upsert (point id = UID, payload = metadata). Both keyed by UID for consistency.
6. **finalize**: Bump `ingestion_version` counter in Redis, emit metrics, write ingestion_manifest.

### Error Handling (Ingestion)
- Scrape HTTP 200 with non-null `error` field → treat as upstream error, retry. Do NOT treat as end-of-catalog.
- Scrape `docs[]` empty with `error == null` → end-of-catalog, stop pagination.
- Parse fail on one doc → skip that doc, log, continue batch.
- AMENDS LLM output invalid schema → reject, do not import, log reason.
- Embed/import fail → retry with backoff; after N failures mark doc as failed, alert.

### Re-Ingest
- Cron job (arq scheduled task) runs periodically (daily/weekly), scans new/modified documents by issueDate/effectDate watermark. Only re-ingests delta.
- Each run bumps `ingestion_version`. Version lives inside cache keys so stale entries auto-miss.

### Manifest
Each ingestion run records: doc count, model_revision, segmenter_version, timestamp, version. Used for cache invalidation + monitoring data-drift detection.

---

## 5. Embeddings + Indexing (Pillar 2)

Batch-first. Metadata filtering is the first gate; vector search is the fallback.

### Embedding Job (offline, arq stage 4)
- Model: `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-base-v2, 768-dim, cosine-normalized). Matches NLP-LegalQA exactly.
- Mandatory: `pyvi.ViTokenizer.tokenize` applied to every text before embedding, both index and query. Segmenter + model revision recorded in manifest and serve as reuse-keys.
- Batch size 32–64 texts. GPU when available, CPU fallback. Never embed individual queries at request time.
- Idempotent: skip nodes already embedded at same content-hash.

### Qdrant Index Layout
```
Collection: "legal_chunks"
  point id   = UID  ({doc_identity}::article::n::clause::n::point::letter)
  vector     = embedding 768d (HNSW, cosine)
  payload    = {
     doc_identity, doc_type, effect_status, effect_date,
     issue_date, field, organ, label (Article/Clause/Point),
     number, parent_article, parent_clause, ingestion_version
  }
```
Payload indexes enabled on filter fields (keyword/integer) for fast metadata filtering.

### Metadata-First Filtering
Every Qdrant query includes payload filters: `effect_status=còn hiệu lực`, `field=giao thông`, optional date range. ANN runs only within the filtered candidate set. This directly implements "metadata filtering is your first gate, vector search is the fallback."

### BM25 Keyword (Hybrid Half)
BM25 lives in Neo4j fulltext (Lucene) — proven on this domain/language, tokenization safe. App-layer RRF fuses dense (Qdrant) + keyword (Neo4j fulltext). Hybrid retained because it improves recall over vector-only (proven in NLP-LegalQA evaluation).

### Neo4j Role
Pure graph (hierarchy traversal, AMENDS, metadata) + Lucene fulltext keyword. No dense vector index.

### Index Health + Re-Embed (Links to Pillar 5)
- Metrics exposed: Qdrant point count, HNSW index size, payload index status → Prometheus.
- On model/segmenter change: bump `embedding_version`, run offline batch re-embed, then swap collection (blue-green or alias) for zero-downtime transition.

---

## 6. Retrieval + Generation (Pillar 3)

Tight request path. Max 8 chunks to LLM. Early-exit wherever possible.

### Flow Detail (intent=retrieve)
See §3 request path diagram above. Key design decisions:

1. **Hard chunk cap**: Maximum 8 chunks sent to LLM context (configurable). Never more, regardless of retrieval count. "More context ≠ better results."
2. **Decompose optimization**: The decomposer prompt itself decides sub-query count. For simple queries it returns a single sub-query `[{"query": original}]` (the NLP-LegalQA prompt already instructs "câu đơn giản thì 1 subquery là đủ"), avoiding a wasteful multi-search fan-out. No separate threshold/router integration needed.
3. **Metadata filter mandatory**: Every Qdrant query carries payload filters. No raw full-collection ANN ever.
4. **Early-exit patterns**:
   - Cache hit → skip search + rerank + gen.
   - Intent = reject/direct_answer → skip retrieval entirely.
   - Empty retrieval result → return "không tìm thấy", skip gen.
5. **Text2cypher self-healing**: On CypherSyntaxError, feed error message back to Gemini for one retry (matches NLP-LegalQA pattern).

### Cross-Encoder Reranker
Model: `AITeamVN/Vietnamese_Reranker`, FP16, max_length 2304. Runs on pool of `rerank_top` candidates (default 15). Batch processing with small batch_size for VRAM safety.

### Heuristic Post-Rerank
- `bãi bỏ` amendment type → score −5.0
- `thay thế` amendment type → score −3.0
- Recency bonus: `max(0, 2.0 − 0.3 × years_old)` based on document effect_date
- Sort descending, take top_k.

### Context Expansion (Optional)
- Sibling Points under the same Clause.
- Article/Clause descendant content (when Article itself is mostly title).
Flag-controlled (`expand`).

### Generation Prompt
Uses `_QA_FEW_SHOT_SYSTEM_PROMPT` from NLP-LegalQA (10 principles including faithfulness to context, handling overlapping docs by effect_date, correct citation format, abolished-provision tagging). Few-shot examples included.

---

## 7. Caching (Pillar 4)

Three-layer Redis cache. Real path: `query → cache → (miss) retrieve+LLM → write back`.

### Layer 1: Query → Answer Cache (FAQ/Repeat)
- Key: `ans:{sha256(normalized_query + ingestion_version)}`
- Value: full answer + sources + intent.
- TTL: long (e.g., 24h). Invalidation via `ingestion_version` in key — re-ingest bumps version, old keys auto-miss.
- HIT: skip router, retrieval, generation entirely. Highest-value cache.

### Layer 2: Query → Retrieval Cache (Top Chunks)
- Key: `ret:{sha256(decomposed_queries_sorted + filter_params + ingestion_version)}`
- Value: top_k chunks (uid + label + score + context snippet). Not the generated answer.
- TTL: medium (e.g., 6h).
- HIT: skip search + rerank, but still run generation (answer depends on current context + date).
- Caches the most expensive retrieval work while keeping answer freshness.

### Layer 3: Data/Index Cache (Hot Lookups)
- Cached hierarchy context strings: `ctx:{uid}` → formatted context.
- Cached amends lists: `amends:{uid}` → amends array.
- Reduces Neo4j round-trips for frequently accessed nodes.
- Short TTL + invalidated by ingestion_version.

### Query Normalization
Before hashing keys: lowercase, collapse whitespace, light stopword strip. Goal: near-duplicate questions map to the same key without losing semantic meaning.

### Invalidation Strategy
Version-based, not TTL-based. `ingestion_version` (Redis counter) bumped on each re-ingest. Version embedded in keys means stale entries naturally miss; no manual flush needed. Old entries expire via TTL.

Negative caching ("not found") avoided or given very short TTL to prevent stale emptiness after re-ingest adds data.

### Feedback Interaction
User 👎 does not immediately invalidate cache (one negative signal is insufficient). Logged to Langfuse. Queries accumulating multiple 👎 flagged for review; manual key invalidation available.

### Cache Metrics (Prometheus)
- `cache_hit_ratio` per layer (answer/retrieval/context).
- `cache_miss_latency` vs `cache_hit_latency`.
- These metrics demonstrate cost/latency savings on CV: "answer-cache hit 60% → 60% fewer LLM calls."

### Failure Mode
Redis down → fail-open (skip cache, run pipeline normally). Cache is optimization, not a hard dependency. Log + metric alert raised.

---

## 8. Monitoring + Evaluation (Pillar 5)

Two distinct subsystems: offline evaluation and online observability.

### A. Evaluation Harness (Offline)
Dataset: labeled QA set (`question`, `answer`, `reference[]` UIDs), analogous to NLP-LegalQA's `qa_dataset/`.

**Retrieval quality**: recall@k, precision@k, MRR, nDCG computed against reference UIDs. Run whenever embedding model, reranker, filter logic, or decompose prompt changes. Enables A/B comparison.

**Answer quality**: LLM-as-judge scores 5 criteria (legal_accuracy, citation correctness, completeness, hallucination absence, structure) using NLP-LegalQA's `_JUDGE_USER_PROMPT` framework. ROUGE/BERTScore as supplementary sanity checks.

**Output**: JSON report + pushed to Langfuse dataset/experiment for version-over-version tracking.

**Trigger**: Manual CLI or CI on retrieval-change PRs. Never run inline during serving.

### B. Online Observability

**Langfuse (LLM-specific)**:
- Trace every request with spans: router / rewrite / decompose / retrieve / rerank / gen, each with latency + tokens + cost.
- Attach user feedback score (👍/👎 from UI) to trace.
- Include retrieval UIDs + scores for debugging.

**Prometheus + Grafana (Infrastructure)**:
- Latency: request p50/p95/p99, per-stage breakdown.
- Cache hit-rate per layer (§7).
- Qdrant/Neo4j health: point count, query latency, index size.
- Ingestion: job success/fail rate, re-ingest watermark, docs/sec throughput.
- Request rate by intent.
- Alert rules: cache hit-rate drop, latency p95 threshold breach, consecutive ingestion failures.

### C. Data Drift Detection
Compare `ingestion_manifest` across runs: doc count delta, new/modified documents. Flag unexpected drift.

On embedding model or segmenter change: run offline eval BEFORE swapping production index (§5). Deploy only if recall does not regress.

### D. Feedback Loop Closure
```
user 👎 → Langfuse score → identify problematic queries
  → review → fix (prompt/filter/data) → re-evaluate offline
  → deploy → monitor 👎 rate decrease
```

### Demonstrable Metrics for Portfolio
- "recall@8 = X%, improved Y% after adding amendment filter"
- "answer-cache hit 60% → 60% reduction in LLM calls"
- "latency p95 < Z ms"
- "feedback-positive rate improved from A% to B% after iteration N"

---

## 9. Graph Schema + AMENDS

This is the domain differentiator. Schema largely matches NLP-LegalQA; validation upgraded to production grade.

### Node Labels (Hierarchy)
```
Document → Part → Chapter → Section → Article → Clause → Point
```
Levels are optional in real documents (decrees omit Part/Section). Parser attaches to nearest present ancestor; never invents placeholder levels.

Metadata nodes: DocumentGroup, DocumentType, EffectStatus, Organization, Signer, Field.

### UID Format (Single Source of Truth)
```
Document : doc_identity                            (56/2024/QH15)
Part     : {doc}::part::{n}
Chapter  : {doc}::part::{n}::chapter::{n}     OR   {doc}::chapter::{n}
Section  : ...::section::{n}
Article  : {doc}::article::{n}
Clause   : {doc}::article::{a}::clause::{n}
Point    : {doc}::article::{a}::clause::{c}::point::{letter}
```
Composite UID prevents collisions on reused numbers across documents (Điều 1 / Khoản 1 / điểm a appear in many docs). Same UID serves as Neo4j node key AND Qdrant point id.

### Relationships
```
Hierarchy : HAS_PART / HAS_CHAPTER / HAS_SECTION / HAS_ARTICLE / HAS_CLAUSE / HAS_POINT
Metadata  : BELONGS_TO_GROUP / HAS_TYPE / HAS_STATUS / ISSUED_BY / SIGNED_BY / IN_FIELD
Cross-doc : RELATED_TO  (from API docListOther)
          : AMENDS {type}  (from LLM extraction, NOT from API)
```

### AMENDS Edge
Source: LLM extraction from preamble + content using instruction prompt (derived from NLP-LegalQA's `data/amends/instruction.txt`). NOT derived from `docListOther` (which maps to `RELATED_TO` only).

Schema per amend entry:
```json
{
  "amending_doc_identity": "...",
  "amending_article": "...",
  "amending_clause": "...",
  "amending_point": null,
  "amend_type": "sửa đổi | bổ sung | bãi bỏ | thay thế | sửa đổi,bổ sung",
  "target_doc_identity": "...",
  "target_article": "...",
  "target_clause": "...",
  "target_point": null
}
```

Import resolves both endpoints to UIDs and executes `MERGE (src)-[:AMENDS {type}]->(tgt)`.

Usage in retrieval (§6):
- `bãi bỏ` / `thay thế` → heuristic penalty −5.0 / −3.0, tag `[ĐÃ BỊ BÃI BỎ]`.
- `sửa đổi,bổ sung` → tag `[LƯU Ý - NỘI DUNG SỬA ĐỔI]` with new content.

Evidence-gated: only create AMENDS when LLM extraction is unambiguous. No speculation.

### Production Upgrade vs Coursework
LLM AMENDS output is schema-validated (Pydantic) before import. Missing required fields or unresolvable targets → reject + log. NLP-LegalQA imports without validation; this project gates strictly.

### Constraints
13 uniqueness constraints matching NLP-LegalQA: `doc_identity` for Document, `id` for metadata nodes, `uid` for all hierarchy nodes. All MERGE-based, idempotent.

---

## 10. Error Handling

Philosophy: fail gracefully, never crash the request path. Each layer has a defined fallback.

### Request Path (Online)
| Component failure | Behavior |
|---|---|
| Router LLM | Fallback to intent `retrieve` (runs full pipeline) |
| Rewriter | Use original query, skip rewrite |
| Decomposer | Fallback to `[{"query": original}]` |
| Qdrant unavailable | Fallback to BM25-only via Neo4j fulltext (degraded but functional) |
| Neo4j unavailable | Fallback to vector-only via Qdrant |
| Both stores unavailable | Return friendly "system temporarily unavailable" message, log + alert |
| Reranker | Use ANN ordering, skip rerank |
| Generator LLM | Retry via tenacity; after exhaustion return friendly error message |
| Redis cache | Fail-open, skip cache, run pipeline |
| Langfuse/Prometheus | Fail-open, no request impact. Log locally |

Principle: observability and cache are optimizations; their failure must not kill requests. Stores and generator are true dependencies with degraded-but-functional fallbacks.

### Ingestion Path (Offline)
See §4 error handling table. Key distinction: scrape HTTP 200 with non-null `error` is treated as upstream error (retry), NOT as end-of-catalog.

### Cross-Cutting
- Structured JSON logging with `trace_id` spanning components → aligns with Langfuse traces.
- `/health` endpoint checks neo4j, qdrant, redis, llm reachability → returns per-component status. Scraped by Prometheus + consumed by UI.
- Timeouts on all LLM calls and DB queries. No hung requests.
- No blind retries on LLM write/generate (non-idempotent). Retry reads/searches only.

---

## 11. Testing Strategy

Layered pyramid focused on highest-risk logic (parser, retrieval, cache invalidation).

### Unit Tests (Many, Fast)
- Parser hierarchy: regex matching for Điều/Khoản/Điểm, footer split, quote-block tracking, NFC normalization, optional-level handling. Fixture-based.
- UID builders: all `build_*_uid` functions, edge cases around number collisions.
- AMENDS schema validation: valid/invalid JSON, missing fields, unresolvable targets → reject path.
- Cache key normalization + version: same query → same key; version bump → key change (miss).
- RRF aggregation: fuse multiple sub-query result sets correctly.
- Heuristic rerank: abolished penalties, recency calculation.
- Query normalization for cache.

### Integration Tests (Fewer, Require Services)
Run in docker-compose test profile or testcontainers (neo4j + qdrant + redis):
- Import parsed JSON → verify node/relationship counts + constraints hold.
- Batch embed → verify Qdrant point count + payload filter works.
- End-to-end retrieval on fixture document: query → expected top_k UIDs match reference.

Never touch production data. Isolated test database/collection.

### Evaluation (Quality Gate, Not Pass/Fail Test)
Recall@k / LLM-judge on labeled dataset (§8A). Run manually or in CI. Produces report, does not block merge but tracks regression over time.

### Contract / Smoke Tests
- `/health` returns all component statuses.
- `/chat` with fixture query → 200, answer present, sources present, Pydantic schema valid.
- Frontend: smoke check SSE/streaming endpoint only.

### Explicitly Not Tested
- LLM output quality (non-deterministic) → handled by eval harness, not unit tests.
- Detailed UI behavior (Next.js) → smoke only.

### CI
- `pytest` unit tests on every PR.
- Integration tests on parser/importer/retrieval changes (require docker).
- Eval runs manually or nightly; reports pushed to Langfuse.
