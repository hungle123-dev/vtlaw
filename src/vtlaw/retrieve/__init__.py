"""Stage 5 — hybrid retrieval.

Vector search finds provisions by meaning; BM25 finds them by keyword.
Neither alone is enough: a query about "không đội mũ bảo hiểm" may
semantic-match the wrong penalty, while BM25 misses paraphrases. Fusing
both with Reciprocal Rank Fusion (RRF) gets the best of both.

A cross-encoder reranker then scores the top candidates against the query
directly — it reads the full query+provision pair, not just their vectors,
so it catches mismatches that a vector similarity score cannot.

After reranking, provisions that a later document abolished or replaced are
demoted, so superseded law does not get cited as current.

Graph-enhanced context building walks the hierarchy UP (Document
→ Article → Clause → Point), SIDEWAYS (sibling Points), and DOWN (children
Clauses/Points) to assemble full legal context for each hit — because a
Point on its own states the offence, but the penalty amount lives in its
parent Clause.

Query decomposition is opt-in because it needs an LLM call; the deterministic
hybrid path remains the default retrieval baseline.
"""

from vtlaw.retrieve.context_builder import (
    build_full_context,
    fetch_children_context,
    fetch_hierarchy,
    fetch_sibling_points,
)
from vtlaw.retrieve.heuristics import (
    ABOLISHED_PENALTY,
    REPLACED_PENALTY,
    apply_heuristic_rerank,
    fetch_abolished_uids,
)
from vtlaw.retrieve.query_parser import QueryDecomposer, SubQuery
from vtlaw.retrieve.router import Intent, QueryRouter
from vtlaw.retrieve.search import (
    Hit,
    HybridRetriever,
    RetrievalResult,
    SearchResult,
    fuse_weighted,
)

__all__ = [
    "ABOLISHED_PENALTY",
    "Hit",
    "HybridRetriever",
    "Intent",
    "QueryDecomposer",
    "QueryRouter",
    "REPLACED_PENALTY",
    "RetrievalResult",
    "SearchResult",
    "SubQuery",
    "apply_heuristic_rerank",
    "build_full_context",
    "fetch_abolished_uids",
    "fetch_children_context",
    "fetch_hierarchy",
    "fetch_sibling_points",
    "fuse_weighted",
]
