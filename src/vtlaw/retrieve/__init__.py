"""Stage 5 — hybrid retrieval.

Vector search finds provisions by meaning; BM25 finds them by keyword.
Neither alone is enough: a query about "không đội mũ bảo hiểm" may
semantic-match the wrong penalty, while BM25 misses paraphrases. Fusing
both with Reciprocal Rank Fusion (RRF) gets the best of both.

A cross-encoder reranker then scores the top candidates against the query
directly — it reads the full query+provision pair, not just their vectors,
so it catches mismatches that a vector similarity score cannot.

After reranking, a legal-safety heuristic penalises provisions that have
been abolished or replaced by newer amendments, and gives a small recency
bonus to documents with recent effect dates.

Graph-enhanced context building walks the hierarchy UP (Document
→ Article → Clause → Point), SIDEWAYS (sibling Points), and DOWN (children
Clauses/Points) to assemble full legal context for each hit — because a
Point on its own states the offence, but the penalty amount lives in its
parent Clause.

Query pipeline modules (router, parser, rewriter, text2cypher) enable
intelligent query routing, decomposition, and multi-turn conversation support.
"""

from vtlaw.retrieve.context_builder import (
    build_full_context,
    fetch_children_context,
    fetch_hierarchy,
    fetch_sibling_points,
)
from vtlaw.retrieve.heuristics import (
    ABOLISHED_PENALTY,
    RECENCY_DECAY_PER_YEAR,
    RECENCY_INITIAL_BONUS,
    REPLACED_PENALTY,
    apply_heuristic_rerank,
    fetch_abolished_uids,
    fetch_amend_status,
    fetch_doc_effect_dates,
)
from vtlaw.retrieve.query_parser import QueryDecomposer, SubQuery
from vtlaw.retrieve.query_rewriter import ChatMessage, QueryRewriter
from vtlaw.retrieve.router import IntentType, QueryRouter
from vtlaw.retrieve.search import Hit, HybridRetriever, RetrievalResult, SearchResult
from vtlaw.retrieve.text2cypher import TextToCypher

__all__ = [
    "ABOLISHED_PENALTY",
    "ChatMessage",
    "Hit",
    "HybridRetriever",
    "IntentType",
    "QueryDecomposer",
    "QueryRewriter",
    "QueryRouter",
    "RECENCY_DECAY_PER_YEAR",
    "RECENCY_INITIAL_BONUS",
    "REPLACED_PENALTY",
    "RetrievalResult",
    "SearchResult",
    "SubQuery",
    "TextToCypher",
    "apply_heuristic_rerank",
    "build_full_context",
    "fetch_abolished_uids",
    "fetch_amend_status",
    "fetch_children_context",
    "fetch_doc_effect_dates",
    "fetch_hierarchy",
    "fetch_sibling_points",
]
