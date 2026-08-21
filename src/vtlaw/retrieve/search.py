"""Hybrid retrieval: vector search + BM25 + rerank.

Three strategies, each covering what the others miss:

1. **Vector search** — finds provisions by meaning. Query "không đội mũ bảo
   hiểm" matches provisions about "người ngồi trên xe không đội mũ bảo hiểm"
   even though the words differ.

2. **BM25 keyword search** — finds provisions by exact terms. Query "168/2024"
   matches only documents with that identity; vector search would return
   anything topically similar.

3. **Cross-encoder rerank** — scores the top candidates against the query
   directly, reading the full pair rather than comparing two vectors. Catches
   mismatches that vector similarity alone cannot.

Results are fused with **Reciprocal Rank Fusion (RRF)**: each strategy ranks
candidates independently, and the final score is the sum of
``1 / (rank + K)`` across all strategies that returned a candidate.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Literal

from neo4j import Session

from vtlaw.config import RERANK_MODEL_REVISION, Settings, get_settings
from vtlaw.embed import Embedder, segment
from vtlaw.graph.client import GraphClient
from vtlaw.graph.schema import PROVISION_LABELS
from vtlaw.parse import article_uid, clause_uid, point_uid
from vtlaw.parse.patterns import RE_DOC_IDENTITY

log = logging.getLogger(__name__)

# Standard RRF damping constant. Larger K makes the ranking more democratic
# (rank differences matter less); smaller K makes it more sensitive to top ranks.
RRF_K = 60

# Cross-encoder input cap. The reranker config advertises 8194 positions; left
# unpinned, every pair pads to that and CPU inference becomes unusable.
RERANK_MAX_LENGTH = 256
# A CPU cross-encoder is loaded beside the bi-encoder already resident in the
# API. Refuse the optional quality profile before a native model load can take
# down a memory-constrained Windows host.
RERANK_MIN_AVAILABLE_MEMORY_BYTES = 2 * 1024**3


def available_memory_bytes() -> int | None:
    """Return currently available physical memory, or ``None`` if unknown."""
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


@dataclass(frozen=True)
class Hit:
    """One candidate provision."""

    uid: str
    score: float
    doc_identity: str
    label: Literal["Article", "Clause", "Point"]
    content: str
    title: str | None = None

    @property
    def embedding_text(self) -> str:
        """Rebuild the text that was embedded (used for reranking)."""
        if self.title:
            return f"{self.title}\n{self.content}".strip()
        return self.content


@dataclass
class SearchResult:
    """Final retrieval result for one query."""

    query: str
    hits: list[Hit]
    strategy: str = "hybrid"  # "hybrid" | "vector" | "bm25" | "exact_citation"

    def summary(self) -> str:
        if not self.hits:
            return f"{self.strategy}: no results"
        top = self.hits[0]
        return (
            f"{self.strategy}: {len(self.hits)} hits, "
            f"top={top.uid} score={top.score:.3f}"
        )


@dataclass(frozen=True)
class CitationTarget:
    """Most-specific provision named by an explicit legal citation."""

    uid: str
    label: Literal["Article", "Clause", "Point"]


@dataclass(frozen=True)
class CitationLookupResult:
    """Whether an exact citation was recognized, independently of its hits."""

    had_full_citation: bool
    hits: list[Hit]


@dataclass
class RetrievalResult:
    """Result after rerank."""

    query: str
    hits: list[Hit]
    retrieval_score: str  # "hybrid" | "vector" | "bm25"
    reranked: bool

    def summary(self) -> str:
        tag = "reranked" if self.reranked else self.retrieval_score
        if not self.hits:
            return f"{tag}: no results"
        top = self.hits[0]
        return f"{tag}: {len(self.hits)} hits, top={top.uid} score={top.score:.3f}"


_CITATION_DOCUMENT = RE_DOC_IDENTITY
_CITATION_ARTICLE = re.compile(r"\bđiều\s+(?P<number>\d+[a-z]?)\b", re.IGNORECASE)
_CITATION_CLAUSE = re.compile(r"\bkhoản\s+(?P<number>\d+[a-z]?)\b", re.IGNORECASE)
_CITATION_POINT = re.compile(r"\bđiểm\s+(?P<letter>[a-zđ])\b", re.IGNORECASE)


def resolve_citation(query: str) -> CitationTarget | None:
    """Resolve a full Vietnamese document citation without an LLM.

    A document identity and article are deliberately required: a bare ``Điều 6``
    is ambiguous across this corpus and must remain a normal hybrid query.
    """
    document = _CITATION_DOCUMENT.search(query)
    article = _CITATION_ARTICLE.search(query)
    if document is None or article is None:
        return None

    doc_identity = document.group("doc").upper()
    article_number = article.group("number").lower()
    clause = _CITATION_CLAUSE.search(query)
    point = _CITATION_POINT.search(query)
    if clause is None:
        return CitationTarget(article_uid(doc_identity, article_number), "Article")

    clause_number = clause.group("number").lower()
    if point is None:
        return CitationTarget(
            clause_uid(doc_identity, article_number, clause_number), "Clause"
        )
    return CitationTarget(
        point_uid(doc_identity, article_number, clause_number, point.group("letter").lower()),
        "Point",
    )


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


def vector_search(
    session: Session,
    embedder: Embedder,
    query: str,
    k: int,
    *,
    as_of: date | None = None,
    temporal: bool = True,
    overfetch: int = 4,
) -> list[Hit]:
    """Vector similarity search against all provision labels."""
    vector = embedder.encode_query(query)
    cutoff = (as_of or date.today()).isoformat()
    hits: list[Hit] = []

    for label in PROVISION_LABELS:
        # Vector index has no pre-filter, so overfetch before filtering.
        # A document that is currently expired may still rank high by vector
        # similarity; the filter removes it afterwards.
        rows = session.run(
            """
            CALL db.index.vector.queryNodes($index, $k * $overfetch, $vector)
            YIELD node, score
            MATCH (d:Document {doc_identity: node.doc_identity})
            WHERE $temporal = false OR (
                d.effect_date <= date($as_of)
                AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
            )
            RETURN node.uid AS uid, node.doc_identity AS doc_identity,
                   node.content AS content, node.title AS title, score
            ORDER BY score DESC
            LIMIT $k
            """,
            index=f"{label.lower()}_embedding",
            k=k,
            overfetch=overfetch,
            vector=vector,
            as_of=cutoff,
            temporal=temporal,
        ).data()

        for row in rows:
            hits.append(
                Hit(
                    uid=row["uid"],
                    score=row["score"],
                    doc_identity=row["doc_identity"],
                    label=label,  # type: ignore[arg-type]
                    content=row["content"] or "",
                    title=row.get("title"),
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


# Lucene's classic query parser treats these as syntax. A user question is not a
# query expression — it is text to match — so every one of them must be escaped
# before it reaches db.index.fulltext.queryNodes. Measured on QA_Part2345: 4 of
# 200 questions crashed the procedure outright, and '/' was in all four. An
# unpaired '/' opens a regex literal, so the parser reads to end-of-input looking
# for the close and raises TokenMgrError: "Encountered <EOF> after prefix ...".
_LUCENE_SPECIAL = r'+-&|!(){}[]^"~*?:\/'


def escape_lucene(text: str) -> str:
    """Escape Lucene query syntax so a question is matched as text, not parsed.

    Also neutralises the boolean keywords, which Lucene reads as operators in
    upper case even when they are ordinary words in the surrounding sentence.
    """
    out = []
    for ch in text:
        if ch in _LUCENE_SPECIAL:
            out.append("\\")
        out.append(ch)
    escaped = "".join(out)
    # AND/OR/NOT are operators only in upper case; lower-casing them keeps the
    # term while dropping the operator meaning.
    for keyword in (" AND ", " OR ", " NOT "):
        escaped = escaped.replace(keyword, keyword.lower())
    return escaped


def bm25_search(
    session: Session,
    query: str,
    k: int,
    *,
    as_of: date | None = None,
    temporal: bool = True,
    overfetch: int = 4,
) -> list[Hit]:
    """BM25 keyword search against all provision labels."""
    # Tokenize for BM25: same segmentation as the embedding model expects, then
    # escape — segmentation can introduce nothing Lucene cares about, but the
    # question itself carries slashes ("25km/h"), colons and parentheses.
    tokenised = escape_lucene(segment(query))
    cutoff = (as_of or date.today()).isoformat()
    hits: list[Hit] = []

    for label in PROVISION_LABELS:
        # Fulltext index also has no pre-filter, so overfetch before filtering.
        rows = session.run(
            """
            CALL db.index.fulltext.queryNodes($index, $search_text)
            YIELD node, score
            MATCH (d:Document {doc_identity: node.doc_identity})
            WHERE $temporal = false OR (
                d.effect_date <= date($as_of)
                AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
            )
            RETURN node.uid AS uid, node.doc_identity AS doc_identity,
                   node.content AS content, node.title AS title, score
            ORDER BY score DESC
            LIMIT $k
            """,
            index=f"{label.lower()}_fulltext",
            search_text=tokenised,
            k=k * overfetch,  # overfetch before filtering
            as_of=cutoff,
            temporal=temporal,
        ).data()

        for row in rows[:k]:
            hits.append(
                Hit(
                    uid=row["uid"],
                    score=row["score"],
                    doc_identity=row["doc_identity"],
                    label=label,  # type: ignore[arg-type]
                    content=row["content"] or "",
                    title=row.get("title"),
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


def citation_search(
    client: GraphClient,
    query: str,
    *,
    as_of: date | None = None,
    temporal: bool = True,
) -> CitationLookupResult:
    """Look up an explicit Điều/Khoản/Điểm citation exactly.

    When ``temporal`` is true, the same document-effective filter as approximate
    retrieval applies. Label-retrieval benchmarks can disable it explicitly
    when their questions do not carry a legal date.
    """
    target = resolve_citation(query)
    if target is None:
        return CitationLookupResult(had_full_citation=False, hits=[])

    cutoff = (as_of or date.today()).isoformat()
    with client.session() as session:
        rows = session.run(
            f"""
            MATCH (n:{target.label} {{uid: $uid}})
            MATCH (d:Document {{doc_identity: n.doc_identity}})
            WHERE $temporal = false OR (
                d.effect_date <= date($as_of)
                AND (d.expire_date IS NULL OR d.expire_date >= date($as_of))
            )
            RETURN n.uid AS uid, n.doc_identity AS doc_identity,
                   n.content AS content, n.title AS title
            """,
            uid=target.uid,
            as_of=cutoff,
            temporal=temporal,
        ).data()

    return CitationLookupResult(
        had_full_citation=True,
        hits=[
            Hit(
                uid=row["uid"],
                score=1.0,
                doc_identity=row["doc_identity"],
                label=target.label,
                content=row["content"] or "",
                title=row.get("title"),
            )
            for row in rows
        ],
    )


# ---------------------------------------------------------------------------
# Fusion and reranking
# ---------------------------------------------------------------------------


def fuse_weighted(
    legs: list[tuple[list[Hit], float]],
    k: int,
    *,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    """Reciprocal Rank Fusion over any number of weighted result lists.

    ``legs`` pairs each ranked list with its weight. Two legs is the usual case
    (vector + BM25); query decomposition produces two per sub-query, and fusing
    them all lets an alternate phrasing surface a provision the original wording
    missed.
    """
    scores: dict[str, float] = {}
    by_uid: dict[str, Hit] = {}

    for hits, weight in legs:
        for rank, hit in enumerate(hits):
            scores[hit.uid] = scores.get(hit.uid, 0.0) + weight / (rank + 1 + rrf_k)
            by_uid.setdefault(hit.uid, hit)

    fused = [
        Hit(
            uid=uid,
            score=score,
            doc_identity=by_uid[uid].doc_identity,
            label=by_uid[uid].label,
            content=by_uid[uid].content,
            title=by_uid[uid].title,
        )
        for uid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return fused[:k]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, revision: str):
    """Load and cache the cross-encoder.

    Cached because the caller is per-request: loading XLM-RoBERTa fresh on every
    query cost ~15s and the API segfaulted holding two copies in memory.
    """
    from sentence_transformers import CrossEncoder

    log.info("loading cross-encoder %s at %s", model_name, revision)
    # max_length pinned: the model reports max_position_embeddings=8194, so the
    # default would pad pairs to 8k tokens on CPU.
    return CrossEncoder(model_name, revision=revision, max_length=RERANK_MAX_LENGTH)


def rerank(
    hits: list[Hit],
    query: str,
    k: int,
    reranker_model: str = "AITeamVN/Vietnamese_Reranker",
    reranker_revision: str = RERANK_MODEL_REVISION,
    min_available_memory_bytes: int = RERANK_MIN_AVAILABLE_MEMORY_BYTES,
) -> list[Hit] | None:
    """Cross-encoder rerank: score (query, provision) pairs directly.

    The cross-encoder reads the full pair rather than comparing two vectors,
    so it catches fine-grained mismatches that a vector similarity score
    cannot. Slower than vector search, so only applied to the top candidates.
    Returns ``None`` when the model cannot run so callers do not misreport a
    fallback as a successful rerank.
    """
    if not hits:
        return hits[:k]
    available_memory = available_memory_bytes()
    if (
        available_memory is not None
        and available_memory < min_available_memory_bytes
    ):
        log.warning(
            "reranker skipped: %d MiB available, %d MiB required before model load",
            available_memory // 1024**2,
            min_available_memory_bytes // 1024**2,
        )
        return None

    try:
        model = _load_cross_encoder(reranker_model, reranker_revision)
        pairs = [(query, hit.embedding_text) for hit in hits]
        scores = model.predict(pairs)
    except (ImportError, ValueError, OSError) as exc:
        log.warning("reranker unavailable (%s); skipping rerank", exc)
        return None

    reranked = [
        Hit(
            uid=hit.uid,
            score=float(score),
            doc_identity=hit.doc_identity,
            label=hit.label,
            content=hit.content,
            title=hit.title,
        )
        for hit, score in zip(hits, scores, strict=False)
    ]
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked[:k]


# ---------------------------------------------------------------------------
# Retriever facade
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Facade for hybrid retrieval with optional rerank."""

    def __init__(
        self,
        client: GraphClient,
        embedder: Embedder,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = settings or get_settings()

    def search(
        self,
        query: str,
        k: int = 10,
        strategy: Literal["hybrid", "vector", "bm25"] = "hybrid",
        fetch_k: int | None = None,
        sub_queries: list[str] | None = None,
        as_of: date | None = None,
        temporal: bool = True,
    ) -> SearchResult:
        """Run retrieval with the chosen strategy.

        Each leg fetches ``fetch_k`` candidates and only the fused list is cut to
        ``k``. Fetching only ``k`` per leg starves fusion of candidates that the
        other leg could promote.

        ``sub_queries`` runs one retrieval pass per phrasing and fuses them all.
        A decomposed question can surface provisions the original wording misses.
        Include the original query in the list to keep its exact terms.
        """
        citation = citation_search(
            self._client, query, as_of=as_of, temporal=temporal
        )
        if citation.had_full_citation:
            return SearchResult(
                query=query,
                hits=citation.hits[:k],
                strategy="exact_citation",
            )

        pool = max(fetch_k or self._settings.fetch_k, k)
        phrasings = sub_queries or [query]
        overfetch = self._settings.overfetch_factor

        with self._client.session() as session:
            if strategy == "vector":
                legs = [
                    (
                        vector_search(
                            session, self._embedder, q, pool,
                            as_of=as_of, temporal=temporal, overfetch=overfetch,
                        ),
                        1.0,
                    )
                    for q in phrasings
                ]
            elif strategy == "bm25":
                legs = [
                    (
                        bm25_search(
                            session, q, pool, as_of=as_of, temporal=temporal,
                            overfetch=overfetch,
                        ),
                        1.0,
                    )
                    for q in phrasings
                ]
            else:  # hybrid
                legs = []
                for q in phrasings:
                    legs.append((
                        vector_search(
                            session, self._embedder, q, pool,
                            as_of=as_of, temporal=temporal, overfetch=overfetch,
                        ),
                        self._settings.rrf_vector_weight,
                    ))
                    legs.append((
                        bm25_search(
                            session, q, pool, as_of=as_of, temporal=temporal,
                            overfetch=overfetch,
                        ),
                        self._settings.rrf_bm25_weight,
                    ))

        hits = fuse_weighted(legs, pool, rrf_k=self._settings.rrf_k)
        return SearchResult(query=query, hits=hits[:k], strategy=strategy)

    def search_and_rerank(
        self,
        query: str,
        k: int = 10,
        strategy: Literal["hybrid", "vector", "bm25"] = "hybrid",
        rerank_top: int | None = None,
        fetch_k: int | None = None,
        reranker_model: str | None = None,
        rerank_enabled: bool | None = None,
        heuristic_rerank: bool = False,
        as_of: date | None = None,
        temporal: bool = True,
        sub_queries: list[str] | None = None,
    ) -> RetrievalResult:
        """Retrieve, rerank, and optionally demote superseded provisions.

        Args:
            query: The user query.
            k: Number of hits to return.
            strategy: "hybrid", "vector", or "bm25".
            rerank_top: Number of candidates to rerank with the cross-encoder.
                The returned result still contains only ``k`` hits.
            fetch_k: Candidates fetched by each retrieval leg before fusion.
            reranker_model: Cross-encoder model name.
            rerank_enabled: Override the configured cross-encoder opt-in.
            heuristic_rerank: If True, demote provisions an amendment has
                abolished or replaced across the candidate pool before returning
                the requested top-k.
            as_of: Legal-effective cutoff applied when ``temporal`` is true.
                ``None`` then uses the current date.
            temporal: Enable document-effective filtering and amendment
                demotion. Disable only for a dated-label benchmark that has no
                legal date; graph hierarchy and hybrid fusion remain enabled.
            sub_queries: Extra phrasings to retrieve with, fused into one list.
                See :meth:`search`.

        Returns:
            RetrievalResult with reranked hits.
        """
        should_rerank = (
            self._settings.rerank_enabled if rerank_enabled is None else rerank_enabled
        )
        candidate_k = max(k, rerank_top or self._settings.rerank_top) if should_rerank else k
        if heuristic_rerank and temporal:
            candidate_k = max(candidate_k, fetch_k or self._settings.fetch_k)
        result = self.search(
            query,
            k=candidate_k,
            strategy=strategy,
            fetch_k=fetch_k,
            sub_queries=sub_queries,
            as_of=as_of,
            temporal=temporal,
        )

        if not result.hits:
            return RetrievalResult(
                query=query, hits=[], retrieval_score=result.strategy, reranked=False
            )

        if should_rerank:
            reranker = reranker_model or self._settings.rerank_model
            reranked_hits = rerank(
                result.hits,
                query,
                candidate_k if heuristic_rerank else k,
                reranker,
                self._settings.rerank_model_revision,
                self._settings.rerank_min_available_memory_mb * 1024**2,
            )
            if reranked_hits is None:
                reranked_hits = result.hits
                should_rerank = False
        else:
            reranked_hits = result.hits

        # Demote provisions a later document abolished or replaced.
        if heuristic_rerank and temporal:
            from vtlaw.retrieve.heuristics import apply_heuristic_rerank
            reranked_hits = apply_heuristic_rerank(
                reranked_hits, self._client, as_of=as_of
            )

        return RetrievalResult(
            query=query,
            hits=reranked_hits[:k],
            retrieval_score=result.strategy,
            reranked=should_rerank,
        )
