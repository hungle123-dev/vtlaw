"""Answer generation: retrieve, build context, generate with LLM.

The AnswerGenerator orchestrates the full pipeline:
1. Retrieve relevant provisions (hybrid search + rerank + heuristic)
2. Build enriched context using graph hierarchy (walk UP/SIDEWAYS/DOWN)
3. Generate answer with LLM using system prompt + user prompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from vtlaw.config import Settings, get_settings
from vtlaw.embed import Embedder
from vtlaw.generate.context_builder import format_provision
from vtlaw.generate.llm_client import LLMClient
from vtlaw.generate.prompts import SYSTEM_PROMPT, build_user_prompt
from vtlaw.graph.client import GraphClient
from vtlaw.retrieve.context_builder import build_full_context
from vtlaw.retrieve.search import Hit, HybridRetriever

log = logging.getLogger(__name__)

NO_HITS_TEXT = "Tôi không tìm thấy quy định pháp luật liên quan đến câu hỏi này."


@dataclass
class Answer:
    """A generated answer with its source provisions."""

    text: str
    sources: list[Hit]
    strategy: str  # "hybrid" | "vector" | "bm25"
    reranked: bool

    def summary(self) -> str:
        tag = "reranked" if self.reranked else self.strategy
        if not self.sources:
            return f"{tag}: no sources"
        top = self.sources[0]
        return f"{tag}: {len(self.sources)} sources, top={top.uid}"


class AnswerGenerator:
    """Orchestrates retrieval + graph-enhanced context + generation."""

    def __init__(
        self,
        client: GraphClient,
        embedder: Embedder,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._retriever = HybridRetriever(client, embedder, self._settings)
        self._llm = llm or LLMClient(self._settings)

    def answer(
        self,
        question: str,
        *,
        strategy: str = "hybrid",
        as_of: date | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        heuristic_rerank: bool = True,
    ) -> Answer:
        """Generate an answer to a question.

        Args:
            question: The user's question.
            strategy: Retrieval strategy ("hybrid", "vector", or "bm25").
            as_of: The "current date" for temporal reasoning. Defaults to today.
            temperature: LLM sampling temperature.
            max_tokens: Maximum tokens in the generated answer.
            heuristic_rerank: Apply amendment penalty + recency bonus.

        Returns:
            Answer with generated text and source provisions.
        """
        log.info("answering: %s (strategy=%s)", question[:80], strategy)

        # Retrieve + rerank + heuristic
        result = self._retriever.search_and_rerank(
            question,
            k=self._settings.context_k,
            strategy=strategy,
            rerank_top=self._settings.rerank_top,
            heuristic_rerank=heuristic_rerank,
            as_of=as_of,
        )

        if not result.hits:
            log.info("no hits retrieved")
            return Answer(
                text=NO_HITS_TEXT,
                sources=[],
                strategy=strategy,
                reranked=result.reranked,
            )

        text = self.generate_from_hits(
            question,
            result.hits,
            as_of=as_of,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return Answer(
            text=text,
            sources=result.hits,
            strategy=result.retrieval_score,
            reranked=result.reranked,
        )

    def generate_from_hits(
        self,
        question: str,
        hits: list[Hit],
        *,
        as_of: date | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Generate answer text for hits that were retrieved elsewhere.

        Split out of :meth:`answer` so a caller holding cached hits (the API's
        retrieval cache) can generate without re-running retrieval.
        """
        if not hits:
            return NO_HITS_TEXT

        # Build enriched context using graph hierarchy
        enriched = build_full_context(self._client, hits, as_of=as_of)
        log.info(
            "enriched context: %d chars from %d provisions",
            sum(len(v) for v in enriched.values()),
            len(hits),
        )

        # Assemble LLM prompt with enriched context
        today = (as_of or date.today()).isoformat()
        parts = [f"Ngày hiện tại: {today}", ""]
        parts.append(f"Các quy định pháp luật liên quan ({len(hits)} điều):")
        parts.append("")

        for i, hit in enumerate(hits, 1):
            parts.append(f"{i}. {enriched.get(hit.uid) or format_provision(hit)}")
            parts.append("")

        user_prompt = build_user_prompt(question, "\n".join(parts))
        text = self._llm.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        log.info("answer: %d chars", len(text))
        return text
