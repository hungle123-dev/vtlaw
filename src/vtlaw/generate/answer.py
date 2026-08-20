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
from vtlaw.generate.citations import assess_citations
from vtlaw.generate.context_builder import format_provision
from vtlaw.generate.llm_client import LLMClient
from vtlaw.generate.prompts import SYSTEM_PROMPT, build_user_prompt
from vtlaw.graph.client import GraphClient
from vtlaw.parse.models import uid_to_citation
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
        # Imported here, not at module scope: query_parser imports LLMClient from
        # this package, so a top-level import closes a cycle
        # (retrieve -> query_parser -> generate -> answer -> retrieve) that breaks
        # `import vtlaw.retrieve` outright.
        from vtlaw.retrieve.query_parser import QueryDecomposer

        self._decomposer = QueryDecomposer(self._llm)

    def answer(
        self,
        question: str,
        *,
        strategy: str = "hybrid",
        as_of: date | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        heuristic_rerank: bool = False,
        decompose: bool | None = None,
    ) -> Answer:
        """Generate an answer to a question.

        Args:
            question: The user's question.
            strategy: Retrieval strategy ("hybrid", "vector", or "bm25").
            as_of: The "current date" for temporal reasoning. Defaults to today.
            temperature: LLM sampling temperature.
            max_tokens: Maximum tokens in the generated answer.
            heuristic_rerank: Demote provisions a later document abolished or
                replaced. This is an amendment-annotation safeguard, not a
                consolidated-text engine, so it remains an explicit opt-in.
            decompose: Retrieve with LLM-generated sub-queries alongside the
                original phrasing. Defaults to ``settings.decompose_queries``.

        Returns:
            Answer with generated text and source provisions.
        """
        log.info("answering: %s (strategy=%s)", question[:80], strategy)

        use_decompose = (
            self._settings.decompose_queries if decompose is None else decompose
        )
        sub_queries = None
        if use_decompose:
            # Keep the user's words as a leg: an LLM paraphrase can improve
            # recall, but it can also lose a discriminating term.
            sub_queries = [question] + [
                sub["query"] for sub in self._decomposer.decompose(question)
            ]
            log.info("decomposed into %d phrasings", len(sub_queries))

        # Retrieve + rerank + heuristic
        result = self._retriever.search_and_rerank(
            question,
            k=self._settings.context_k,
            strategy=strategy,
            rerank_top=self._settings.rerank_top,
            heuristic_rerank=heuristic_rerank,
            as_of=as_of,
            sub_queries=sub_queries,
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
        text = self._enforce_grounded_citations(question, text, hits)

        log.info("answer: %d chars", len(text))
        return text

    def _enforce_grounded_citations(
        self, question: str, answer: str, hits: list[Hit]
    ) -> str:
        """Repair one invalid citation pass, then prefer deterministic evidence."""
        check = assess_citations(answer, hits)
        if check.status == "verified":
            return answer

        allowed = "\n\n".join(
            f"[{uid_to_citation(hit.uid)}]\n{hit.embedding_text}" for hit in hits
        )
        repair_prompt = f"""Viết lại câu trả lời pháp luật dưới đây.
Chỉ giữ các dữ kiện được hỗ trợ bởi ngữ cảnh trước đó và chỉ dùng các trích dẫn
trong danh sách cho phép. Phải có ít nhất một trích dẫn đầy đủ. Không giải thích
quy trình kiểm tra.

Câu hỏi: {question}

Câu trả lời cần sửa:
{answer}

Evidence và trích dẫn cho phép:
{allowed}"""
        try:
            repaired = self._llm.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn chỉ được sửa câu trả lời theo evidence đã cho.",
                    },
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure must not leak claims
            log.warning("citation repair failed: %s", exc)
            return self._grounded_fallback(hits)

        if assess_citations(repaired, hits).status == "verified":
            return repaired
        log.warning("citation repair remained ungrounded; returning source fallback")
        return self._grounded_fallback(hits)

    @staticmethod
    def _grounded_fallback(hits: list[Hit]) -> str:
        sources = "\n".join(
            f"- {uid_to_citation(hit.uid)}: {hit.content}" for hit in hits
        )
        return (
            "Tôi không thể xác minh một câu trả lời tự động chỉ từ các nguồn đã "
            "truy xuất. Dữ liệu đã truy xuất:\n" + sources
        )
