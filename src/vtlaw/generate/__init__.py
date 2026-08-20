"""LLM generation: context building, prompt templates, and answer generation.

Stage 6 of the pipeline: takes retrieved provisions and generates a
grounded answer with citations.
"""

from vtlaw.generate.answer import AnswerGenerator
from vtlaw.generate.citations import CitationCheck, CitationStatus, assess_citations
from vtlaw.generate.context_builder import build_context, format_provision
from vtlaw.generate.llm_client import LLMClient
from vtlaw.generate.prompts import SYSTEM_PROMPT, build_user_prompt

__all__ = [
    "AnswerGenerator",
    "CitationCheck",
    "CitationStatus",
    "build_context",
    "format_provision",
    "LLMClient",
    "SYSTEM_PROMPT",
    "assess_citations",
    "build_user_prompt",
]
