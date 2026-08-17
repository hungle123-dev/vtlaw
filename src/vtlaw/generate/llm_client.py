"""OpenAI-compatible LLM client.

Wraps the OpenAI client to work with any OpenAI-compatible API
(OpenAI, OpenRouter, Ollama, vLLM, etc). The base_url and api_key
come from settings.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from vtlaw.config import Settings, get_settings

log = logging.getLogger(__name__)


class LLMClient:
    """OpenAI-compatible chat completion client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = OpenAI(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            timeout=self._settings.llm_timeout_s,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion from a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature. Lower = more deterministic.
            max_tokens: Maximum tokens in the response. None lets the model
                decide its own limit (needed for Gemini which returns empty
                content when max_tokens is set too low).

        Returns:
            The generated text.
        """
        log.info(
            "LLM request: model=%s messages=%d temp=%.2f",
            self._settings.llm_model,
            len(messages),
            temperature,
        )

        kwargs: dict = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        log.info("LLM response: %d chars", len(content))
        return content
