"""OpenAI-compatible LLM client.

Wraps the OpenAI client to work with any OpenAI-compatible API
(OpenAI, OpenRouter, Ollama, vLLM, etc). The base_url and api_key
come from settings.
"""

from __future__ import annotations

import logging
import random
import re
import time

from openai import OpenAI, RateLimitError

from vtlaw.config import Settings, get_settings

log = logging.getLogger(__name__)

# Gemini reports the wait it wants inside the error body: "Please retry in 52.79s".
# Honouring that beats guessing, because the provider knows when the window rolls.
_RETRY_DELAY_RE = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)

# Cap on a server-suggested wait. A provider asking for longer than this is really
# saying the daily quota is gone, and blocking a request thread for minutes is
# worse than failing and letting the caller decide.
MAX_SERVER_DELAY_S = 65.0


def _suggested_delay(error: Exception) -> float | None:
    """Extract the provider's own retry hint, if it offered one."""
    match = _RETRY_DELAY_RE.search(str(error))
    if not match:
        return None
    try:
        return min(float(match.group(1)), MAX_SERVER_DELAY_S)
    except ValueError:
        return None


class LLMClient:
    """OpenAI-compatible chat completion client."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = OpenAI(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key,
            timeout=self._settings.llm_timeout_s,
            max_retries=0,  # retried here instead, honouring the provider's hint
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a completion from a list of messages.

        Retries on 429. A rate limit is a "wait and come back", not a failure:
        the free tier allows 10 requests/minute, so any loop over a dataset
        trips it, and without a retry every call after the first burst is lost.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature. Lower = more deterministic.
            max_tokens: Maximum tokens in the response. None lets the model
                decide its own limit (needed for Gemini which returns empty
                content when max_tokens is set too low).

        Returns:
            The generated text.

        Raises:
            RateLimitError: If still limited after the configured retries.
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

        attempts = self._settings.llm_max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except RateLimitError as exc:
                if attempt == attempts:
                    log.error("rate limited after %d attempts, giving up", attempts)
                    raise
                # Prefer the provider's own hint; fall back to exponential backoff.
                # The jitter matters when several workers hit the limit together —
                # without it they retry in lockstep and trip it again.
                delay = _suggested_delay(exc) or (
                    self._settings.llm_retry_base_s * 2 ** (attempt - 1)
                )
                delay += random.uniform(0, 0.5)
                log.warning(
                    "rate limited (attempt %d/%d), waiting %.1fs",
                    attempt,
                    attempts,
                    delay,
                )
                time.sleep(delay)

        content = response.choices[0].message.content or ""
        log.info("LLM response: %d chars", len(content))
        return content
