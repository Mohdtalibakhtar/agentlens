"""Anthropic Claude as the judge model.

The system prompt is the same for every trace, so we cache it via
``cache_control``. After the first request the cached prefix costs
~10% of full token rate, which matters when judging hundreds of traces
in a single eval run.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1024


class AnthropicJudge:
    """Judge backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicJudge requires the anthropic package. "
                "Install with: pip install tracecheck[anthropic]"
            ) from exc

        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def query(self, system: str, user: str) -> str:
        """Send the prompt with prompt-caching on the system block."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
