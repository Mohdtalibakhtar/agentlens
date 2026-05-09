"""Judge protocol — the contract every LLM-as-judge backend must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Judge(Protocol):
    """A pluggable LLM-as-judge backend.

    Implementations: :class:`tracecheck.judges.fake.FakeJudge` for tests,
    :class:`tracecheck.judges.anthropic.AnthropicJudge` for production.
    """

    name: str

    def query(self, system: str, user: str) -> str:
        """Send the prompt and return the model's raw text response.

        Args:
            system: System-level instructions (cached when the backend
                supports it, since this is invariant across many traces).
            user: User message containing the trace digest under evaluation.

        Returns:
            Raw model response text. Each evaluator parses its own
            structured verdict out of this.
        """
        ...
