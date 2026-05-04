"""Deterministic, offline judge for unit tests."""

from __future__ import annotations

from typing import Callable


class FakeJudge:
    """A judge that returns a canned response without making any LLM call.

    Pass either a static ``response`` string, or a callable that maps
    ``(system, user)`` to a response — useful for asserting per-input
    behavior in tests.

    Every call is recorded in ``self.calls`` so tests can verify what
    the evaluator actually sent to the judge.
    """

    name = "fake"

    def __init__(
        self,
        response: str | Callable[[str, str], str] = '{"on_topic": true, "score": 1.0, "reason": "stub"}',
    ) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def query(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if callable(self._response):
            return self._response(system, user)
        return self._response
