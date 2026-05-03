"""Pattern-match a trace for known failure-mode shapes.

Tags assigned in v1:

- ``INFINITE_LOOP`` — same ``(tool name, input)`` pair appears at least
  ``LOOP_THRESHOLD`` times anywhere in the trace. Catches non-adjacent
  oscillation that ``step_efficiency`` cannot see.
- ``CONTEXT_WINDOW_OVERFLOW`` — an ERROR step whose message matches a
  known context-limit phrasing.
- ``TOOL_CALL_ERROR`` — any ERROR step that does not match a more
  specific category.

Deferred to a later iteration: ``HALLUCINATED_TOOL_ARG``,
``TRUNCATED_TRACE`` (too noisy without finer heuristics).
"""

from __future__ import annotations

import json
import logging
from enum import Enum

from agentlens.evaluators.base import Evaluator, EvaluatorResult
from agentlens.schema import StepType, Trace

logger = logging.getLogger(__name__)

LOOP_THRESHOLD = 3

CONTEXT_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "context window",
    "context length",
    "context_length_exceeded",
    "maximum context",
    "token limit",
    "max tokens",
)


class FailureMode(str, Enum):
    """Failure-mode tags emitted by ``FailureModesEvaluator``."""

    INFINITE_LOOP = "infinite_loop"
    CONTEXT_WINDOW_OVERFLOW = "context_window_overflow"
    TOOL_CALL_ERROR = "tool_call_error"


class FailureModesEvaluator(Evaluator):
    """Tag a trace with detected failure-mode categories."""

    name = "failure_modes"

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        modes: list[FailureMode] = []
        if _has_infinite_loop(trace):
            modes.append(FailureMode.INFINITE_LOOP)
        modes.extend(_classify_error_steps(trace))
        modes = _dedupe_preserve_order(modes)

        passed = not modes
        details = (
            "No failure modes detected."
            if passed
            else f"Detected: {', '.join(m.value for m in modes)}."
        )
        return EvaluatorResult(
            evaluator=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
            metadata={"modes": [m.value for m in modes]},
        )


def _has_infinite_loop(trace: Trace, threshold: int = LOOP_THRESHOLD) -> bool:
    """Return True if any (name, input) tool-call key recurs `threshold`+ times."""
    counts: dict[tuple[str | None, str | None], int] = {}
    for step in trace.tool_calls():
        key = (step.name, _hashable(step.input))
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= threshold:
            return True
    return False


def _classify_error_steps(trace: Trace) -> list[FailureMode]:
    """Tag each ERROR step with its most specific failure mode."""
    modes: list[FailureMode] = []
    for step in trace.steps:
        if step.type != StepType.ERROR:
            continue
        msg = (step.error or "").lower()
        if any(p in msg for p in CONTEXT_OVERFLOW_PATTERNS):
            modes.append(FailureMode.CONTEXT_WINDOW_OVERFLOW)
        else:
            modes.append(FailureMode.TOOL_CALL_ERROR)
    return modes


def _hashable(value: object) -> str | None:
    """Stable string form of an arbitrary input value."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def _dedupe_preserve_order(modes: list[FailureMode]) -> list[FailureMode]:
    seen: set[FailureMode] = set()
    out: list[FailureMode] = []
    for m in modes:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
