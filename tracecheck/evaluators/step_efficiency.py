"""Step efficiency evaluator.

Flags three kinds of waste in an agent trace:

- **Loops**: two adjacent tool_call steps with identical name and input.
- **Excess**: more tool_call steps than ``expected_tools`` length.
- **Retries**: any step with type RETRY.

Under-shooting (fewer tool calls than expected) is *not* inefficient —
that is a flow-correctness concern, handled by ``tool_accuracy``.
"""

from __future__ import annotations

import logging

from tracecheck.evaluators.base import Evaluator, EvaluatorResult
from tracecheck.schema import Step, StepType, Trace

logger = logging.getLogger(__name__)

LOOP_PENALTY = 0.5
EXCESS_PENALTY = 0.2
RETRY_PENALTY = 0.1


class StepEfficiencyEvaluator(Evaluator):
    """Score a trace for tool-call efficiency.

    Pure function of the trace: no LLM calls, no I/O.
    """

    name = "step_efficiency"

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        """Return a result with a graduated score and binary pass/fail."""
        tool_calls = trace.tool_calls()
        retries = sum(1 for s in trace.steps if s.type == StepType.RETRY)
        loops = _count_consecutive_duplicates(tool_calls)
        excess = _compute_excess(trace, tool_calls)

        penalties = (
            LOOP_PENALTY * loops
            + EXCESS_PENALTY * excess
            + RETRY_PENALTY * retries
        )
        score = max(0.0, 1.0 - penalties)
        passed = loops == 0 and excess == 0 and retries == 0

        return EvaluatorResult(
            evaluator=self.name,
            passed=passed,
            score=score,
            details=_format_details(loops, excess, retries, len(tool_calls)),
            metadata={
                "tool_call_count": len(tool_calls),
                "expected_count": (
                    len(trace.expected_tools)
                    if trace.expected_tools is not None
                    else None
                ),
                "loops": loops,
                "excess": excess,
                "retries": retries,
            },
        )


def _count_consecutive_duplicates(tool_calls: list[Step]) -> int:
    """Count adjacent tool_call pairs with identical name and input."""
    return sum(
        1
        for prev, curr in zip(tool_calls, tool_calls[1:])
        if prev.name == curr.name and prev.input == curr.input
    )


def _compute_excess(trace: Trace, tool_calls: list[Step]) -> int:
    """Return tool calls beyond the expected count, or 0 if no expectation."""
    if trace.expected_tools is None:
        return 0
    return max(0, len(tool_calls) - len(trace.expected_tools))


def _format_details(loops: int, excess: int, retries: int, tool_count: int) -> str:
    if loops == 0 and excess == 0 and retries == 0:
        return f"Efficient: {tool_count} tool call(s), no loops or retries."
    problems: list[str] = []
    if loops:
        problems.append(f"{loops} consecutive duplicate tool call(s)")
    if excess:
        problems.append(f"{excess} excess tool call(s) over expected")
    if retries:
        problems.append(f"{retries} retry step(s)")
    return "Inefficient: " + ", ".join(problems) + "."
