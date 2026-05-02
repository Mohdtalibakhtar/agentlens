"""Tool call accuracy evaluator.

Compares the ordered sequence of tool calls in a trace against the
``expected_tools`` list defined on the trace. Strict equality: same names,
same order, same length.
"""

from __future__ import annotations

import logging

from agentlens.evaluators.base import Evaluator, EvaluatorResult
from agentlens.schema import Trace

logger = logging.getLogger(__name__)


class ToolAccuracyEvaluator(Evaluator):
    """Strict ordered match of tool_call names against expected_tools."""

    name = "tool_accuracy"

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        """Return pass/fail based on exact ordered match of tool calls."""
        if trace.expected_tools is None:
            return EvaluatorResult(
                evaluator=self.name,
                passed=False,
                score=0.0,
                details="No expected_tools defined on trace; cannot evaluate.",
            )

        actual = [s.name or "" for s in trace.tool_calls()]
        expected = trace.expected_tools
        passed = actual == expected
        details = (
            f"Tool sequence matches expected ({len(expected)} calls)."
            if passed
            else f"Expected {expected}, got {actual}."
        )
        return EvaluatorResult(
            evaluator=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details,
            metadata={"expected": expected, "actual": actual},
        )
