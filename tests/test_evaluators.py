"""Tests for the tool_accuracy evaluator."""

from __future__ import annotations

from agentlens.evaluators.tool_accuracy import ToolAccuracyEvaluator
from agentlens.schema import Step, StepType, Trace


def _trace(expected: list[str] | None, tool_names: list[str]) -> Trace:
    steps = [Step(type=StepType.TOOL_CALL, name=name) for name in tool_names]
    return Trace(
        trace_id="t",
        agent_name="a",
        expected_tools=expected,
        steps=steps,
    )


def test_passes_on_exact_match() -> None:
    result = ToolAccuracyEvaluator().evaluate(_trace(["a", "b", "c"], ["a", "b", "c"]))
    assert result.passed
    assert result.score == 1.0


def test_fails_on_wrong_order() -> None:
    result = ToolAccuracyEvaluator().evaluate(_trace(["a", "b", "c"], ["a", "c", "b"]))
    assert not result.passed
    assert result.score == 0.0


def test_fails_on_extra_tool() -> None:
    result = ToolAccuracyEvaluator().evaluate(_trace(["a", "b"], ["a", "b", "c"]))
    assert not result.passed


def test_fails_on_missing_tool() -> None:
    result = ToolAccuracyEvaluator().evaluate(_trace(["a", "b", "c"], ["a", "b"]))
    assert not result.passed


def test_skips_when_no_expected_tools() -> None:
    trace = Trace(trace_id="t", agent_name="a", steps=[])
    result = ToolAccuracyEvaluator().evaluate(trace)
    assert not result.passed
    assert "No expected_tools" in result.details


def test_ignores_non_tool_steps() -> None:
    trace = Trace(
        trace_id="t",
        agent_name="a",
        expected_tools=["a"],
        steps=[
            Step(type=StepType.LLM_CALL, output="thinking"),
            Step(type=StepType.TOOL_CALL, name="a"),
            Step(type=StepType.LLM_CALL, output="reply"),
        ],
    )
    result = ToolAccuracyEvaluator().evaluate(trace)
    assert result.passed


def test_metadata_includes_expected_and_actual() -> None:
    result = ToolAccuracyEvaluator().evaluate(_trace(["a"], ["b"]))
    assert result.metadata["expected"] == ["a"]
    assert result.metadata["actual"] == ["b"]
