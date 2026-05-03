"""Tests for the step_efficiency evaluator."""

from __future__ import annotations

import pytest

from agentlens.evaluators.step_efficiency import StepEfficiencyEvaluator
from agentlens.schema import Step, StepType, Trace


def _tool(name: str, inp: dict | None = None) -> Step:
    return Step(type=StepType.TOOL_CALL, name=name, input=inp)


def _trace(steps: list[Step], expected: list[str] | None = None) -> Trace:
    return Trace(trace_id="t", agent_name="a", expected_tools=expected, steps=steps)


def test_efficient_trace_passes() -> None:
    trace = _trace(
        [_tool("a", {"x": 1}), _tool("b", {"y": 2})],
        expected=["a", "b"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert result.passed
    assert result.score == 1.0


def test_consecutive_duplicate_calls_flagged_as_loop() -> None:
    trace = _trace(
        [
            _tool("get", {"id": 1}),
            _tool("get", {"id": 1}),
            _tool("get", {"id": 1}),
        ],
        expected=["get"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert not result.passed
    assert result.metadata["loops"] == 2


def test_different_inputs_are_not_a_loop() -> None:
    trace = _trace(
        [_tool("get", {"id": 1}), _tool("get", {"id": 2})],
        expected=["get", "get"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert result.metadata["loops"] == 0
    assert result.passed


def test_excess_tool_calls_flagged() -> None:
    trace = _trace(
        [_tool("a"), _tool("b"), _tool("c")],
        expected=["a", "b"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert not result.passed
    assert result.metadata["excess"] == 1


def test_under_shooting_is_not_inefficient() -> None:
    trace = _trace([_tool("a")], expected=["a", "b", "c"])
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert result.passed
    assert result.metadata["excess"] == 0


def test_retry_steps_flagged() -> None:
    trace = _trace(
        [
            _tool("a"),
            Step(type=StepType.RETRY, name="a"),
            _tool("a", {"x": 1}),
        ],
        expected=["a"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert not result.passed
    assert result.metadata["retries"] == 1


def test_no_expected_tools_only_checks_loops_and_retries() -> None:
    trace = _trace([_tool("a", {"x": 1}), _tool("a", {"x": 1})])
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert not result.passed
    assert result.metadata["expected_count"] is None
    assert result.metadata["loops"] == 1


def test_score_combines_loop_and_excess_penalties() -> None:
    trace = _trace(
        [_tool("get", {"id": 1}), _tool("get", {"id": 1})],
        expected=["get"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    # 1 loop (penalty 0.5) + 1 excess (penalty 0.2) -> score 0.3
    assert result.score == pytest.approx(0.3)


def test_score_clamped_to_zero() -> None:
    trace = _trace(
        [_tool("a", {"x": 1})] * 10,
        expected=["a"],
    )
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert result.score == 0.0


def test_metadata_includes_all_counts() -> None:
    trace = _trace([_tool("a"), _tool("b")], expected=["a"])
    result = StepEfficiencyEvaluator().evaluate(trace)
    assert result.metadata["tool_call_count"] == 2
    assert result.metadata["expected_count"] == 1
    assert result.metadata["loops"] == 0
    assert result.metadata["excess"] == 1
    assert result.metadata["retries"] == 0
