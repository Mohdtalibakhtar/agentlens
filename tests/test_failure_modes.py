"""Tests for the failure_modes evaluator."""

from __future__ import annotations

from agentlens.evaluators.failure_modes import FailureModesEvaluator
from agentlens.schema import Step, StepType, Trace


def _trace(steps: list[Step]) -> Trace:
    return Trace(trace_id="t", agent_name="a", steps=steps)


def _tool(name: str, inp: dict | None = None) -> Step:
    return Step(type=StepType.TOOL_CALL, name=name, input=inp)


def test_no_failures_passes() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace([_tool("a"), _tool("b")])
    )
    assert result.passed
    assert result.score == 1.0
    assert result.metadata["modes"] == []


def test_three_consecutive_identical_calls_flagged_as_loop() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace([_tool("a", {"id": 1})] * 3)
    )
    assert not result.passed
    assert "infinite_loop" in result.metadata["modes"]


def test_non_adjacent_repetition_flagged_as_loop() -> None:
    """The thing step_efficiency cannot see: A,B,A,B,A pattern."""
    result = FailureModesEvaluator().evaluate(
        _trace(
            [
                _tool("a", {"id": 1}),
                _tool("b", {"q": "x"}),
                _tool("a", {"id": 1}),
                _tool("b", {"q": "y"}),
                _tool("a", {"id": 1}),
            ]
        )
    )
    assert not result.passed
    assert "infinite_loop" in result.metadata["modes"]


def test_two_repeats_below_threshold_not_a_loop() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace([_tool("a", {"id": 1}), _tool("a", {"id": 1})])
    )
    assert "infinite_loop" not in result.metadata["modes"]
    assert result.passed


def test_loop_with_different_inputs_not_detected() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace(
            [
                _tool("a", {"id": 1}),
                _tool("a", {"id": 2}),
                _tool("a", {"id": 3}),
            ]
        )
    )
    assert "infinite_loop" not in result.metadata["modes"]
    assert result.passed


def test_context_window_overflow_detected() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace([Step(type=StepType.ERROR, error="context window exceeded")])
    )
    assert "context_window_overflow" in result.metadata["modes"]


def test_token_limit_phrasing_detected_as_context_overflow() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace([Step(type=StepType.ERROR, error="Hit max tokens limit")])
    )
    assert "context_window_overflow" in result.metadata["modes"]


def test_generic_error_step_tagged_tool_call_error() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace([Step(type=StepType.ERROR, error="connection refused")])
    )
    assert "tool_call_error" in result.metadata["modes"]
    assert "context_window_overflow" not in result.metadata["modes"]


def test_multiple_modes_detected_simultaneously() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace(
            [
                _tool("get", {"id": 1}),
                _tool("get", {"id": 1}),
                _tool("get", {"id": 1}),
                Step(type=StepType.ERROR, error="context window exceeded"),
            ]
        )
    )
    modes = result.metadata["modes"]
    assert "infinite_loop" in modes
    assert "context_window_overflow" in modes


def test_modes_are_deduplicated() -> None:
    result = FailureModesEvaluator().evaluate(
        _trace(
            [
                Step(type=StepType.ERROR, error="some tool error"),
                Step(type=StepType.ERROR, error="another tool error"),
            ]
        )
    )
    assert result.metadata["modes"].count("tool_call_error") == 1


def test_score_is_binary() -> None:
    pass_trace = _trace([_tool("a")])
    fail_trace = _trace([Step(type=StepType.ERROR, error="oops")])
    assert FailureModesEvaluator().evaluate(pass_trace).score == 1.0
    assert FailureModesEvaluator().evaluate(fail_trace).score == 0.0
