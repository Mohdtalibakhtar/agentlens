"""Tests for the Pydantic trace schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentlens.schema import Step, StepType, Trace


def test_minimal_trace_validates() -> None:
    trace = Trace(trace_id="t1", agent_name="a", steps=[])
    assert trace.trace_id == "t1"
    assert trace.steps == []
    assert trace.expected_tools is None


def test_tool_calls_filters_correctly() -> None:
    trace = Trace(
        trace_id="t1",
        agent_name="a",
        steps=[
            Step(type=StepType.LLM_CALL, output="thinking"),
            Step(type=StepType.TOOL_CALL, name="foo"),
            Step(type=StepType.ERROR, error="boom"),
            Step(type=StepType.TOOL_CALL, name="bar"),
        ],
    )
    assert [s.name for s in trace.tool_calls()] == ["foo", "bar"]


def test_step_type_serializes_to_string() -> None:
    step = Step(type=StepType.TOOL_CALL, name="foo")
    assert step.model_dump()["type"] == "tool_call"


def test_step_type_accepts_string_input() -> None:
    step = Step.model_validate({"type": "tool_call", "name": "foo"})
    assert step.type == StepType.TOOL_CALL


def test_invalid_step_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Step.model_validate({"type": "not_a_real_type"})


def test_trace_round_trips_through_json() -> None:
    original = Trace(
        trace_id="t1",
        agent_name="a",
        expected_tools=["x"],
        steps=[Step(type=StepType.TOOL_CALL, name="x")],
    )
    raw = original.model_dump_json()
    restored = Trace.model_validate_json(raw)
    assert restored == original
