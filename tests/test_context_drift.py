"""Tests for the context_drift evaluator using FakeJudge."""

from __future__ import annotations

from tracecheck.evaluators.context_drift import ContextDriftEvaluator
from tracecheck.judges.fake import FakeJudge
from tracecheck.schema import Step, StepType, Trace


def _trace_with_input(user_input: str = "test query") -> Trace:
    return Trace(
        trace_id="t",
        agent_name="a",
        user_input=user_input,
        steps=[
            Step(type=StepType.LLM_CALL, output="thinking"),
            Step(type=StepType.TOOL_CALL, name="search", input={"q": "x"}, output={"hits": 0}),
        ],
    )


def test_passes_when_judge_returns_on_topic() -> None:
    judge = FakeJudge(response='{"on_topic": true, "score": 1.0, "reason": "stayed on topic"}')
    result = ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input())
    assert result.passed
    assert result.score == 1.0
    assert result.details == "stayed on topic"


def test_fails_when_judge_returns_drifted() -> None:
    judge = FakeJudge(
        response='{"on_topic": false, "score": 0.2, "reason": "agent searched for unrelated topic"}'
    )
    result = ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input())
    assert not result.passed
    assert result.score == 0.2
    assert "unrelated" in result.details


def test_handles_judge_response_wrapped_in_code_fence() -> None:
    judge = FakeJudge(
        response='```json\n{"on_topic": true, "score": 0.9, "reason": "fine"}\n```'
    )
    result = ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input())
    assert result.passed
    assert result.score == 0.9


def test_handles_judge_response_with_extra_prose() -> None:
    judge = FakeJudge(
        response='Here is my verdict: {"on_topic": false, "score": 0.4, "reason": "tangent"}'
    )
    result = ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input())
    assert not result.passed
    assert result.score == 0.4


def test_unparseable_judge_response_marks_parse_error() -> None:
    judge = FakeJudge(response="I cannot evaluate this.")
    result = ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input())
    assert not result.passed
    assert result.score == 0.0
    assert result.metadata.get("parse_error") is True


def test_user_prompt_includes_user_input_and_steps() -> None:
    judge = FakeJudge(response='{"on_topic": true, "score": 1.0, "reason": "ok"}')
    ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input("refund please"))
    system, user = judge.calls[0]
    assert "refund please" in user
    assert "tool_call" in user
    assert "search" in user


def test_metadata_records_judge_name_and_raw() -> None:
    raw = '{"on_topic": true, "score": 1.0, "reason": "ok"}'
    judge = FakeJudge(response=raw)
    result = ContextDriftEvaluator(judge=judge).evaluate(_trace_with_input())
    assert result.metadata["judge"] == "fake"
    assert result.metadata["raw"] == raw


def test_long_step_fields_truncated_in_prompt() -> None:
    judge = FakeJudge(response='{"on_topic": true, "score": 1.0, "reason": "ok"}')
    huge = "x" * 5000
    trace = Trace(
        trace_id="t",
        agent_name="a",
        user_input="q",
        steps=[Step(type=StepType.TOOL_CALL, name="f", input={"big": huge}, output={"big": huge})],
    )
    ContextDriftEvaluator(judge=judge).evaluate(trace)
    _, user = judge.calls[0]
    assert "..." in user
    assert len(user) < 5000
