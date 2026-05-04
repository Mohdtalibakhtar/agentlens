"""Tests for the output_quality evaluator using FakeJudge."""

from __future__ import annotations

from agentlens.evaluators.output_quality import OutputQualityEvaluator
from agentlens.judges.fake import FakeJudge
from agentlens.schema import Step, StepType, Trace


def _trace(
    final_reply: str = "Refunded $49.99 to your card.",
    rubric: str | None = None,
) -> Trace:
    metadata = {"rubric": rubric} if rubric else {}
    return Trace(
        trace_id="t",
        agent_name="a",
        user_input="refund please",
        metadata=metadata,
        steps=[
            Step(type=StepType.TOOL_CALL, name="issue_refund", output={"status": "ok"}),
            Step(type=StepType.LLM_CALL, output=final_reply),
        ],
    )


def test_passes_when_judge_says_quality_meets_rubric() -> None:
    judge = FakeJudge(response='{"passes": true, "score": 0.95, "reason": "rubric met"}')
    result = OutputQualityEvaluator(
        judge=judge, default_rubric="Reply must confirm refund."
    ).evaluate(_trace())
    assert result.passed
    assert result.score == 0.95
    assert result.details == "rubric met"


def test_fails_when_judge_says_quality_misses_rubric() -> None:
    judge = FakeJudge(
        response='{"passes": false, "score": 0.3, "reason": "no dollar amount"}'
    )
    result = OutputQualityEvaluator(
        judge=judge, default_rubric="Reply must include dollar amount."
    ).evaluate(_trace(final_reply="I have processed your refund."))
    assert not result.passed
    assert result.score == 0.3


def test_per_trace_rubric_overrides_default() -> None:
    judge = FakeJudge(response='{"passes": true, "score": 1.0, "reason": "ok"}')
    trace = _trace(rubric="Per-trace rubric here.")
    OutputQualityEvaluator(
        judge=judge, default_rubric="Default rubric here."
    ).evaluate(trace)
    _, user_prompt = judge.calls[0]
    assert "Per-trace rubric here." in user_prompt
    assert "Default rubric here." not in user_prompt


def test_default_rubric_used_when_no_per_trace_rubric() -> None:
    judge = FakeJudge(response='{"passes": true, "score": 1.0, "reason": "ok"}')
    OutputQualityEvaluator(
        judge=judge, default_rubric="The default rubric."
    ).evaluate(_trace())
    _, user_prompt = judge.calls[0]
    assert "The default rubric." in user_prompt


def test_no_rubric_fails_without_calling_judge() -> None:
    judge = FakeJudge(response='{"passes": true, "score": 1.0, "reason": "ok"}')
    result = OutputQualityEvaluator(judge=judge).evaluate(_trace())
    assert not result.passed
    assert "No rubric" in result.details
    assert judge.calls == []


def test_extracts_final_llm_call_as_reply() -> None:
    judge = FakeJudge(response='{"passes": true, "score": 1.0, "reason": "ok"}')
    OutputQualityEvaluator(
        judge=judge, default_rubric="r"
    ).evaluate(_trace(final_reply="The unique final reply text."))
    _, user_prompt = judge.calls[0]
    assert "The unique final reply text." in user_prompt


def test_unparseable_judge_response_marks_parse_error() -> None:
    judge = FakeJudge(response="cannot evaluate")
    result = OutputQualityEvaluator(
        judge=judge, default_rubric="r"
    ).evaluate(_trace())
    assert not result.passed
    assert result.metadata.get("parse_error") is True


def test_metadata_records_judge_and_rubric() -> None:
    judge = FakeJudge(response='{"passes": true, "score": 1.0, "reason": "ok"}')
    result = OutputQualityEvaluator(
        judge=judge, default_rubric="my rubric"
    ).evaluate(_trace())
    assert result.metadata["judge"] == "fake"
    assert result.metadata["rubric"] == "my rubric"


def test_handles_fenced_json_response() -> None:
    judge = FakeJudge(
        response='```json\n{"passes": true, "score": 0.85, "reason": "fine"}\n```'
    )
    result = OutputQualityEvaluator(
        judge=judge, default_rubric="r"
    ).evaluate(_trace())
    assert result.passed
    assert result.score == 0.85
