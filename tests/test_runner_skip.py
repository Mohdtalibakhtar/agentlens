"""Runner-level tests for skip-aware evaluator scheduling."""

from __future__ import annotations

from agentlens.runner import _evaluate_trace
from agentlens.evaluators.base import Evaluator, EvaluatorResult
from agentlens.schema import Step, StepType, Trace


class _AlwaysPass(Evaluator):
    name = "always_pass"

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        return EvaluatorResult(
            evaluator=self.name, passed=True, score=1.0, details="ok"
        )


class _AlwaysFail(Evaluator):
    name = "always_fail"

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        return EvaluatorResult(
            evaluator=self.name, passed=False, score=0.0, details="nope"
        )


class _Deferred(Evaluator):
    name = "deferred"
    skip_if_others_failed = True

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        self.calls += 1
        return EvaluatorResult(
            evaluator=self.name, passed=True, score=1.0, details="ran"
        )


def _trace() -> Trace:
    return Trace(
        trace_id="t",
        agent_name="a",
        steps=[Step(type=StepType.LLM_CALL, output="x")],
    )


def test_deferred_runs_when_others_pass() -> None:
    deferred = _Deferred()
    results = _evaluate_trace(_trace(), [_AlwaysPass(), deferred])
    assert deferred.calls == 1
    assert results[1].details == "ran"
    assert results[1].metadata.get("skipped") is not True


def test_deferred_skipped_when_any_other_fails() -> None:
    deferred = _Deferred()
    results = _evaluate_trace(_trace(), [_AlwaysPass(), _AlwaysFail(), deferred])
    assert deferred.calls == 0
    assert results[2].metadata["skipped"] is True
    assert results[2].passed is True
    assert "Skipped" in results[2].details


def test_deferred_position_preserved_in_output() -> None:
    """Even when listed first, the deferred result lands at index 0."""
    deferred = _Deferred()
    results = _evaluate_trace(_trace(), [deferred, _AlwaysFail()])
    assert deferred.calls == 0
    assert results[0].metadata["skipped"] is True
    assert results[1].evaluator == "always_fail"


def test_skipped_does_not_count_as_failure_in_aggregate() -> None:
    """passed=True for skipped results so they do not double-count failure."""
    deferred = _Deferred()
    results = _evaluate_trace(_trace(), [_AlwaysFail(), deferred])
    # Aggregate is computed by run_evals as all(r.passed); skipped result
    # has passed=True so the failure is attributable only to AlwaysFail.
    aggregate = all(r.passed for r in results)
    assert aggregate is False  # because AlwaysFail
    # but if we strip the explicit failure we should see a clean pass:
    aggregate_minus_fail = all(r.passed for r in results if r.evaluator != "always_fail")
    assert aggregate_minus_fail is True
