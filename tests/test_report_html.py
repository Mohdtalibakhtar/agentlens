"""Tests for the static HTML report renderer."""

from __future__ import annotations

from tracecheck.evaluators.base import EvaluatorResult
from tracecheck.report_html import to_html
from tracecheck.runner import TraceReport


def _result(name: str, passed: bool, *, skipped: bool = False, score: float = 1.0, details: str = "ok") -> EvaluatorResult:
    metadata: dict = {"skipped": True} if skipped else {}
    return EvaluatorResult(
        evaluator=name, passed=passed, score=score, details=details, metadata=metadata
    )


def _report(trace_id: str, results: list[EvaluatorResult], passed: bool | None = None) -> TraceReport:
    if passed is None:
        passed = all(r.passed for r in results)
    return TraceReport(trace_id=trace_id, agent_name="agent", results=results, passed=passed)


def test_returns_html_document() -> None:
    out = to_html([_report("t1", [_result("e", True)])])
    assert out.startswith("<!DOCTYPE html>")
    assert "</html>" in out


def test_renders_pass_pill_for_passing_trace() -> None:
    out = to_html([_report("t1", [_result("e", True)])])
    assert 'pill-pass">PASS' in out


def test_renders_fail_pill_for_failing_trace() -> None:
    out = to_html([_report("t1", [_result("e", False, details="bad")])])
    assert 'pill-fail">FAIL' in out


def test_renders_skip_pill_for_skipped_evaluator() -> None:
    out = to_html(
        [_report("t1", [_result("e", True), _result("d", True, skipped=True)])]
    )
    assert 'pill-skip">SKIP' in out


def test_summary_shows_aggregate_counts() -> None:
    reports = [
        _report("a", [_result("e", True)]),
        _report("b", [_result("e", False)]),
        _report("c", [_result("e", True)]),
    ]
    out = to_html(reports)
    assert "3 traces" in out
    assert "2 passed" in out
    assert "1 failed" in out


def test_escapes_user_content() -> None:
    """trace_id, evaluator names, and details must be HTML-escaped."""
    result = _result("eval", False, details="<script>alert('xss')</script>")
    out = to_html([_report("trace<&>id", [result])])
    assert "<script>alert" not in out
    assert "&lt;script&gt;" in out
    assert "trace&lt;&amp;&gt;id" in out


def test_self_contained_no_external_resources() -> None:
    out = to_html([_report("t1", [_result("e", True)])])
    assert "<link " not in out
    assert "@import" not in out
    assert "<script src" not in out


def test_metadata_rendered_in_collapsible_details() -> None:
    result = EvaluatorResult(
        evaluator="e",
        passed=True,
        score=1.0,
        details="ok",
        metadata={"expected": ["a", "b"], "actual": ["a", "b"]},
    )
    out = to_html([_report("t1", [result])])
    assert "<details" in out
    assert "<summary>metadata</summary>" in out
    assert "expected" in out


def test_empty_report_list_produces_valid_document() -> None:
    out = to_html([])
    assert out.startswith("<!DOCTYPE html>")
    assert "0 traces" in out


def test_includes_title_in_head() -> None:
    out = to_html([_report("t1", [_result("e", True)])], title="My Run")
    assert "<title>My Run</title>" in out
