"""Smoke tests for the runner pipeline."""

from __future__ import annotations

from pathlib import Path

from tracecheck.ingest.json import load_traces
from tracecheck.runner import run_evals

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_pipeline_on_sample_traces() -> None:
    traces = load_traces(EXAMPLES / "sample_traces.jsonl")
    reports = run_evals(traces, EXAMPLES / "evals.yaml")

    assert len(reports) == 3
    by_id = {r.trace_id: r for r in reports}
    assert by_id["support_001_pass"].passed
    assert not by_id["support_002_fail"].passed
    assert not by_id["support_003_edge"].passed
