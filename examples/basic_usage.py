"""Programmatic API example.

Run from the repo root:
    poetry run python examples/basic_usage.py
"""

from __future__ import annotations

from pathlib import Path

from tracecheck import load_traces, run_evals
from tracecheck.report import to_text

EXAMPLES_DIR = Path(__file__).parent


def main() -> None:
    traces = load_traces(EXAMPLES_DIR / "sample_traces.jsonl")
    reports = run_evals(traces, EXAMPLES_DIR / "evals.yaml")
    print(to_text(reports))


if __name__ == "__main__":
    main()
