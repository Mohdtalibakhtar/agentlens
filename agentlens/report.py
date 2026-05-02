"""Render TraceReport lists as JSON or human readable text."""

from __future__ import annotations

import json

from agentlens.runner import TraceReport


def to_json(reports: list[TraceReport]) -> str:
    """Serialize reports as a pretty JSON array."""
    return json.dumps([r.model_dump() for r in reports], indent=2, default=str)


def to_text(reports: list[TraceReport]) -> str:
    """Render reports as a compact human readable summary."""
    lines: list[str] = []
    for report in reports:
        lines.append(f"Trace: {report.trace_id} ({report.agent_name})")
        for r in report.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"  [{mark}] {r.evaluator}: {r.details}")
        lines.append(f"  -> {'PASS' if report.passed else 'FAIL'}")
        lines.append("")

    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    lines.append(f"Aggregate: {passed}/{total} traces passed")
    return "\n".join(lines)
