"""Single-file HTML report renderer.

Produces a self-contained HTML document — no external CSS, JS, fonts,
or images. Open in a browser, attach to a CI artifact, or screenshot.

Same data shape as ``to_text`` / ``to_json``: takes a list of
``TraceReport`` objects, renders pass / fail / skip per evaluator
per trace plus an aggregate summary.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any

from agentlens.evaluators.base import EvaluatorResult
from agentlens.runner import TraceReport


def to_html(reports: list[TraceReport], title: str = "agentlens report") -> str:
    """Render a self-contained HTML document for the given reports."""
    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    failed = total - passed
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    summary = _render_summary(total, passed, failed, when)
    cards = "\n".join(_render_trace_card(r) for r in reports)

    return _DOCUMENT.format(
        title=html.escape(title),
        summary=summary,
        cards=cards or _EMPTY_STATE,
    )


def _render_summary(total: int, passed: int, failed: int, when: str) -> str:
    pct = (100 * passed / total) if total else 0
    return f"""
    <header class="summary">
      <h1>agentlens report</h1>
      <div class="meta">{when} · {total} traces · {passed} passed · {failed} failed · {pct:.0f}%</div>
    </header>
    """


def _render_trace_card(report: TraceReport) -> str:
    overall_class = "pass" if report.passed else "fail"
    overall_label = "PASS" if report.passed else "FAIL"
    rows = "\n".join(_render_evaluator_row(r) for r in report.results)
    return f"""
    <section class="card">
      <header class="card-header">
        <h2>{html.escape(report.trace_id)}</h2>
        <span class="agent-name">{html.escape(report.agent_name)}</span>
        <span class="pill pill-{overall_class}">{overall_label}</span>
      </header>
      <table class="results">
        <tbody>
{rows}
        </tbody>
      </table>
    </section>
    """


def _render_evaluator_row(result: EvaluatorResult) -> str:
    if result.metadata.get("skipped"):
        verdict_class, verdict_label = "skip", "SKIP"
    elif result.passed:
        verdict_class, verdict_label = "pass", "PASS"
    else:
        verdict_class, verdict_label = "fail", "FAIL"

    score_text = f"{result.score:.2f}"
    metadata_block = _render_metadata(result.metadata) if result.metadata else ""

    return f"""        <tr>
          <td><span class="pill pill-{verdict_class}">{verdict_label}</span></td>
          <td class="evaluator-name">{html.escape(result.evaluator)}</td>
          <td class="score">{score_text}</td>
          <td class="details">
            {html.escape(result.details)}
            {metadata_block}
          </td>
        </tr>"""


def _render_metadata(metadata: dict[str, Any]) -> str:
    pretty = json.dumps(metadata, indent=2, default=str)
    return f"""
            <details class="metadata">
              <summary>metadata</summary>
              <pre>{html.escape(pretty)}</pre>
            </details>"""


_EMPTY_STATE = """
    <section class="card">
      <header class="card-header">
        <h2>(no traces in this run)</h2>
      </header>
    </section>
"""

_DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f6f8fa;
      color: #1f2328;
      line-height: 1.5;
    }}
    header.summary {{
      background: #24292f;
      color: #fff;
      padding: 24px 32px;
    }}
    header.summary h1 {{
      margin: 0 0 4px 0;
      font-size: 24px;
      letter-spacing: -0.01em;
    }}
    header.summary .meta {{
      font-size: 14px;
      color: #c9d1d9;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 16px;
    }}
    .card {{
      background: #fff;
      border: 1px solid #d0d7de;
      border-radius: 6px;
      margin-bottom: 16px;
      overflow: hidden;
    }}
    .card-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid #d0d7de;
      background: #f6f8fa;
    }}
    .card-header h2 {{
      margin: 0;
      font-size: 16px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
    .agent-name {{
      color: #57606a;
      font-size: 14px;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.04em;
    }}
    .card-header .pill {{ margin-left: auto; }}
    .pill-pass {{ background: #d1f7c4; color: #1a7f37; }}
    .pill-fail {{ background: #ffd7d5; color: #cf222e; }}
    .pill-skip {{ background: #ddf4ff; color: #0969da; }}
    .results {{
      width: 100%;
      border-collapse: collapse;
    }}
    .results td {{
      padding: 12px 16px;
      border-top: 1px solid #d0d7de;
      vertical-align: top;
    }}
    .results tr:first-child td {{ border-top: 0; }}
    .results td:first-child {{ width: 60px; }}
    .evaluator-name {{
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 13px;
      width: 160px;
      color: #57606a;
    }}
    .score {{
      width: 50px;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 13px;
      color: #57606a;
    }}
    .details {{ color: #1f2328; }}
    details.metadata {{
      margin-top: 8px;
      font-size: 13px;
    }}
    details.metadata summary {{
      cursor: pointer;
      color: #57606a;
      user-select: none;
    }}
    details.metadata pre {{
      background: #f6f8fa;
      padding: 8px;
      border-radius: 4px;
      overflow-x: auto;
      font-size: 12px;
      margin: 4px 0;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    }}
  </style>
</head>
<body>
  {summary}
  <main>
{cards}
  </main>
</body>
</html>
"""
