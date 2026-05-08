"""Typer based CLI: ``agentlens run --traces ... --config ...``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from agentlens.ingest.json import load_traces
from agentlens.report import to_json, to_text
from agentlens.report_html import to_html
from agentlens.runner import run_evals

app = typer.Typer(help="Evaluate multi step AI agent traces.")
logger = logging.getLogger(__name__)

VALID_OUTPUTS = {"text", "json", "html"}


@app.callback()
def _root() -> None:
    """Top-level callback so that ``run`` stays a named subcommand."""


@app.command()
def run(
    traces: Path = typer.Option(..., "--traces", help="Path to traces .jsonl or .json"),
    config: Path = typer.Option(..., "--config", help="Path to evaluator config YAML"),
    output: str = typer.Option("text", "--output", help="Output format: text, json, or html"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Write rendered output to this file instead of stdout (recommended for html).",
    ),
) -> None:
    """Run evaluators against a trace file. Exits non-zero if any trace fails."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if output not in VALID_OUTPUTS:
        raise typer.BadParameter(
            f"--output must be one of {sorted(VALID_OUTPUTS)}, got {output!r}"
        )

    loaded = load_traces(traces)
    reports = run_evals(loaded, config)

    if output == "html":
        rendered = to_html(reports)
    elif output == "json":
        rendered = to_json(reports)
    else:
        rendered = to_text(reports)

    if out is not None:
        out.write_text(rendered)
        typer.echo(f"Wrote {output} report to {out}")
    else:
        typer.echo(rendered)

    if not all(r.passed for r in reports):
        raise typer.Exit(code=1)
