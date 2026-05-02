"""Typer based CLI: ``agentlens run --traces ... --config ...``."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from agentlens.ingest.json import load_traces
from agentlens.report import to_json, to_text
from agentlens.runner import run_evals

app = typer.Typer(help="Evaluate multi step AI agent traces.")
logger = logging.getLogger(__name__)


@app.callback()
def _root() -> None:
    """Top-level callback so that ``run`` stays a named subcommand."""


@app.command()
def run(
    traces: Path = typer.Option(..., "--traces", help="Path to traces .jsonl or .json"),
    config: Path = typer.Option(..., "--config", help="Path to evaluator config YAML"),
    output: str = typer.Option("text", "--output", help="Output format: text or json"),
) -> None:
    """Run evaluators against a trace file. Exits non-zero if any trace fails."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    loaded = load_traces(traces)
    reports = run_evals(loaded, config)

    rendered = to_json(reports) if output == "json" else to_text(reports)
    typer.echo(rendered)

    if not all(r.passed for r in reports):
        raise typer.Exit(code=1)
