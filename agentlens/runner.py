"""Orchestrate evaluator runs across a list of traces."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from agentlens.evaluators.base import Evaluator, EvaluatorResult
from agentlens.evaluators.tool_accuracy import ToolAccuracyEvaluator
from agentlens.schema import Trace

logger = logging.getLogger(__name__)


EVALUATOR_REGISTRY: dict[str, type[Evaluator]] = {
    "tool_accuracy": ToolAccuracyEvaluator,
}


class TraceReport(BaseModel):
    """Aggregated results of all evaluators against a single trace."""

    trace_id: str
    agent_name: str
    results: list[EvaluatorResult]
    passed: bool


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML eval config from disk."""
    return yaml.safe_load(Path(path).read_text())


def build_evaluators(config: dict[str, Any]) -> list[Evaluator]:
    """Instantiate evaluators named in the ``evaluators`` config list."""
    names = config.get("evaluators", [])
    evaluators: list[Evaluator] = []
    for name in names:
        cls = EVALUATOR_REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown evaluator: %s — skipping", name)
            continue
        evaluators.append(cls())
    return evaluators


def run_evals(
    traces: list[Trace],
    config: dict[str, Any] | str | Path,
) -> list[TraceReport]:
    """Run configured evaluators against every trace and return reports.

    Args:
        traces: Loaded trace objects.
        config: Either a parsed config dict or a path to a YAML file.

    Returns:
        One TraceReport per input trace.
    """
    cfg = config if isinstance(config, dict) else load_config(config)
    evaluators = build_evaluators(cfg)

    reports: list[TraceReport] = []
    for trace in traces:
        results = [e.evaluate(trace) for e in evaluators]
        reports.append(
            TraceReport(
                trace_id=trace.trace_id,
                agent_name=trace.agent_name,
                results=results,
                passed=all(r.passed for r in results),
            )
        )
    return reports
