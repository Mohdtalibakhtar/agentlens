"""Orchestrate evaluator runs across a list of traces."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from tracecheck.evaluators.base import Evaluator, EvaluatorResult
from tracecheck.evaluators.context_drift import ContextDriftEvaluator
from tracecheck.evaluators.failure_modes import FailureModesEvaluator
from tracecheck.evaluators.output_quality import OutputQualityEvaluator
from tracecheck.evaluators.step_efficiency import StepEfficiencyEvaluator
from tracecheck.evaluators.tool_accuracy import ToolAccuracyEvaluator
from tracecheck.judges.base import Judge
from tracecheck.schema import Trace

logger = logging.getLogger(__name__)


EVALUATOR_REGISTRY: dict[str, type[Evaluator]] = {
    "tool_accuracy": ToolAccuracyEvaluator,
    "step_efficiency": StepEfficiencyEvaluator,
    "failure_modes": FailureModesEvaluator,
    "context_drift": ContextDriftEvaluator,
    "output_quality": OutputQualityEvaluator,
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


def build_evaluators(
    config: dict[str, Any],
    judge: Judge | None = None,
) -> list[Evaluator]:
    """Instantiate evaluators named in the ``evaluators`` config list.

    Args:
        config: Parsed YAML config.
        judge: Pre-built judge instance; required if any selected evaluator
            has ``requires_judge = True``. If omitted and the config
            contains a ``judge:`` section, a judge is built from there.
    """
    names = config.get("evaluators", [])
    if judge is None and "judge" in config:
        judge = build_judge(config["judge"])

    evaluators: list[Evaluator] = []
    for name in names:
        cls = EVALUATOR_REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown evaluator: %s — skipping", name)
            continue
        if cls.requires_judge and judge is None:
            raise ValueError(
                f"Evaluator {name!r} requires a judge but none was configured. "
                f"Add a 'judge' section to the config."
            )
        section = config.get(name, {}) or {}
        evaluators.append(cls.from_config(section, judge=judge))
    return evaluators


def build_judge(judge_config: dict[str, Any]) -> Judge:
    """Construct a Judge from a ``judge:`` config block.

    Recognised providers: ``anthropic``. ``fake`` is supported for
    testing and yields a default-positive verdict.
    """
    provider = judge_config.get("provider", "anthropic")
    if provider == "anthropic":
        from tracecheck.judges.anthropic import AnthropicJudge

        kwargs: dict[str, Any] = {}
        if "model" in judge_config:
            kwargs["model"] = judge_config["model"]
        if "max_tokens" in judge_config:
            kwargs["max_tokens"] = judge_config["max_tokens"]
        return AnthropicJudge(**kwargs)
    if provider == "fake":
        from tracecheck.judges.fake import FakeJudge

        if "response" in judge_config:
            return FakeJudge(response=judge_config["response"])
        return FakeJudge()
    raise ValueError(f"Unknown judge provider: {provider!r}")


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
        results = _evaluate_trace(trace, evaluators)
        reports.append(
            TraceReport(
                trace_id=trace.trace_id,
                agent_name=trace.agent_name,
                results=results,
                passed=all(r.passed for r in results),
            )
        )
    return reports


def _evaluate_trace(trace: Trace, evaluators: list[Evaluator]) -> list[EvaluatorResult]:
    """Run evaluators against a single trace, deferring skip-aware ones.

    Two-pass strategy: first run every evaluator that does not defer,
    then run the deferred ones only if all the others passed. The
    output preserves the original evaluator order.
    """
    results: dict[int, EvaluatorResult] = {}
    deferred: list[int] = []

    for i, ev in enumerate(evaluators):
        if ev.skip_if_others_failed:
            deferred.append(i)
        else:
            results[i] = ev.evaluate(trace)

    others_all_passed = all(r.passed for r in results.values())

    for i in deferred:
        ev = evaluators[i]
        if not others_all_passed:
            results[i] = EvaluatorResult(
                evaluator=ev.name,
                passed=True,
                score=0.0,
                details="Skipped: other evaluators failed on this trace.",
                metadata={"skipped": True},
            )
        else:
            results[i] = ev.evaluate(trace)

    return [results[i] for i in range(len(evaluators))]
