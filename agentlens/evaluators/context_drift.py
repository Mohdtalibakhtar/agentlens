"""Context drift evaluator: did the agent stay on topic across all steps?

This is the first LLM-as-judge evaluator. It sends a digest of the trace
to a configurable :class:`Judge` backend and parses a structured verdict.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agentlens.evaluators.base import Evaluator, EvaluatorResult
from agentlens.judges.base import Judge
from agentlens.schema import Step, StepType, Trace

logger = logging.getLogger(__name__)

MAX_FIELD_CHARS = 200

SYSTEM_PROMPT = """You evaluate whether an AI agent stayed on topic relative to the user's request.

Respond ONLY with valid JSON in this exact shape:
{"on_topic": <true|false>, "score": <float between 0 and 1>, "reason": "<one sentence>"}

Rules:
- on_topic=true if every step in the trace serves the original user request.
- on_topic=false if the agent drifts to unrelated tasks, performs unrequested actions, or chases tangents.
- score reflects how on-topic the trace is overall (1.0 = perfectly on-topic, 0.0 = entirely off-topic).
- reason must cite the step(s) that caused drift if any. Otherwise note what kept it on-topic.
- Output JSON only. No prose, no code fences."""


class ContextDriftEvaluator(Evaluator):
    """Ask a judge model whether the trace stayed on topic."""

    name = "context_drift"
    requires_judge = True

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        user_prompt = _build_user_prompt(trace)
        raw = self._judge.query(SYSTEM_PROMPT, user_prompt)
        verdict = _parse_verdict(raw)

        if verdict is None:
            return EvaluatorResult(
                evaluator=self.name,
                passed=False,
                score=0.0,
                details=f"Could not parse judge response. Raw: {raw[:200]!r}",
                metadata={"judge": self._judge.name, "raw": raw, "parse_error": True},
            )

        passed = bool(verdict.get("on_topic", False))
        score = float(verdict.get("score", 0.0))
        reason = str(verdict.get("reason", ""))
        return EvaluatorResult(
            evaluator=self.name,
            passed=passed,
            score=score,
            details=reason,
            metadata={"judge": self._judge.name, "raw": raw},
        )


def _build_user_prompt(trace: Trace) -> str:
    """Compose a compact, judge-friendly digest of the trace."""
    lines: list[str] = []
    lines.append(f"User request: {trace.user_input or '(none)'}")
    lines.append("")
    lines.append("Agent trace:")
    for i, step in enumerate(trace.steps, 1):
        lines.append(f"  {i}. {_summarize_step(step)}")
    return "\n".join(lines)


def _summarize_step(step: Step) -> str:
    """One-line summary of a step suitable for the judge prompt."""
    label = step.type.value
    name = f" name={step.name!r}" if step.name else ""
    extras: list[str] = []
    if step.input is not None:
        extras.append(f"input={_truncate(step.input)}")
    if step.output is not None:
        extras.append(f"output={_truncate(step.output)}")
    if step.error:
        extras.append(f"error={_truncate(step.error)}")
    suffix = (" " + " ".join(extras)) if extras else ""
    return f"[{label}]{name}{suffix}"


def _truncate(value: Any) -> str:
    """Stringify and clip a step field so the judge prompt stays small."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "..."


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """Tolerantly extract the JSON verdict from the judge's response."""
    candidates = [raw]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    bare = re.search(r"\{.*\}", raw, re.DOTALL)
    if bare:
        candidates.append(bare.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None
