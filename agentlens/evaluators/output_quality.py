"""Output quality evaluator: does the agent's final reply meet a rubric?

Last evaluator in the standard pipeline. Set ``skip_if_others_failed =
True`` so the runner only invokes the judge once the deterministic
evaluators have all passed — otherwise we would burn judge tokens on
traces that are already known broken.

Rubric resolution order:

1. ``trace.metadata["rubric"]`` if present (per-scenario rubric).
2. The ``default_rubric`` injected from the YAML
   ``output_quality.rubric`` block.
3. If neither exists the evaluator records a parse-style failure and
   does not call the judge.
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

SYSTEM_PROMPT = """You evaluate whether an AI agent's final reply meets a quality rubric.

Respond ONLY with valid JSON in this exact shape:
{"passes": <true|false>, "score": <float between 0 and 1>, "reason": "<one sentence>"}

Rules:
- passes=true if the reply meets every criterion in the rubric.
- score reflects overall quality (1.0 = fully meets the rubric, 0.0 = fails entirely).
- reason cites specific rubric criteria — which were met, which were missed.
- Output JSON only. No prose, no code fences."""


class OutputQualityEvaluator(Evaluator):
    """Score the agent's final reply against a rubric using a judge."""

    name = "output_quality"
    requires_judge = True
    skip_if_others_failed = True

    def __init__(self, judge: Judge, default_rubric: str | None = None) -> None:
        self._judge = judge
        self._default_rubric = default_rubric

    @classmethod
    def from_config(
        cls,
        config_section: dict[str, Any],
        judge: Judge | None = None,
    ) -> "OutputQualityEvaluator":
        if judge is None:
            raise ValueError("output_quality requires a judge to be configured.")
        return cls(judge=judge, default_rubric=config_section.get("rubric"))

    def evaluate(self, trace: Trace) -> EvaluatorResult:
        rubric = trace.metadata.get("rubric") or self._default_rubric
        if not rubric:
            return EvaluatorResult(
                evaluator=self.name,
                passed=False,
                score=0.0,
                details="No rubric configured (need output_quality.rubric in YAML or trace.metadata.rubric).",
                metadata={"judge": self._judge.name, "rubric": None},
            )

        user_prompt = _build_user_prompt(trace, rubric)
        raw = self._judge.query(SYSTEM_PROMPT, user_prompt)
        verdict = _parse_verdict(raw)

        if verdict is None:
            return EvaluatorResult(
                evaluator=self.name,
                passed=False,
                score=0.0,
                details=f"Could not parse judge response. Raw: {raw[:200]!r}",
                metadata={
                    "judge": self._judge.name,
                    "rubric": rubric,
                    "raw": raw,
                    "parse_error": True,
                },
            )

        passed = bool(verdict.get("passes", False))
        score = float(verdict.get("score", 0.0))
        reason = str(verdict.get("reason", ""))
        return EvaluatorResult(
            evaluator=self.name,
            passed=passed,
            score=score,
            details=reason,
            metadata={"judge": self._judge.name, "rubric": rubric, "raw": raw},
        )


def _build_user_prompt(trace: Trace, rubric: str) -> str:
    """Compose a judge-friendly digest with rubric, request, trace, and final reply."""
    final = _final_output(trace) or "(no final reply found)"
    lines: list[str] = [
        "Rubric:",
        rubric.strip(),
        "",
        f"User request: {trace.user_input or '(none)'}",
        "",
        "Trace:",
    ]
    for i, step in enumerate(trace.steps, 1):
        lines.append(f"  {i}. {_summarize_step(step)}")
    lines.extend(["", f"Agent's final reply: {final}"])
    return "\n".join(lines)


def _final_output(trace: Trace) -> str:
    """Pull the most recent llm_call output as the agent's final reply."""
    for step in reversed(trace.steps):
        if step.type == StepType.LLM_CALL and step.output is not None:
            return _stringify(step.output)
    return ""


def _summarize_step(step: Step) -> str:
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


def _stringify(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _truncate(value: Any) -> str:
    text = _stringify(value)
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "..."


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """Tolerant JSON extractor: accepts raw JSON, fenced JSON, JSON in prose."""
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
