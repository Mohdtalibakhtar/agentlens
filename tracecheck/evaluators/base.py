"""Abstract evaluator interface and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tracecheck.schema import Trace

if TYPE_CHECKING:
    from tracecheck.judges.base import Judge


class EvaluatorResult(BaseModel):
    """Outcome of running a single evaluator against a single trace.

    Attributes:
        evaluator: Name of the evaluator that produced this result.
        passed: True if the evaluator considers the trace acceptable.
        score: Continuous score in [0.0, 1.0] for grading or thresholds.
        details: Human readable explanation of the verdict.
        metadata: Free-form per-evaluator extras (e.g. expected vs actual,
            ``skipped: True`` for deferred evaluators that were not run).
    """

    evaluator: str
    passed: bool
    score: float
    details: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evaluator(ABC):
    """Base class for all evaluators.

    Class attributes:
        name: Unique identifier registered in EVALUATOR_REGISTRY.
        requires_judge: True if this evaluator needs an LLM-as-judge
            backend; the runner injects a Judge into the constructor.
        skip_if_others_failed: True if this evaluator should be deferred
            until other evaluators have run on the same trace, and
            skipped entirely if any of them failed. Used by
            output_quality so we do not burn judge tokens on already
            broken traces.
    """

    name: str = "base"
    requires_judge: bool = False
    skip_if_others_failed: bool = False

    @abstractmethod
    def evaluate(self, trace: Trace) -> EvaluatorResult:
        """Evaluate a trace and return a single result."""

    @classmethod
    def from_config(
        cls,
        config_section: dict[str, Any],
        judge: "Judge | None" = None,
    ) -> "Evaluator":
        """Build this evaluator from a per-evaluator YAML config section.

        Default implementation: pass the judge if required, else no
        arguments. Subclasses with extra config (e.g. rubric) override.
        """
        if cls.requires_judge:
            if judge is None:
                raise ValueError(
                    f"Evaluator {cls.name!r} requires a judge but none was provided."
                )
            return cls(judge=judge)
        return cls()
