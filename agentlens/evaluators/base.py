"""Abstract evaluator interface and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agentlens.schema import Trace


class EvaluatorResult(BaseModel):
    """Outcome of running a single evaluator against a single trace.

    Attributes:
        evaluator: Name of the evaluator that produced this result.
        passed: True if the evaluator considers the trace acceptable.
        score: Continuous score in [0.0, 1.0] for grading or thresholds.
        details: Human readable explanation of the verdict.
        metadata: Free-form per-evaluator extras (e.g. expected vs actual).
    """

    evaluator: str
    passed: bool
    score: float
    details: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evaluator(ABC):
    """Base class for all evaluators.

    Subclasses must set a unique ``name`` and implement ``evaluate``.
    """

    name: str = "base"

    @abstractmethod
    def evaluate(self, trace: Trace) -> EvaluatorResult:
        """Evaluate a trace and return a single result."""
