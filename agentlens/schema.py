"""Pydantic models that define the agent trace input format."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """The kind of action recorded in a single trace step."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRY = "retry"
    ERROR = "error"


class TokenUsage(BaseModel):
    """Token counts for a single LLM call."""

    prompt: int = 0
    completion: int = 0
    total: int = 0


class Step(BaseModel):
    """A single step within an agent trace.

    Attributes:
        type: Kind of step (llm_call, tool_call, retry, error).
        name: Tool name for tool_call, model name for llm_call.
        input: Step input (dict or string).
        output: Step output (dict or string).
        latency_ms: Wall clock latency of the step in milliseconds.
        tokens: Token usage if applicable.
        timestamp: Wall clock timestamp at start of step.
        error: Error message if type is ERROR.
    """

    type: StepType
    name: str | None = None
    input: Any | None = None
    output: Any | None = None
    latency_ms: float | None = None
    tokens: TokenUsage | None = None
    timestamp: datetime | None = None
    error: str | None = None


class Trace(BaseModel):
    """A full agent trace: one user request, the steps the agent took, and metadata.

    Attributes:
        trace_id: Stable identifier for this trace.
        agent_name: Name of the agent that produced the trace.
        user_input: The user request that triggered the agent.
        steps: Ordered list of steps the agent took.
        expected_tools: Optional ordered list of tool names the agent should
            have called. Used by the tool_accuracy evaluator.
        metadata: Free-form extra fields.
    """

    trace_id: str
    agent_name: str
    user_input: str | None = None
    steps: list[Step]
    expected_tools: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def tool_calls(self) -> list[Step]:
        """Return only the tool_call steps, in order."""
        return [s for s in self.steps if s.type == StepType.TOOL_CALL]
