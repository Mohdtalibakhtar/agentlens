"""Convert a Pydantic AI agent run into a tracecheck `Trace`.

Pydantic AI represents a single `Agent.run()` as an ordered list of
``ModelRequest`` / ``ModelResponse`` messages. Each message has
``parts`` like ``TextPart``, ``ToolCallPart``, ``ToolReturnPart``,
``UserPromptPart``, ``SystemPromptPart``, ``RetryPromptPart``.

This adapter walks those parts and produces a flat list of `Step`
objects in chronological order. A ``ToolCallPart`` and its matching
``ToolReturnPart`` are merged into a single ``tool_call`` step
(matched by ``tool_call_id``), so the trace reads like the
human-intuitive *"called X with args Y, got back Z"* — not as two
separate events.

The adapter is **duck-typed**: we never ``import pydantic_ai``. Users
pass us the ``AgentRunResult`` they already have, and we read fields
by attribute name. That keeps Pydantic AI out of the tracecheck
dependency graph for users who do not use it.
"""

from __future__ import annotations

import uuid
from typing import Any

from tracecheck.schema import Step, StepType, Trace


def pydantic_ai_to_trace(
    messages_or_result: Any,
    *,
    trace_id: str | None = None,
    agent_name: str = "pydantic_ai_agent",
    expected_tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Convert Pydantic AI message history into a tracecheck `Trace`.

    Args:
        messages_or_result: Either the object returned by
            ``Agent.run()`` (whose ``.all_messages()`` returns the
            full message history), or a raw list of messages.
        trace_id: Identifier to record on the trace. Defaults to a
            fresh UUID — pass a stable id when you want CI runs to
            line up with the same trace across replays.
        agent_name: Name to record on the trace.
        expected_tools: Optional ordered list of tool names you expect
            this run to call. Required if you want ``tool_accuracy``
            to score the trace. Author it per scenario in a golden
            test set.
        metadata: Free-form metadata. Set ``metadata={"rubric": "..."}``
            to attach a per-trace rubric for ``output_quality``.

    Returns:
        A `Trace` with one `Step` per LLM text response and per
        tool call. Tool returns are merged into their matching call.
    """
    messages = _resolve_messages(messages_or_result)
    return Trace(
        trace_id=trace_id or str(uuid.uuid4()),
        agent_name=agent_name,
        user_input=_extract_user_input(messages),
        steps=_build_steps(messages),
        expected_tools=expected_tools,
        metadata=metadata or {},
    )


def _resolve_messages(obj: Any) -> list:
    """Accept an ``AgentRunResult`` or a raw message list."""
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "all_messages") and callable(obj.all_messages):
        return list(obj.all_messages())
    if hasattr(obj, "new_messages") and callable(obj.new_messages):
        return list(obj.new_messages())
    raise TypeError(
        "pydantic_ai_to_trace expects an AgentRunResult or a list of "
        f"messages; got {type(obj).__name__}"
    )


def _extract_user_input(messages: list) -> str | None:
    """Pull the first ``UserPromptPart`` content as the user input."""
    for msg in messages:
        for part in _iter_parts(msg):
            if _part_kind(part) != "user-prompt":
                continue
            content = getattr(part, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        return item
            return None
    return None


def _build_steps(messages: list) -> list[Step]:
    """Walk messages in order, emitting one `Step` per relevant part.

    Tool calls and their matching tool returns merge into a single
    ``tool_call`` step. Unmatched tool calls keep their input but
    have no output.
    """
    steps: list[Step] = []
    pending: dict[str, Step] = {}  # tool_call_id -> open Step

    for msg in messages:
        msg_timestamp = getattr(msg, "timestamp", None)

        for part in _iter_parts(msg):
            kind = _part_kind(part)

            if kind == "text":
                steps.append(
                    Step(
                        type=StepType.LLM_CALL,
                        output=getattr(part, "content", None),
                        timestamp=msg_timestamp,
                    )
                )

            elif kind == "tool-call":
                step = Step(
                    type=StepType.TOOL_CALL,
                    name=getattr(part, "tool_name", None),
                    input=getattr(part, "args", None),
                    timestamp=msg_timestamp,
                )
                steps.append(step)
                tool_call_id = getattr(part, "tool_call_id", None)
                if tool_call_id is not None:
                    pending[tool_call_id] = step

            elif kind == "tool-return":
                tool_call_id = getattr(part, "tool_call_id", None)
                matching = pending.pop(tool_call_id, None) if tool_call_id else None
                if matching is not None:
                    matching.output = getattr(part, "content", None)

            elif kind == "retry-prompt":
                steps.append(
                    Step(
                        type=StepType.RETRY,
                        name=getattr(part, "tool_name", None),
                        input=getattr(part, "content", None),
                        timestamp=getattr(part, "timestamp", None) or msg_timestamp,
                    )
                )

    return steps


def _iter_parts(msg: Any) -> list:
    parts = getattr(msg, "parts", None)
    return list(parts) if parts is not None else []


def _part_kind(part: Any) -> str | None:
    """Read the ``part_kind`` discriminator, with a class-name fallback.

    Pydantic AI sets ``part_kind`` explicitly on each part type. The
    fallback is defensive — if a downstream framework forwards
    similarly-shaped parts without the discriminator, the class name
    is usually enough.
    """
    explicit = getattr(part, "part_kind", None)
    if isinstance(explicit, str):
        return explicit

    name = type(part).__name__.lower()
    if "toolcall" in name:
        return "tool-call"
    if "toolreturn" in name:
        return "tool-return"
    if "userprompt" in name:
        return "user-prompt"
    if "systemprompt" in name:
        return "system-prompt"
    if "retryprompt" in name:
        return "retry-prompt"
    if "textpart" in name:
        return "text"
    return None
