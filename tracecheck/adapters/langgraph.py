"""Convert LangChain / LangGraph callback events into a tracecheck `Trace`.

LangGraph agents emit a stream of events via ``graph.astream_events()``
(or via LangChain's classic callback interface). Each event is a dict
roughly shaped like:

    {
        "event": "on_tool_start" | "on_tool_end" | "on_chat_model_start" | ...,
        "name": "search_tool",
        "run_id": "abc-123",
        "data": {"input": ..., "output": ...},
        "metadata": {"ls_model_name": "gpt-4", ...},
    }

This adapter walks that list, pairs each *start* event with its
matching *end* (or *error*) by ``run_id``, and emits one `Step` per
LLM call and per tool call. Other event kinds (chain start/end,
parser events, retriever events, etc.) are dropped.

Duck-typed: we do not import ``langchain`` or ``langgraph``. The user
calls ``graph.astream_events(...)`` themselves, collects the list,
and hands it to us.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from tracecheck.schema import Step, StepType, Trace

# LangChain v2 uses ``on_chat_model_*``; older code emits ``on_llm_*``.
CHAT_START_EVENTS = ("on_chat_model_start", "on_llm_start")
CHAT_END_EVENTS = ("on_chat_model_end", "on_llm_end")
CHAT_ERROR_EVENTS = ("on_chat_model_error", "on_llm_error")
TOOL_START_EVENT = "on_tool_start"
TOOL_END_EVENT = "on_tool_end"
TOOL_ERROR_EVENT = "on_tool_error"


def langgraph_events_to_trace(
    events: Iterable[dict],
    *,
    trace_id: str | None = None,
    agent_name: str = "langgraph_agent",
    user_input: str | None = None,
    expected_tools: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Convert a list of LangChain/LangGraph events into a `Trace`.

    Args:
        events: Iterable of event dicts (from
            ``await graph.astream_events(...)`` collected via async-for).
        trace_id: Identifier to record. Defaults to a fresh UUID.
        agent_name: Recorded on the trace.
        user_input: If you already know the user's query, pass it.
            Otherwise we try to extract it from the first chat model
            call's input messages.
        expected_tools: Optional ordered list of expected tool names.
            Required if you want ``tool_accuracy`` to score the trace.
        metadata: Free-form metadata. Set ``metadata={"rubric": "..."}``
            for ``output_quality``.

    Returns:
        A `Trace` with one `Step` per tool call and per LLM response.
    """
    event_list = list(events)
    extracted_user_input = user_input or _extract_user_input(event_list)
    steps = _build_steps(event_list)

    return Trace(
        trace_id=trace_id or str(uuid.uuid4()),
        agent_name=agent_name,
        user_input=extracted_user_input,
        steps=steps,
        expected_tools=expected_tools,
        metadata=metadata or {},
    )


def _build_steps(events: list[dict]) -> list[Step]:
    """Pair start/end events by ``run_id`` and emit one step per pair."""
    end_by_run_id = _index_end_events(events)

    steps: list[Step] = []
    for evt in events:
        kind = evt.get("event", "")
        run_id = evt.get("run_id")
        end = end_by_run_id.get(run_id) if run_id else None

        if kind in CHAT_START_EVENTS:
            steps.append(_build_llm_step(evt, end))
        elif kind == TOOL_START_EVENT:
            steps.append(_build_tool_step(evt, end))

    return steps


def _index_end_events(events: list[dict]) -> dict[str, dict]:
    """Map ``run_id`` to its terminal event (success or error)."""
    end_by_run_id: dict[str, dict] = {}
    for evt in events:
        kind = evt.get("event", "")
        run_id = evt.get("run_id")
        if not run_id:
            continue
        if (
            kind in CHAT_END_EVENTS
            or kind in CHAT_ERROR_EVENTS
            or kind == TOOL_END_EVENT
            or kind == TOOL_ERROR_EVENT
        ):
            end_by_run_id[run_id] = evt
    return end_by_run_id


def _build_llm_step(start: dict, end: dict | None) -> Step:
    metadata = start.get("metadata") or {}
    name = (
        metadata.get("ls_model_name")
        or metadata.get("model")
        or start.get("name")
    )
    error_msg = _extract_error_message(end)
    output = _extract_chat_output(end) if end and not error_msg else None

    return Step(
        type=StepType.ERROR if error_msg else StepType.LLM_CALL,
        name=name,
        input=(start.get("data") or {}).get("input"),
        output=output,
        error=error_msg,
    )


def _build_tool_step(start: dict, end: dict | None) -> Step:
    error_msg = _extract_error_message(end)
    output = (end.get("data") or {}).get("output") if end and not error_msg else None

    return Step(
        type=StepType.ERROR if error_msg else StepType.TOOL_CALL,
        name=start.get("name"),
        input=(start.get("data") or {}).get("input"),
        output=output,
        error=error_msg,
    )


def _extract_error_message(end: dict | None) -> str | None:
    if not end:
        return None
    kind = end.get("event", "")
    if "error" not in kind:
        return None
    err = (end.get("data") or {}).get("error")
    return str(err) if err is not None else "LangGraph reported an error"


def _extract_chat_output(end: dict) -> Any:
    """Read text content out of a chat-model end event.

    LangChain wraps responses in several shapes (AIMessage, ChatResult,
    Generation, plain dict). Try the most common and fall back to the
    raw payload — evaluators can still match on it.
    """
    output = (end.get("data") or {}).get("output")
    if output is None:
        return None

    content = getattr(output, "content", None)
    if content is not None:
        return content

    if isinstance(output, dict):
        if "content" in output:
            return output["content"]
        gens = output.get("generations")
        if isinstance(gens, list) and gens:
            first = gens[0]
            if isinstance(first, list) and first:
                first = first[0]
            text = getattr(first, "text", None)
            if text is not None:
                return text
            if isinstance(first, dict):
                if "text" in first:
                    return first["text"]
                msg = first.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
    return output


def _extract_user_input(events: list[dict]) -> str | None:
    """Pull the first human message from the first chat call's input."""
    for evt in events:
        if evt.get("event") not in CHAT_START_EVENTS:
            continue
        messages = _input_messages(evt)
        if messages is None:
            continue
        for msg in messages:
            text = _human_message_text(msg)
            if text is not None:
                return text
        return None
    return None


def _input_messages(evt: dict) -> list | None:
    """Pull a flat list of messages out of a chat-model start event."""
    inp = (evt.get("data") or {}).get("input")
    if isinstance(inp, dict) and "messages" in inp:
        msgs = inp["messages"]
        # Sometimes it's a list of lists (one per prompt template)
        if isinstance(msgs, list) and msgs and isinstance(msgs[0], list):
            return msgs[0]
        return msgs
    if isinstance(inp, list):
        if inp and isinstance(inp[0], list):
            return inp[0]
        return inp
    return None


def _human_message_text(msg: Any) -> str | None:
    """Extract content from a 'human' role message in any of its shapes."""
    msg_type = getattr(msg, "type", None)
    if msg_type == "human":
        return getattr(msg, "content", None)

    if isinstance(msg, dict):
        if msg.get("type") in ("human", "user") or msg.get("role") in ("human", "user"):
            return msg.get("content")

    if isinstance(msg, (list, tuple)) and len(msg) == 2:
        role, content = msg
        if role in ("human", "user") and isinstance(content, str):
            return content
    return None
