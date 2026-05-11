"""Tests for the LangGraph / LangChain adapter.

Duck-typed: we never import langchain or langgraph. The fixtures
mimic the dict shape ``graph.astream_events()`` produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tracecheck.adapters.langgraph import langgraph_events_to_trace
from tracecheck.schema import StepType


def _chat_start(run_id: str, *, model: str = "gpt-4", messages: list | None = None) -> dict:
    return {
        "event": "on_chat_model_start",
        "name": "ChatOpenAI",
        "run_id": run_id,
        "data": {"input": {"messages": messages or []}},
        "metadata": {"ls_model_name": model},
    }


def _chat_end(run_id: str, content: str) -> dict:
    return {
        "event": "on_chat_model_end",
        "name": "ChatOpenAI",
        "run_id": run_id,
        "data": {"output": {"content": content}},
    }


def _chat_error(run_id: str, message: str) -> dict:
    return {
        "event": "on_chat_model_error",
        "name": "ChatOpenAI",
        "run_id": run_id,
        "data": {"error": message},
    }


def _tool_start(run_id: str, name: str, args: dict) -> dict:
    return {
        "event": "on_tool_start",
        "name": name,
        "run_id": run_id,
        "data": {"input": args},
    }


def _tool_end(run_id: str, output: dict) -> dict:
    return {
        "event": "on_tool_end",
        "run_id": run_id,
        "data": {"output": output},
    }


def _tool_error(run_id: str, message: str) -> dict:
    return {
        "event": "on_tool_error",
        "run_id": run_id,
        "data": {"error": message},
    }


def test_pairs_tool_start_and_end_by_run_id() -> None:
    events = [
        _tool_start("r1", "get_order", {"user_id": "u_1"}),
        _tool_end("r1", {"order_id": "o_99"}),
    ]
    trace = langgraph_events_to_trace(events)
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.type == StepType.TOOL_CALL
    assert step.name == "get_order"
    assert step.input == {"user_id": "u_1"}
    assert step.output == {"order_id": "o_99"}


def test_chat_model_start_and_end_produce_llm_step() -> None:
    events = [
        _chat_start("r1", model="gpt-4o"),
        _chat_end("r1", "Hello there."),
    ]
    trace = langgraph_events_to_trace(events)
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.type == StepType.LLM_CALL
    assert step.name == "gpt-4o"
    assert step.output == "Hello there."


def test_full_three_tool_flow() -> None:
    events = [
        _chat_start("c1", messages=[{"type": "human", "content": "Refund please"}]),
        _chat_end("c1", "I will look up the order."),
        _tool_start("t1", "get_order", {"user_id": "u_1"}),
        _tool_end("t1", {"order_id": "o_99"}),
        _tool_start("t2", "verify_item_mismatch", {"order_id": "o_99"}),
        _tool_end("t2", {"mismatch": True}),
        _tool_start("t3", "issue_refund", {"order_id": "o_99"}),
        _tool_end("t3", {"status": "ok"}),
        _chat_start("c2"),
        _chat_end("c2", "Refunded $49.99 to your card."),
    ]
    trace = langgraph_events_to_trace(events)
    tool_names = [s.name for s in trace.steps if s.type == StepType.TOOL_CALL]
    assert tool_names == ["get_order", "verify_item_mismatch", "issue_refund"]
    llm_outputs = [s.output for s in trace.steps if s.type == StepType.LLM_CALL]
    assert "Refunded $49.99 to your card." in llm_outputs


def test_extracts_user_input_from_human_message_dict() -> None:
    events = [
        _chat_start(
            "c1",
            messages=[
                {"type": "system", "content": "You are helpful."},
                {"type": "human", "content": "Refund my order"},
            ],
        ),
        _chat_end("c1", "Sure."),
    ]
    trace = langgraph_events_to_trace(events)
    assert trace.user_input == "Refund my order"


def test_extracts_user_input_from_role_tuple() -> None:
    events = [
        _chat_start("c1", messages=[("system", "Be brief."), ("human", "Hi")]),
        _chat_end("c1", "Hi."),
    ]
    trace = langgraph_events_to_trace(events)
    assert trace.user_input == "Hi"


def test_extracts_user_input_from_object_with_type_attr() -> None:
    @dataclass
    class _Msg:
        type: str
        content: str

    events = [
        _chat_start("c1", messages=[_Msg("system", "Be brief."), _Msg("human", "Hello")]),
        _chat_end("c1", "Hi."),
    ]
    trace = langgraph_events_to_trace(events)
    assert trace.user_input == "Hello"


def test_explicit_user_input_overrides_extraction() -> None:
    events = [
        _chat_start("c1", messages=[{"type": "human", "content": "wrong"}]),
        _chat_end("c1", "ok"),
    ]
    trace = langgraph_events_to_trace(events, user_input="right")
    assert trace.user_input == "right"


def test_tool_error_creates_error_step() -> None:
    events = [
        _tool_start("t1", "bad_tool", {"x": 1}),
        _tool_error("t1", "connection refused"),
    ]
    trace = langgraph_events_to_trace(events)
    step = trace.steps[0]
    assert step.type == StepType.ERROR
    assert step.error == "connection refused"
    assert step.name == "bad_tool"


def test_chat_model_error_creates_error_step() -> None:
    events = [
        _chat_start("c1"),
        _chat_error("c1", "context window exceeded"),
    ]
    trace = langgraph_events_to_trace(events)
    step = trace.steps[0]
    assert step.type == StepType.ERROR
    assert step.error == "context window exceeded"


def test_unmatched_start_event_has_no_output() -> None:
    """A start event without a matching end still produces a step with input only."""
    events = [_tool_start("t1", "search", {"q": "x"})]
    trace = langgraph_events_to_trace(events)
    assert len(trace.steps) == 1
    assert trace.steps[0].input == {"q": "x"}
    assert trace.steps[0].output is None


def test_unrelated_events_are_dropped() -> None:
    """Chain, retriever, parser events should not show up in the trace."""
    events = [
        {"event": "on_chain_start", "name": "AgentExecutor", "run_id": "x1", "data": {}},
        _chat_start("c1"),
        _chat_end("c1", "hello"),
        {"event": "on_chain_end", "name": "AgentExecutor", "run_id": "x1", "data": {}},
        {"event": "on_retriever_start", "name": "Chroma", "run_id": "r1", "data": {}},
        {"event": "on_retriever_end", "name": "Chroma", "run_id": "r1", "data": {}},
    ]
    trace = langgraph_events_to_trace(events)
    assert len(trace.steps) == 1
    assert trace.steps[0].type == StepType.LLM_CALL


def test_handles_ai_message_object_for_chat_output() -> None:
    """LangChain often wraps responses in an AIMessage with .content."""

    @dataclass
    class _AIMessage:
        content: str

    events = [
        _chat_start("c1"),
        {
            "event": "on_chat_model_end",
            "run_id": "c1",
            "data": {"output": _AIMessage(content="From object.")},
        },
    ]
    trace = langgraph_events_to_trace(events)
    assert trace.steps[0].output == "From object."


def test_handles_chat_result_generations_shape() -> None:
    """Some LangChain end events return ChatResult-like dicts."""
    events = [
        _chat_start("c1"),
        {
            "event": "on_chat_model_end",
            "run_id": "c1",
            "data": {
                "output": {
                    "generations": [
                        [{"text": "from generations"}]
                    ]
                }
            },
        },
    ]
    trace = langgraph_events_to_trace(events)
    assert trace.steps[0].output == "from generations"


def test_legacy_on_llm_events_also_recognised() -> None:
    """``on_llm_start`` / ``on_llm_end`` are the older equivalents."""
    events = [
        {
            "event": "on_llm_start",
            "name": "OpenAI",
            "run_id": "c1",
            "data": {"input": {"messages": []}},
            "metadata": {"ls_model_name": "text-davinci-003"},
        },
        {
            "event": "on_llm_end",
            "run_id": "c1",
            "data": {"output": {"content": "legacy."}},
        },
    ]
    trace = langgraph_events_to_trace(events)
    assert trace.steps[0].type == StepType.LLM_CALL
    assert trace.steps[0].name == "text-davinci-003"
    assert trace.steps[0].output == "legacy."


def test_passes_through_trace_id_agent_name_expected_metadata() -> None:
    events = [_chat_start("c1"), _chat_end("c1", "ok")]
    trace = langgraph_events_to_trace(
        events,
        trace_id="custom",
        agent_name="my_lg_agent",
        expected_tools=["a", "b"],
        metadata={"rubric": "say hello"},
    )
    assert trace.trace_id == "custom"
    assert trace.agent_name == "my_lg_agent"
    assert trace.expected_tools == ["a", "b"]
    assert trace.metadata["rubric"] == "say hello"


def test_default_trace_id_is_uuid() -> None:
    events = [_chat_start("c1"), _chat_end("c1", "ok")]
    trace = langgraph_events_to_trace(events)
    assert len(trace.trace_id) >= 32


def test_empty_event_list_yields_empty_trace() -> None:
    trace = langgraph_events_to_trace([])
    assert trace.steps == []


def test_step_order_matches_event_order() -> None:
    events = [
        _tool_start("t1", "first_tool", {}),
        _tool_end("t1", {}),
        _chat_start("c1"),
        _chat_end("c1", "between"),
        _tool_start("t2", "second_tool", {}),
        _tool_end("t2", {}),
    ]
    trace = langgraph_events_to_trace(events)
    names = [s.name for s in trace.steps]
    assert names[0] == "first_tool"
    assert names[1] in ("gpt-4", "ChatOpenAI")  # llm step in the middle
    assert names[2] == "second_tool"
