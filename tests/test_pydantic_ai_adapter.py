"""Tests for the Pydantic AI adapter.

The adapter duck-types its input — it never imports pydantic_ai. So
these tests use plain dataclasses that mimic the shape of Pydantic AI
messages and parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from tracecheck.adapters.pydantic_ai import pydantic_ai_to_trace
from tracecheck.schema import StepType


@dataclass
class _Part:
    part_kind: str
    content: Any = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    args: Any = None
    timestamp: Any = None


@dataclass
class _Message:
    parts: list[_Part]
    kind: str = "response"
    timestamp: Any = None


@dataclass
class _AgentRunResult:
    _messages: list[_Message] = field(default_factory=list)

    def all_messages(self) -> list[_Message]:
        return list(self._messages)


def _user_msg(text: str) -> _Message:
    return _Message(
        parts=[_Part(part_kind="user-prompt", content=text)],
        kind="request",
    )


def _llm_text_msg(text: str, ts: datetime | None = None) -> _Message:
    return _Message(
        parts=[_Part(part_kind="text", content=text)],
        kind="response",
        timestamp=ts,
    )


def _tool_call_msg(tool: str, args: Any, call_id: str) -> _Message:
    return _Message(
        parts=[
            _Part(
                part_kind="tool-call",
                tool_name=tool,
                args=args,
                tool_call_id=call_id,
            )
        ],
        kind="response",
    )


def _tool_return_msg(tool: str, content: Any, call_id: str) -> _Message:
    return _Message(
        parts=[
            _Part(
                part_kind="tool-return",
                tool_name=tool,
                content=content,
                tool_call_id=call_id,
            )
        ],
        kind="request",
    )


def test_single_llm_response() -> None:
    trace = pydantic_ai_to_trace([_llm_text_msg("Hello there.")])
    assert len(trace.steps) == 1
    assert trace.steps[0].type == StepType.LLM_CALL
    assert trace.steps[0].output == "Hello there."


def test_user_prompt_extracted_as_user_input() -> None:
    trace = pydantic_ai_to_trace(
        [_user_msg("Refund my order"), _llm_text_msg("Sure.")]
    )
    assert trace.user_input == "Refund my order"


def test_multimodal_user_prompt_takes_first_string() -> None:
    msg = _Message(
        parts=[_Part(part_kind="user-prompt", content=["What is this?", b"\x89PNG..."])],
        kind="request",
    )
    trace = pydantic_ai_to_trace([msg])
    assert trace.user_input == "What is this?"


def test_tool_call_and_return_merge_into_one_step() -> None:
    messages = [
        _user_msg("Refund please"),
        _tool_call_msg("get_order", {"user_id": "u_1"}, "call_xyz"),
        _tool_return_msg("get_order", {"order_id": "o_99"}, "call_xyz"),
        _llm_text_msg("Done."),
    ]
    trace = pydantic_ai_to_trace(messages)
    tool_steps = [s for s in trace.steps if s.type == StepType.TOOL_CALL]
    assert len(tool_steps) == 1
    assert tool_steps[0].name == "get_order"
    assert tool_steps[0].input == {"user_id": "u_1"}
    assert tool_steps[0].output == {"order_id": "o_99"}


def test_unmatched_tool_call_keeps_input_only() -> None:
    """A ToolCallPart without a matching ToolReturnPart still produces a step."""
    messages = [
        _tool_call_msg("orphan", {"x": 1}, "call_a"),
        _llm_text_msg("Done."),
    ]
    trace = pydantic_ai_to_trace(messages)
    tool_steps = [s for s in trace.steps if s.type == StepType.TOOL_CALL]
    assert len(tool_steps) == 1
    assert tool_steps[0].input == {"x": 1}
    assert tool_steps[0].output is None


def test_three_tool_call_flow() -> None:
    messages = [
        _user_msg("Refund please"),
        _tool_call_msg("get_order", {"user_id": "u_1"}, "c1"),
        _tool_return_msg("get_order", {"order_id": "o_99"}, "c1"),
        _tool_call_msg("verify_item_mismatch", {"order_id": "o_99"}, "c2"),
        _tool_return_msg("verify_item_mismatch", {"mismatch": True}, "c2"),
        _tool_call_msg("issue_refund", {"order_id": "o_99"}, "c3"),
        _tool_return_msg("issue_refund", {"status": "ok"}, "c3"),
        _llm_text_msg("Refunded $49.99 to your card."),
    ]
    trace = pydantic_ai_to_trace(messages)
    tool_names = [s.name for s in trace.steps if s.type == StepType.TOOL_CALL]
    assert tool_names == ["get_order", "verify_item_mismatch", "issue_refund"]


def test_retry_prompt_creates_retry_step() -> None:
    retry_msg = _Message(
        parts=[
            _Part(
                part_kind="retry-prompt",
                content="Argument was invalid; try again.",
                tool_name="get_order",
            )
        ],
        kind="request",
    )
    trace = pydantic_ai_to_trace([_tool_call_msg("get_order", {}, "c1"), retry_msg])
    retry_steps = [s for s in trace.steps if s.type == StepType.RETRY]
    assert len(retry_steps) == 1
    assert retry_steps[0].name == "get_order"


def test_resolves_agent_run_result_via_all_messages() -> None:
    result = _AgentRunResult(_messages=[_llm_text_msg("Hi")])
    trace = pydantic_ai_to_trace(result)
    assert len(trace.steps) == 1


def test_rejects_invalid_input_type() -> None:
    with pytest.raises(TypeError):
        pydantic_ai_to_trace("not a messages object")


def test_passes_through_expected_tools_metadata_and_trace_id() -> None:
    trace = pydantic_ai_to_trace(
        [_llm_text_msg("hi")],
        trace_id="custom_id",
        agent_name="my_agent",
        expected_tools=["get_order", "issue_refund"],
        metadata={"rubric": "must include amount"},
    )
    assert trace.trace_id == "custom_id"
    assert trace.agent_name == "my_agent"
    assert trace.expected_tools == ["get_order", "issue_refund"]
    assert trace.metadata["rubric"] == "must include amount"


def test_default_trace_id_is_uuid() -> None:
    trace = pydantic_ai_to_trace([_llm_text_msg("hi")])
    assert len(trace.trace_id) >= 32  # uuid4 produces 36-char string with hyphens


def test_message_timestamp_propagates_to_step() -> None:
    ts = datetime.now(timezone.utc)
    trace = pydantic_ai_to_trace([_llm_text_msg("hi", ts=ts)])
    assert trace.steps[0].timestamp == ts


def test_part_kind_fallback_to_class_name() -> None:
    """Parts without an explicit part_kind are classified by class name."""

    @dataclass
    class TextPart:
        content: str

    @dataclass
    class ToolCallPart:
        tool_name: str
        args: Any
        tool_call_id: str

    msg = _Message(
        parts=[
            TextPart(content="hello"),
            ToolCallPart(tool_name="search", args={"q": "x"}, tool_call_id="c1"),
        ]
    )
    trace = pydantic_ai_to_trace([msg])
    types = [s.type for s in trace.steps]
    assert types == [StepType.LLM_CALL, StepType.TOOL_CALL]


def test_falls_back_to_new_messages_if_all_messages_missing() -> None:
    class _ResultWithNewMessagesOnly:
        def new_messages(self) -> list[_Message]:
            return [_llm_text_msg("Hi from new_messages")]

    trace = pydantic_ai_to_trace(_ResultWithNewMessagesOnly())
    assert len(trace.steps) == 1
    assert trace.steps[0].output == "Hi from new_messages"


def test_multiple_tool_calls_in_one_response_message() -> None:
    """Pydantic AI can put multiple ToolCallParts in a single ModelResponse."""
    msg = _Message(
        parts=[
            _Part(
                part_kind="tool-call",
                tool_name="search_a",
                args={"q": "x"},
                tool_call_id="ca",
            ),
            _Part(
                part_kind="tool-call",
                tool_name="search_b",
                args={"q": "y"},
                tool_call_id="cb",
            ),
        ],
        kind="response",
    )
    return_a = _tool_return_msg("search_a", {"hits": 1}, "ca")
    return_b = _tool_return_msg("search_b", {"hits": 2}, "cb")
    trace = pydantic_ai_to_trace([msg, return_a, return_b])
    names = [s.name for s in trace.steps if s.type == StepType.TOOL_CALL]
    assert names == ["search_a", "search_b"]
    outputs = [s.output for s in trace.steps if s.type == StepType.TOOL_CALL]
    assert outputs == [{"hits": 1}, {"hits": 2}]
