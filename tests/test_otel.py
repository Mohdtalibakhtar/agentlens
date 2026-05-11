"""Tests for OpenTelemetry OTLP/JSON ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tracecheck.ingest.json import load_traces
from tracecheck.ingest.otel import (
    _attrs_to_dict,
    _classify_span,
    _extract_tokens,
    load_otel_traces,
    parse_otel_data,
)
from tracecheck.schema import StepType


def _attr(key: str, value: object) -> dict:
    """Build an OTLP attribute entry from a Python value."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, str):
        return {"key": key, "value": {"stringValue": value}}
    raise TypeError(type(value))


def _span(
    trace_id: str,
    span_id: str,
    name: str,
    *,
    start_ns: int = 1_700_000_000_000_000_000,
    end_ns: int | None = None,
    attrs: dict | None = None,
    status_code: str | None = None,
    status_message: str | None = None,
) -> dict:
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns if end_ns is not None else start_ns + 50_000_000),
        "attributes": [_attr(k, v) for k, v in (attrs or {}).items()],
        "status": (
            {"code": status_code, "message": status_message}
            if status_code or status_message
            else {}
        ),
    }


def _document(spans: list[dict], service_name: str = "support_agent") -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [_attr("service.name", service_name)]
                },
                "scopeSpans": [{"spans": spans}],
            }
        ]
    }


def test_parses_simple_two_span_trace() -> None:
    doc = _document(
        [
            _span(
                "trace_001",
                "span_a",
                "chat.completions",
                attrs={
                    "gen_ai.system": "anthropic",
                    "gen_ai.request.model": "claude-sonnet-4-5",
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 25,
                },
            ),
            _span(
                "trace_001",
                "span_b",
                "execute_tool",
                start_ns=1_700_000_000_500_000_000,
                attrs={"gen_ai.tool.name": "get_order"},
            ),
        ]
    )
    traces = parse_otel_data(doc)
    assert len(traces) == 1
    trace = traces[0]
    assert trace.trace_id == "trace_001"
    assert trace.agent_name == "support_agent"
    assert [s.type for s in trace.steps] == [StepType.LLM_CALL, StepType.TOOL_CALL]
    assert trace.steps[1].name == "get_order"


def test_groups_spans_by_trace_id() -> None:
    doc = _document(
        [
            _span("trace_a", "s1", "chat", attrs={"gen_ai.system": "openai"}),
            _span("trace_b", "s2", "chat", attrs={"gen_ai.system": "openai"}),
            _span("trace_a", "s3", "tool_call", attrs={"gen_ai.tool.name": "search"}),
        ]
    )
    traces = parse_otel_data(doc)
    ids = sorted(t.trace_id for t in traces)
    assert ids == ["trace_a", "trace_b"]
    by_id = {t.trace_id: t for t in traces}
    assert len(by_id["trace_a"].steps) == 2
    assert len(by_id["trace_b"].steps) == 1


def test_orders_spans_within_trace_by_start_time() -> None:
    doc = _document(
        [
            _span(
                "t",
                "later",
                "chat",
                start_ns=2_000_000_000_000_000_000,
                attrs={"gen_ai.system": "openai"},
            ),
            _span(
                "t",
                "earlier",
                "tool_call",
                start_ns=1_000_000_000_000_000_000,
                attrs={"gen_ai.tool.name": "search"},
            ),
        ]
    )
    trace = parse_otel_data(doc)[0]
    assert trace.steps[0].type == StepType.TOOL_CALL
    assert trace.steps[1].type == StepType.LLM_CALL


def test_extracts_tokens_new_convention() -> None:
    tokens = _extract_tokens(
        {"gen_ai.usage.input_tokens": 200, "gen_ai.usage.output_tokens": 80}
    )
    assert tokens is not None
    assert tokens.prompt == 200
    assert tokens.completion == 80
    assert tokens.total == 280


def test_extracts_tokens_legacy_convention() -> None:
    tokens = _extract_tokens(
        {"gen_ai.usage.prompt_tokens": 150, "gen_ai.usage.completion_tokens": 40}
    )
    assert tokens is not None
    assert tokens.prompt == 150
    assert tokens.completion == 40


def test_no_tokens_returns_none() -> None:
    assert _extract_tokens({"gen_ai.system": "openai"}) is None


def test_extracts_latency_from_start_and_end() -> None:
    doc = _document(
        [
            _span(
                "t",
                "s",
                "chat",
                start_ns=1_700_000_000_000_000_000,
                end_ns=1_700_000_000_123_000_000,
                attrs={"gen_ai.system": "openai"},
            ),
        ]
    )
    step = parse_otel_data(doc)[0].steps[0]
    assert step.latency_ms == pytest.approx(123.0)


def test_error_status_creates_error_step() -> None:
    doc = _document(
        [
            _span(
                "t",
                "s",
                "tool_call",
                attrs={"gen_ai.tool.name": "broken_tool"},
                status_code="STATUS_CODE_ERROR",
                status_message="context window exceeded",
            ),
        ]
    )
    step = parse_otel_data(doc)[0].steps[0]
    assert step.error == "context window exceeded"


def test_skips_spans_without_trace_id() -> None:
    doc = {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", "x")]},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": "noop",
                                "startTimeUnixNano": "0",
                                "endTimeUnixNano": "1",
                                "attributes": [],
                            }
                        ]
                    }
                ],
            }
        ]
    }
    assert parse_otel_data(doc) == []


def test_skips_spans_we_do_not_recognize() -> None:
    doc = _document(
        [
            _span("t", "s1", "unrelated_db_query"),
            _span("t", "s2", "chat", attrs={"gen_ai.system": "openai"}),
        ]
    )
    trace = parse_otel_data(doc)[0]
    assert len(trace.steps) == 1
    assert trace.steps[0].type == StepType.LLM_CALL


def test_service_name_becomes_agent_name() -> None:
    doc = _document(
        [_span("t", "s", "chat", attrs={"gen_ai.system": "openai"})],
        service_name="rag_assistant",
    )
    assert parse_otel_data(doc)[0].agent_name == "rag_assistant"


def test_missing_resource_spans_raises() -> None:
    with pytest.raises(ValueError, match="OTLP"):
        parse_otel_data({"not_otel": True})


def test_load_otel_traces_from_file(tmp_path: Path) -> None:
    doc = _document([_span("t", "s", "chat", attrs={"gen_ai.system": "openai"})])
    file = tmp_path / "spans.otel.json"
    file.write_text(json.dumps(doc))
    traces = load_otel_traces(file)
    assert len(traces) == 1


def test_load_otel_traces_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_otel_traces(tmp_path / "missing.json")


def test_load_traces_auto_detects_otel(tmp_path: Path) -> None:
    """load_traces() should route OTLP/JSON automatically — same CLI, no flag."""
    doc = _document([_span("t", "s", "chat", attrs={"gen_ai.system": "openai"})])
    file = tmp_path / "spans.json"
    file.write_text(json.dumps(doc))
    traces = load_traces(file)
    assert len(traces) == 1
    assert traces[0].trace_id == "t"


def test_classify_tool_call_from_gen_ai_attribute() -> None:
    assert (
        _classify_span({"name": "anything"}, {"gen_ai.tool.name": "search"})
        == StepType.TOOL_CALL
    )


def test_classify_tool_call_from_span_name_heuristic() -> None:
    assert _classify_span({"name": "execute_tool_foo"}, {}) == StepType.TOOL_CALL


def test_classify_llm_call_from_gen_ai_prefix() -> None:
    assert _classify_span({"name": "x"}, {"gen_ai.system": "openai"}) == StepType.LLM_CALL


def test_classify_returns_none_for_unrelated_span() -> None:
    assert _classify_span({"name": "db.query"}, {"db.statement": "SELECT 1"}) is None


def test_attrs_to_dict_handles_mixed_value_types() -> None:
    attrs = _attrs_to_dict(
        [
            _attr("str", "hello"),
            _attr("int", 42),
            _attr("float", 3.14),
            _attr("bool", True),
        ]
    )
    assert attrs == {"str": "hello", "int": 42, "float": 3.14, "bool": True}
