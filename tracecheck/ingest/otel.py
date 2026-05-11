"""Load traces from an OpenTelemetry OTLP/JSON span export.

OpenTelemetry is the standard tracing format most production AI agents
already emit (via LangChain callbacks, OpenAI/Anthropic instrumentation,
Pydantic Logfire, etc.). Most exporters dump OTLP/JSON. This module
parses that file format and groups spans into tracecheck `Trace`
objects, one per `traceId`.

The mapping follows the **OpenTelemetry GenAI semantic conventions**
(https://opentelemetry.io/docs/specs/semconv/gen-ai/) where applicable:

- `gen_ai.tool.name` ........... `Step.name` on a `TOOL_CALL` step
- `gen_ai.request.model` ....... `Step.name` on an `LLM_CALL` step
- `gen_ai.usage.input_tokens`
  or `gen_ai.usage.prompt_tokens` ........ `Step.tokens.prompt`
- `gen_ai.usage.output_tokens`
  or `gen_ai.usage.completion_tokens` .... `Step.tokens.completion`
- `span.startTimeUnixNano` ..... `Step.timestamp`
- `endTime - startTime` ........ `Step.latency_ms`
- `span.status.code == ERROR` .. `Step` flagged as ERROR (or `error` field set)

OTel-ingested traces have no `expected_tools` — the user authors a
golden test set separately and merges it. The LLM-as-judge evaluators
(`context_drift`, `output_quality`) work without `expected_tools` and
are the natural pairing for OTel-only pipelines.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracecheck.schema import Step, StepType, TokenUsage, Trace

logger = logging.getLogger(__name__)


def load_otel_traces(path: str | Path) -> list[Trace]:
    """Parse an OTLP/JSON file into tracecheck `Trace` objects.

    Args:
        path: Path to an OTLP/JSON span export.

    Returns:
        One `Trace` per unique `traceId` in the file. Spans are
        ordered by start time within each trace.

    Raises:
        FileNotFoundError: Path does not exist.
        ValueError: File is not valid OTLP/JSON.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OTel file not found: {p}")

    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p}: {exc}") from exc

    return parse_otel_data(data)


def parse_otel_data(data: dict) -> list[Trace]:
    """Parse an in-memory OTLP/JSON document into `Trace` objects.

    Exposed so other ingestion paths (auto-detect in
    :func:`tracecheck.ingest.json.load_traces`, future Logfire adapter,
    etc.) can call it directly without going through the filesystem.
    """
    if not isinstance(data, dict) or "resourceSpans" not in data:
        raise ValueError("Document is not OTLP/JSON (missing 'resourceSpans').")

    spans_by_trace: dict[str, list[tuple[dict, dict]]] = {}
    service_name_by_trace: dict[str, str] = {}

    for resource_span in data.get("resourceSpans", []):
        resource_attrs = _attrs_to_dict(
            resource_span.get("resource", {}).get("attributes", [])
        )
        service_name = resource_attrs.get("service.name", "unknown_agent")

        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                trace_id = span.get("traceId")
                if not trace_id:
                    continue
                attrs = _attrs_to_dict(span.get("attributes", []))
                spans_by_trace.setdefault(trace_id, []).append((span, attrs))
                service_name_by_trace.setdefault(trace_id, service_name)

    traces: list[Trace] = []
    for trace_id, span_pairs in spans_by_trace.items():
        span_pairs.sort(key=lambda pair: _start_ns(pair[0]))
        steps = [
            step
            for span, attrs in span_pairs
            if (step := _span_to_step(span, attrs)) is not None
        ]
        if not steps:
            continue
        traces.append(
            Trace(
                trace_id=trace_id,
                agent_name=service_name_by_trace[trace_id],
                steps=steps,
            )
        )
    return traces


def _span_to_step(span: dict, attrs: dict) -> Step | None:
    """Convert a single OTel span into a `Step`, or None if irrelevant."""
    step_type = _classify_span(span, attrs)
    if step_type is None:
        return None

    return Step(
        type=step_type,
        name=_extract_name(span, attrs, step_type),
        input=_first(attrs, "gen_ai.prompt", "input", "function.arguments"),
        output=_first(attrs, "gen_ai.completion", "output", "function.result"),
        tokens=_extract_tokens(attrs),
        timestamp=_ns_to_datetime(span.get("startTimeUnixNano")),
        latency_ms=_calc_latency_ms(span),
        error=_extract_error(span),
    )


def _classify_span(span: dict, attrs: dict) -> StepType | None:
    """Decide which `StepType` this span should map to."""
    if "gen_ai.tool.name" in attrs:
        return StepType.TOOL_CALL

    name = (span.get("name") or "").lower()
    if "tool" in name and ("call" in name or "execute" in name or "invoke" in name):
        return StepType.TOOL_CALL

    if any(k.startswith("gen_ai.") for k in attrs):
        return StepType.LLM_CALL

    if _is_error_span(span):
        return StepType.ERROR

    return None


def _extract_name(span: dict, attrs: dict, step_type: StepType) -> str | None:
    if step_type == StepType.TOOL_CALL:
        return attrs.get("gen_ai.tool.name") or span.get("name")
    if step_type == StepType.LLM_CALL:
        return (
            attrs.get("gen_ai.request.model")
            or attrs.get("gen_ai.system")
            or span.get("name")
        )
    return span.get("name")


def _extract_tokens(attrs: dict) -> TokenUsage | None:
    """Read token usage from either old or new GenAI convention."""
    prompt = attrs.get("gen_ai.usage.input_tokens")
    if prompt is None:
        prompt = attrs.get("gen_ai.usage.prompt_tokens")
    completion = attrs.get("gen_ai.usage.output_tokens")
    if completion is None:
        completion = attrs.get("gen_ai.usage.completion_tokens")

    if prompt is None and completion is None:
        return None

    p = int(prompt) if prompt is not None else 0
    c = int(completion) if completion is not None else 0
    return TokenUsage(prompt=p, completion=c, total=p + c)


def _extract_error(span: dict) -> str | None:
    if not _is_error_span(span):
        return None
    status = span.get("status", {})
    return status.get("message") or "OTel span status ERROR"


def _is_error_span(span: dict) -> bool:
    return span.get("status", {}).get("code") in ("STATUS_CODE_ERROR", "ERROR", 2)


def _calc_latency_ms(span: dict) -> float | None:
    start = _to_int(span.get("startTimeUnixNano"))
    end = _to_int(span.get("endTimeUnixNano"))
    if start is None or end is None:
        return None
    return (end - start) / 1e6


def _ns_to_datetime(ns: Any) -> datetime | None:
    value = _to_int(ns)
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1e9, tz=timezone.utc)


def _start_ns(span: dict) -> int:
    return _to_int(span.get("startTimeUnixNano")) or 0


def _to_int(value: Any) -> int | None:
    """OTLP/JSON encodes 64-bit ints as strings to survive JSON."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _attrs_to_dict(attr_list: list) -> dict[str, Any]:
    """Flatten OTLP attribute list to a `{key: python_value}` dict."""
    out: dict[str, Any] = {}
    for attr in attr_list:
        key = attr.get("key")
        value = attr.get("value")
        if key is None or value is None:
            continue
        out[key] = _extract_attr_value(value)
    return out


def _extract_attr_value(value: dict) -> Any:
    """Read the typed leaf out of an OTLP attribute value envelope."""
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        return [_extract_attr_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            v.get("key"): _extract_attr_value(v.get("value", {}))
            for v in value["kvlistValue"].get("values", [])
            if "key" in v
        }
    return None


def _first(attrs: dict, *keys: str) -> Any | None:
    """Return the first non-None value found among the candidate keys."""
    for key in keys:
        if key in attrs and attrs[key] is not None:
            return attrs[key]
    return None
