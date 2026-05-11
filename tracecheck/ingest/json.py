"""Load traces from a .jsonl or .json file into Trace objects.

The .json loader auto-detects OpenTelemetry OTLP/JSON: if the document
has a top-level ``resourceSpans`` key it is routed to the OTel parser
(:mod:`tracecheck.ingest.otel`). Same CLI command, no new flag — users
can hand either a native tracecheck JSONL or an OTLP/JSON span export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from tracecheck.schema import Trace

logger = logging.getLogger(__name__)


def load_traces(path: str | Path) -> list[Trace]:
    """Load traces from a JSON, JSONL, or OTLP/JSON file.

    Args:
        path: Path to a ``.jsonl`` (one trace per line) or ``.json``
            file. ``.json`` may be a single trace, an array of traces,
            or an OTLP/JSON span export (auto-detected).

    Returns:
        List of validated ``Trace`` instances.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is not .json or .jsonl.
        ValidationError: If a record fails Pydantic validation.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trace file not found: {p}")

    if p.suffix == ".jsonl":
        return _load_jsonl(p)
    if p.suffix == ".json":
        return _load_json(p)
    raise ValueError(f"Unsupported file extension: {p.suffix}. Use .json or .jsonl")


def _load_jsonl(p: Path) -> list[Trace]:
    traces: list[Trace] = []
    with p.open() as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                traces.append(Trace.model_validate_json(stripped))
            except Exception:
                logger.error("Failed to parse trace on line %d of %s", i, p)
                raise
    return traces


def _load_json(p: Path) -> list[Trace]:
    data = json.loads(p.read_text())
    if isinstance(data, dict) and "resourceSpans" in data:
        from tracecheck.ingest.otel import parse_otel_data

        return parse_otel_data(data)
    if isinstance(data, list):
        return [Trace.model_validate(t) for t in data]
    return [Trace.model_validate(data)]
