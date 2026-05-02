# agentlens

Evaluate multi step AI agent traces, not just single LLM responses.

## Why

Most eval tools score one input/output pair at a time. Production agents take 7 tool calls, retry, hit context limits, and *then* produce an answer. The trace is the unit you actually need to evaluate. agentlens ingests full agent traces and scores them on tool accuracy, context handling, step efficiency, failure modes, and final output quality.

## Quick start

```bash
pip install agentlens  # PyPI release coming soon

agentlens run \
  --traces examples/sample_traces.jsonl \
  --config examples/evals.yaml
```

Or programmatically:

```python
from agentlens import load_traces, run_evals
from agentlens.report import to_text

traces = load_traces("traces.jsonl")
reports = run_evals(traces, "evals.yaml")
print(to_text(reports))
```

The CLI exits with code 1 if any trace fails, so you can drop it into CI.

## Evaluators

| Evaluator | What it checks | Status |
|---|---|---|
| `tool_accuracy` | Did the agent call the right tools in the right order? | Built |
| `step_efficiency` | Steps taken vs minimum needed; flags loops | Coming |
| `context_drift` | Does the agent stay on topic across steps? (LLM as judge) | Coming |
| `failure_modes` | Categorises errors (loops, context overflow, hallucinated args) | Coming |
| `output_quality` | Final reply scored against a rubric (LLM as judge) | Coming |

## Trace format

A trace is a JSON object with an ordered list of steps:

```json
{
  "trace_id": "support_001",
  "agent_name": "support_agent",
  "user_input": "I got the wrong color sweater, refund please",
  "expected_tools": ["get_order", "verify_item_mismatch", "issue_refund"],
  "steps": [
    {"type": "tool_call", "name": "get_order",
     "input": {"user_id": "u_123"}, "output": {"order_id": "o_99"}},
    {"type": "tool_call", "name": "verify_item_mismatch",
     "input": {"order_id": "o_99"}, "output": {"mismatch": true}},
    {"type": "tool_call", "name": "issue_refund",
     "input": {"order_id": "o_99"}, "output": {"status": "ok"}},
    {"type": "llm_call", "output": "Refunded $49.99 to your card."}
  ]
}
```

Step types: `llm_call`, `tool_call`, `retry`, `error`. Each step may carry `latency_ms`, `tokens`, `timestamp`, and an `error` message. See [agentlens/schema.py](agentlens/schema.py) for the full Pydantic spec.

A `.jsonl` file is one trace per line. A `.json` file may be a single trace or an array.

## Roadmap

- [x] Trace ingestion (JSON, JSONL)
- [x] Tool call accuracy evaluator
- [ ] Step efficiency evaluator
- [ ] Failure mode detection
- [ ] Context drift evaluator
- [ ] Output quality evaluator
- [ ] OpenTelemetry span ingest
- [ ] Pydantic AI native integration
- [ ] LangGraph trace adapter
- [ ] PyPI release

## Examples

See [examples/](examples/):
- [basic_usage.py](examples/basic_usage.py) — programmatic API
- [sample_traces.jsonl](examples/sample_traces.jsonl) — three traces (pass, fail, edge case)
- [evals.yaml](examples/evals.yaml) — minimal config

## Architecture

```
traces.jsonl ──► ingest ──► Trace[] ──► runner ──► [Evaluator.evaluate(trace)] ──► report
                                            ▲
                                         evals.yaml
```

The library is intentionally small: a Pydantic schema, a loader, a registry of evaluators, and a runner that walks every trace through every configured evaluator. CLI and library share the same code path.

## License

MIT
