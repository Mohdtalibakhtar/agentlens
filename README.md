# Agentlens

Evaluate multi step AI agent traces, not just single LLM responses.

## Why

Most eval tools score one input/output pair at a time. Production agents take 7 tool calls, retry, hit context limits, and *then* produce an answer. The trace is the unit you actually need to evaluate. agentlens ingests full agent traces and scores them on tool accuracy, context handling, step efficiency, failure modes, and final output quality.

## Quick start

```bash
pip install agentlens         # core, deterministic evaluators only
pip install agentlens[llm]    # adds the LLM-as-judge extras (anthropic SDK)

agentlens run \
  --traces examples/sample_traces.jsonl \
  --config examples/evals.yaml
```

Sample output:

```
Trace: support_001_pass (support_agent)
  [PASS] tool_accuracy:    Tool sequence matches expected (3 calls).
  [PASS] step_efficiency:  Efficient: 3 tool call(s), no loops or retries.
  [PASS] failure_modes:    No failure modes detected.
  [PASS] context_drift:    Stayed on topic.
  [PASS] output_quality:   Reply confirms refund and dollar amount.
  -> PASS

Trace: support_002_fail (support_agent)
  [FAIL] tool_accuracy:    Expected [get_order, verify_item_mismatch, issue_refund],
                           got [get_order, get_order, get_order, search_products].
  [FAIL] step_efficiency:  Inefficient: 2 consecutive duplicate tool call(s),
                           1 excess tool call(s) over expected.
  [FAIL] failure_modes:    Detected: infinite_loop, context_window_overflow.
  [PASS] context_drift:    Stayed on topic but failed mechanically.
  [SKIP] output_quality:   Skipped: other evaluators failed on this trace.
  -> FAIL

Aggregate: 1/2 traces passed
exit code: 1
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
| `step_efficiency` | Tool calls vs expected; flags consecutive duplicate calls and retries | Built |
| `failure_modes` | Tags traces with known failure shapes (loops, context overflow, tool errors) | Built |
| `context_drift` | Does the agent stay on topic across steps? (LLM as judge) | Built |
| `output_quality` | Final reply scored against a rubric, only after others pass (LLM as judge) | Built |

`output_quality` is **deferred**: the runner only invokes the judge after every other evaluator on the same trace passes. Already-broken traces do not burn judge tokens, and the report renders `[SKIP]` for them.

## LLM-as-judge configuration

The two judge-based evaluators (`context_drift`, `output_quality`) need a backend. Configure it in your YAML:

```yaml
evaluators:
  - tool_accuracy
  - step_efficiency
  - failure_modes
  - context_drift
  - output_quality

judge:
  provider: anthropic        # or "fake" for tests
  model: claude-sonnet-4-5
  max_tokens: 1024

output_quality:
  rubric: |
    The reply must accurately reflect the tool-call outcomes.
    Confirm dollar amounts when a refund was issued.
```

Rubric resolution is per-trace first, then YAML default — set `trace.metadata.rubric` to override per scenario.

The `AnthropicJudge` enables prompt caching on the system block, so judging N traces costs roughly `1 + 0.1 * (N − 1)` system-prompt tokens. See [agentlens/judges/](agentlens/judges/) for the protocol and the `FakeJudge` used in tests.

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
- [x] Step efficiency evaluator
- [x] Failure mode detection
- [x] Context drift evaluator
- [x] Output quality evaluator
- [ ] OpenTelemetry span ingest
- [ ] Pydantic AI native integration
- [ ] LangGraph trace adapter
- [ ] PyPI release

## Examples

See [examples/](examples/):
- [basic_usage.py](examples/basic_usage.py) — programmatic API
- [sample_traces.jsonl](examples/sample_traces.jsonl) — three traces (pass, fail, edge case)
- [evals.yaml](examples/evals.yaml) — minimal deterministic config (no LLM key needed)
- [evals_with_judge.yaml](examples/evals_with_judge.yaml) — full config with the LLM-as-judge evaluators enabled

## Architecture

```
traces.jsonl ──► ingest ──► Trace[] ──► runner ──► [Evaluator.evaluate(trace)] ──► report
                                            ▲
                                         evals.yaml
```

The library is intentionally small: a Pydantic schema, a loader, a registry of evaluators, and a runner that walks every trace through every configured evaluator. CLI and library share the same code path.

## License

MIT
