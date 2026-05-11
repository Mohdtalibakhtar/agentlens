# Tracecheck

Evaluate multi step AI agent traces, not just single LLM responses.

## Why

Most eval tools score one input/output pair at a time. Production agents take 7 tool calls, retry, hit context limits, and *then* produce an answer. The trace is the unit you actually need to evaluate. tracecheck ingests full agent traces and scores them on tool accuracy, context handling, step efficiency, failure modes, and final output quality.

## Quick start

```bash
pip install tracecheck         # core, deterministic evaluators only
pip install tracecheck[llm]    # adds the LLM-as-judge extras (anthropic SDK)

tracecheck run \
  --traces examples/sample_traces.jsonl \
  --config examples/evals.yaml

# write a self-contained HTML report you can open in a browser
tracecheck run \
  --traces examples/sample_traces.jsonl \
  --config examples/evals.yaml \
  --output html --out report.html
```

Sample text output:

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
from tracecheck import load_traces, run_evals
from tracecheck.report import to_text

traces = load_traces("traces.jsonl")
reports = run_evals(traces, "evals.yaml")
print(to_text(reports))
```

The CLI exits with code 1 if any trace fails, so you can drop it into CI.

## Integrating tracecheck into your agent

The trace data has to come from *your* agent. tracecheck reads traces; it does not produce them. Three steps to wire it in.

### 1. Log each step from inside your agent

Around each tool call and LLM call, append a step record. About ten lines of code total.

```python
import json, uuid
from datetime import datetime, timezone
from pathlib import Path

steps = []

def log(step_type, **kwargs):
    steps.append({
        "type": step_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    })

# inside your handler:
hits = semantic_search(query)
log("tool_call", name="semantic_search", input={"q": query}, output={"n_hits": len(hits)})

reply = llm.generate(query, hits)
log("llm_call", output=reply)

# at end of each request, append one trace line:
Path("traces.jsonl").open("a").write(json.dumps({
    "trace_id": str(uuid.uuid4()),
    "agent_name": "my_agent",
    "user_input": query,
    "steps": steps,
}) + "\n")
```

If you already use a framework with built-in tracing (LangChain callbacks, Pydantic AI events, OpenTelemetry, Logfire, etc.) you can adapt their output to this schema instead of writing logging from scratch.

### 2. Author a golden test set

One JSON line per scenario you care about, listing the user input and the tools you expect the agent to call, in order. This is your assertion file — same role as `assert result == expected` in pytest. tracecheck has no opinion on what your agent should do; you tell it.

```jsonl
{"trace_id":"refund_ok",       "agent_name":"my_agent", "user_input":"Refund my order",   "expected_tools":["get_order","verify","issue_refund"]}
{"trace_id":"already_shipped", "agent_name":"my_agent", "user_input":"Cancel order o_99", "expected_tools":["get_order","check_shipping"]}
```

### 3. Replay the golden inputs through your agent, then run tracecheck

A short driver script feeds each `user_input` through your real agent (with the logger turned on) so the agent appends actual steps to `traces.jsonl`. Merge the `expected_tools` from your golden set onto the captured traces.

Then:

```bash
tracecheck run --traces traces.jsonl --config evals.yaml
```

Drop that command into a GitHub Actions step on every PR. Exit code 1 blocks the merge if any trace regresses.

## Evaluators

| Evaluator | What it checks | Status |
|---|---|---|
| `tool_accuracy` | Did the agent call the right tools in the right order? | Built |
| `step_efficiency` | Tool calls vs expected; flags consecutive duplicate calls and retries | Built |
| `failure_modes` | Tags traces with known failure shapes (loops, context overflow, tool errors) | Built |
| `context_drift` | Does the agent stay on topic across steps? (LLM as judge) | Built |
| `output_quality` | Final reply scored against a rubric, only after others pass (LLM as judge) | Built |

`output_quality` is **deferred**: the runner only invokes the judge after every other evaluator on the same trace passes. Already-broken traces do not burn judge tokens, and the report renders `[SKIP]` for them.

### What each evaluator uniquely catches

Stacking the five evaluators gives pinpoint diagnostics, because each catches a different *kind* of failure. Imagine an agent expected to call tools `[A, B, C]`:

| Trace shape | `tool_accuracy` | `step_efficiency` | `failure_modes` | `context_drift` | `output_quality` |
|---|---|---|---|---|---|
| `[A, B, C]` — perfect | PASS | PASS | PASS | PASS | PASS |
| `[A, A, A, B]` — looped on A | FAIL | FAIL | FAIL `infinite_loop` | PASS | SKIP |
| `[A, B, A, B, A, B]` — non-adjacent oscillation | FAIL | soft pass | **FAIL `infinite_loop`** | PASS | SKIP |
| `[A, B, search_unrelated, C]` — drifted off-topic | FAIL | FAIL | PASS | **FAIL** | SKIP |
| `[A, B, C]` + reply *"refunded $500"* but no refund tool ran | PASS | PASS | PASS | PASS | **FAIL** |
| `[A, B]` + step `error: "context window exceeded"` | PASS | PASS | **FAIL `context_overflow`** | PASS | SKIP |

The bolded cells are the evaluators that *only* detect that failure shape. A single evaluator alone would silently let those rows through.

### Quick mental shortcut

> **tool_accuracy** — *did you call the right things*
> **step_efficiency** — *without thrashing*
> **failure_modes** — *and if you broke, was it a known break*
> **context_drift** — *and you stayed on topic*
> **output_quality** — *and your final answer was actually good*

## How tracecheck compares to existing eval tools

There are excellent eval tools already. tracecheck does not replace them; it fills a different slot.

| Tool | Unit of evaluation | LLM key needed? | Form factor | Best for |
|---|---|---|---|---|
| **tracecheck** | Full agent trace (multi step) | Optional — 3 of 5 evaluators are deterministic | Library + CLI, drops into CI | Multi-step agent regression testing |
| Ragas | RAG retrieval + answer | Yes (LLM-judge metrics) | Library | RAG-specific metrics (faithfulness, context precision, etc.) |
| DeepEval | Single test case | Yes for most metrics | pytest-style framework | Single-turn LLM unit testing |
| TruLens | App + feedback functions | Yes | Framework + observability layer | Observability with feedback evaluation |
| Promptfoo | Prompt + provider matrix | Configurable | YAML test runner | Prompt regression across providers |

The shape that makes tracecheck useful when the others fall short:

- **Trace as the unit, not a single output.** If your agent loops three times, calls a wrong tool, and *then* produces a plausible reply, the existing tools score the reply. tracecheck scores the path.
- **Deterministic by default.** `tool_accuracy`, `step_efficiency`, and `failure_modes` need zero LLM calls. You can run hundreds of traces in CI with no API key, no rate limits, no cost. The LLM-as-judge evaluators are opt-in for finer grained checks.
- **CI-native exit codes.** No dashboard, no service. `exit 1` blocks the merge. Same form factor as `pytest`, `ruff`, `mypy`.

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

The `AnthropicJudge` enables prompt caching on the system block, so judging N traces costs roughly `1 + 0.1 * (N − 1)` system-prompt tokens. See [tracecheck/judges/](tracecheck/judges/) for the protocol and the `FakeJudge` used in tests.

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

Step types: `llm_call`, `tool_call`, `retry`, `error`. Each step may carry `latency_ms`, `tokens`, `timestamp`, and an `error` message. See [tracecheck/schema.py](tracecheck/schema.py) for the full Pydantic spec.

A `.jsonl` file is one trace per line. A `.json` file may be a single trace or an array.

## OpenTelemetry ingestion

If your agent already emits OpenTelemetry spans (LangChain callbacks, OpenAI/Anthropic instrumentation, Pydantic Logfire, OpenInference, etc.), point tracecheck at the OTLP/JSON span export directly — no hand-rolled logger required:

```bash
tracecheck run --traces my_otel_spans.json --config evals.yaml
```

The loader auto-detects OTLP/JSON (any file with a top-level `resourceSpans` key) and maps the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) onto the trace schema:

| OTel attribute | Maps to |
|---|---|
| `gen_ai.tool.name` | `Step.name` on a `tool_call` step |
| `gen_ai.request.model` / `gen_ai.system` | `Step.name` on an `llm_call` step |
| `gen_ai.usage.input_tokens` (or `prompt_tokens`) | `Step.tokens.prompt` |
| `gen_ai.usage.output_tokens` (or `completion_tokens`) | `Step.tokens.completion` |
| `span.startTimeUnixNano` | `Step.timestamp` |
| `endTime − startTime` | `Step.latency_ms` |
| `span.status.code = ERROR` | `Step` flagged with the error message |
| `resource.service.name` | `Trace.agent_name` |

Spans are grouped by `traceId` and ordered by start time. Spans that don't match a recognised pattern are dropped quietly.

```python
from tracecheck import load_otel_traces

traces = load_otel_traces("spans.json")
```

**One caveat:** OTel-ingested traces have no `expected_tools` — OTel does not carry your test assertions. To run `tool_accuracy` on them you write a small golden test set and merge `expected_tools` in. The LLM-as-judge evaluators (`context_drift`, `output_quality`) work fine without it.

## Pydantic AI integration

If you build with [Pydantic AI](https://ai.pydantic.dev/), the adapter turns an `AgentRunResult` directly into a tracecheck `Trace` — no hand-rolled logger, no OTel exporter, just one function call after each run:

```python
from pydantic_ai import Agent
from tracecheck import pydantic_ai_to_trace, run_evals

agent = Agent("anthropic:claude-sonnet-4-5", tools=[get_order, verify, issue_refund])

result = await agent.run("Refund my order")

trace = pydantic_ai_to_trace(
    result,
    trace_id="refund_happy_path",
    agent_name="support_agent",
    expected_tools=["get_order", "verify", "issue_refund"],
)

reports = run_evals([trace], "evals.yaml")
```

What the adapter does:

- Walks the `result.all_messages()` list in chronological order
- Emits one `llm_call` step per `TextPart`
- Merges each `ToolCallPart` with its matching `ToolReturnPart` (linked by `tool_call_id`) into a single `tool_call` step — `input` from the call args, `output` from the return content
- `RetryPromptPart` becomes a `retry` step so `step_efficiency` and `failure_modes` can score it
- Pulls `user_input` from the first `UserPromptPart`

The adapter is **duck-typed**: tracecheck does not import `pydantic_ai`, so installing tracecheck adds zero new dependencies to your project. You install Pydantic AI; you hand us the result.

## LangChain / LangGraph integration

If your agent is built with LangGraph (or any LangChain runnable), collect the events emitted by `graph.astream_events(...)` and hand them to the adapter:

```python
from tracecheck import langgraph_events_to_trace, run_evals

events = []
async for ev in graph.astream_events({"input": "Refund my order"}, version="v2"):
    events.append(ev)

trace = langgraph_events_to_trace(
    events,
    trace_id="refund_happy_path",
    agent_name="support_agent",
    expected_tools=["get_order", "verify", "issue_refund"],
)

reports = run_evals([trace], "evals.yaml")
```

What the adapter does:

- Pairs every `on_*_start` event with its matching `on_*_end` (or `on_*_error`) by `run_id`
- Emits one `tool_call` step per `on_tool_start` (input from call args, output from the return)
- Emits one `llm_call` step per `on_chat_model_start` / `on_llm_start` (model name from `metadata.ls_model_name`)
- Tool errors and chat-model errors become `error` steps
- Drops unrelated event types (chain, retriever, parser, etc.) so the trace stays focused on the agent's decisions
- Tries to extract `user_input` from the first chat call's human message

Duck-typed: tracecheck does not import `langchain` or `langgraph`. Same zero-dependency story as the Pydantic AI adapter.

## Roadmap

- [x] Trace ingestion (JSON, JSONL)
- [x] Tool call accuracy evaluator
- [x] Step efficiency evaluator
- [x] Failure mode detection
- [x] Context drift evaluator
- [x] Output quality evaluator
- [x] Static HTML report (`--output html`)
- [x] OpenTelemetry span ingest
- [x] PyPI release
- [x] Pydantic AI native integration
- [x] LangGraph / LangChain adapter

## Examples

See [examples/](examples/):
- [basic_usage.py](examples/basic_usage.py) — programmatic API
- [sample_traces.jsonl](examples/sample_traces.jsonl) — three traces (pass, fail, edge case)
- [evals.yaml](examples/evals.yaml) — minimal deterministic config (no LLM key needed)
- [evals_with_judge.yaml](examples/evals_with_judge.yaml) — full config with the LLM-as-judge evaluators enabled

For the fastest possible end-to-end demo, clone the repo and run the example traces straight away:

```bash
pip install tracecheck
git clone https://github.com/Mohdtalibakhtar/tracecheck && cd tracecheck/examples
tracecheck run --traces sample_traces.jsonl --config evals.yaml --output html --out report.html
open report.html
```

## Architecture

```
traces.jsonl ──► ingest ──► Trace[] ──► runner ──► [Evaluator.evaluate(trace)] ──► report
                                            ▲
                                         evals.yaml
```

The library is intentionally small: a Pydantic schema, a loader, a registry of evaluators, and a runner that walks every trace through every configured evaluator. CLI and library share the same code path.

## License

MIT
