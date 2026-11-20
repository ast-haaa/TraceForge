# docs/DECISIONS.md
# Architecture & Design Decisions

## D1 – Flat step list, not a graph
Steps are stored as a sequential list (Step.index) rather than a DAG.
LangGraph's execution is already linearised at record time; replaying in
order is simpler and the UI is easier to render.  If true graph branching
is needed in Phase 3+ we can add a `parent_step_id` column.

## D2 – Event split from Step
Splitting events (llm_call | tool_call | state) out of Step allows the
replay engine to match on `(node_name, input_hash, occurrence)` rather
than sequential index.  This means replay is robust to prompt-length
changes that shift token counts but don't change the logical call.

## D3 – input_hash computed after redaction
Hashing happens *after* PII/secret stripping.  Consequence: a prompt
containing a customer e-mail and the same prompt with the e-mail redacted
hash to the same value.  This is intentional – it means replay still works
even if the redaction regex changes slightly between runs.

## D4 – No Celery/Redis in MVP
Background work uses FastAPI BackgroundTasks.  Good enough for single-node
demo.  Celery would add operational complexity for no benefit at this scale.

## D5 – SQLite default, Postgres opt-in via DATABASE_URL
SQLite is zero-config for local dev and CI.  `check_same_thread=False` is
set so FastAPI's thread-pool workers can share a connection.  Postgres
support is free because SQLAlchemy abstracts the dialect.

## D6 – EvalLabel separate from Trace
`injected_fault` and `ground_truth_step` live in `EvalLabel`, not `Trace`.
This guarantees that `investigate.py` cannot accidentally read ground-truth
labels (enforced by a test that checks the module's import list).

## D7 – run_id for idempotency
The SDK accepts a caller-supplied `run_id` (UUID4 by default).  Retried
ingest calls are no-ops: the existing Trace row is returned.  This is
critical for at-least-once delivery in fault-injection benchmarks.

## D8 – UTC-aware datetimes everywhere
All `DateTime` columns use `timezone=True` and
`datetime.datetime.now(datetime.timezone.utc)`.  Avoids DST confusion and
makes cross-timezone comparisons safe.

## D9 – Pure-Python stub LLM for demo
The demo agent uses a deterministic Python function instead of a live LLM.
This means the demo runs with no API key, CI is cheap, and boundary
conditions are testable without flakiness.  Swapping in a real LLM
requires only changing `_stub_llm()`.

## D10 – Generated tests assert on behaviour, not LLM text
Tests mock tool functions via `unittest.mock` and assert on *side effects*
(e.g. `issue_refund` not called), not on the LLM's exact response text.
This keeps tests stable across model versions.

## D11 – Fork Execution Model
Forking branches an existing trace at `step_index`:
1. Steps prior to `step_index` are replayed from recording (`source=EventSource.REPLAYED`).
2. At `step_index`, overrides are applied (replace tool output, prompt, or model; `source=EventSource.MOCKED`).
3. Steps after `step_index` run live code forward from restored `Step.state_before` (`source=EventSource.LIVE`).
Lineage is maintained via `Trace.parent_trace_id` and `Trace.fork_step`.

## D12 – Side-effect Sandbox Isolation
During a fork, side-effecting operations (such as `issue_refund`) write strictly to an isolated
`_sandbox_refund_ledger`, leaving production and real ledgers completely untouched.

## D13 – UNRECORDED_CALL Fail-Fast Policy
If a forked run attempts a call that has no recording and no user-supplied mock, the fork stops
immediately. The trace is marked `status=FAILED` with an explicit `UNRECORDED_CALL` error listing
the node, kind, and input hash. Plain software determinism never guesses or hallucinates fallbacks.

## D14 – Event-by-Event Diff Engine
`diff_traces()` compares two traces event by event, identifying the exact `first_divergent_step`,
`first_divergent_node`, and full set of `changed_events`. The server-rendered UI presents
aligned side-by-side steps with soft red (#ffeef0) for the original and soft green (#e6ffed) for
the fork, strictly preserving HTML escaping without `|safe`.

## D15 – Deterministic Test Generation & Subprocess Validation
Generated regression tests derive their invariants deterministically from diff analysis between
the failed and fixed runs. Tests are validated by running a pytest subprocess against both the buggy
variant (must fail) and the fixed variant (must pass), plus a positive control to ensure over-blocking
agents (e.g. always-deny) are rejected. File paths strictly use integer trace IDs, code is never executed
in-process via `exec()`, and all displays are HTML-escaped.

> **Refund-Domain Template Notice:**
> The current test generator (`agentreplay/test_gen.py`) is intentionally a **refund-domain template**
> tailored to the LangGraph Refund Agent. It imports `demo_agent` internals (e.g. `ORDERS`, `demo_agent.tools`,
> `issue_refund`, and `reset_ledger`) and generates assertions based on refund policy boundaries (`days_old`).
> It does **not** generalise to arbitrary or third-party agent architectures yet. Generalising test generation
> to arbitrary agents will require semantic tool-effect schema extraction and LLM-assisted synthesis in future phases.

### Values Derived from Traces vs. Template Constants
- **Derived from Traces**:
  - `failed_trace_id`, `fixed_trace_id`: Integer IDs of the failing run and verified fork run.
  - `agent_name`: Derived from `failed_trace.agent_name`.
  - `customer_id`: Target customer identifier extracted from trace input / final output.
  - `amount`: Target transaction amount extracted from `failed_trace.final_output["amount"]`.
  - `failed_days_old`: The actual failing order's age (e.g. 12 days) extracted from `failed_trace.final_output["days_old"]` or step state (not hardcoded).
  - `item`: Item description extracted from the target order in trace state.
  - `fixed_agent_version`: Version identifier recorded on `fixed_trace.agent_version`.
- **Template Constants & Policy Boundary Rules**:
  - Negative control boundary values: `8` (the immediate policy boundary: 7 + 1 days) and `30` (the outer window corresponding to the stale 30-day policy). Parametrized together with the trace-derived `failed_days_old` (e.g. `[8, 12, 30]`).
  - Positive control boundary values: `7` (exact boundary day: within 7-day policy limit) and `3` (nominal inside-window order). Both must result in approved refunds and ledger modification.
  - Side-effect assertions: Negative controls assert `result == 'deny'`, `refund_called == 0`, and empty ledger; positive controls assert `result == 'approve'`, `refund_called == 1`, and non-empty ledger.
  - Fixture `agent_under_test`: Dynamically resolves the agent callable from `AGENTREPLAY_AGENT` (`"module:callable"`), ensuring generated tests contain no variant branches, `stale_policy` flags, or hardcoded result dictionaries.

## D16 – Database Schema Evolution & Dev Auto-Migration
The lightweight SQLite column auto-migration in `agentreplay/db.py` (`ALTER TABLE generated_tests ADD COLUMN ...`)
is strictly a dev-only convenience to support fast local schema evolution without wiping existing traces.
Production environments must disable this mechanism and execute version-controlled schema migrations using Alembic.

## D17 – Real LLM Adapter, Determinism, & Non-Deterministic Variance
The agent framework supports live OpenAI-compatible endpoints (`demo_agent/llm.py`) alongside the deterministic stub:
- **Provider Switching**: Controlled by `LLM_PROVIDER=stub|openai_compatible` (default: `stub`). CI and all unit tests run 100% offline without network access.
- **Pydantic Validation & Retries**: Decisions are validated against `DecisionOutput(decision="approve"|"deny", reason=str)`. On malformed JSON or validation failure, it retries once with explicit JSON instructions; subsequent failures are logged as event errors.
- **Rate Limit Resilience**: Implements exponential backoff respecting HTTP 429 `Retry-After` headers.
- **Secret Isolation**: `LLM_API_KEY` and `Authorization` headers are stripped prior to event storage. No database column stores secrets.
- **Prompt Determinism**: Computed `days_old` is injected into the prompt, strictly excluding volatile values (such as today's date, ISO timestamps, or UUIDs) to ensure prompt reproducibility across dates.

### Real-LLM Empirical Variance (Groq `openai/gpt-oss-120b`)
Running `scripts/live_check.py` 10 times with `temperature=0.0` against Groq on the stale policy scenario (order 12 days old under 30-day policy) revealed:
- **Decision Consistency**: 10/10 runs approved the refund (100% approval rate, 0% denial, 0% error).
- **Token Variance**: 10/10 runs consumed exactly 238 prompt tokens and 73 completion tokens (311 total tokens).
- **Latency Variance**: Mean latency was 866.6 ms, with an observed minimum of 655 ms and maximum of 1082 ms.
- **Replay Fidelity**: Replaying recorded real-LLM trace #95 with `ReplayEngine.replay_agent` matched 5/5 events (100.0% fidelity) with zero live network calls.
- **Fork Safety**: Forking trace #95 at Step 3 with the 7-day policy replayed steps 1–2, applied the override at step 3, invoked the live LLM at step 4 to yield `DENY`, and kept all side-effecting tools sandboxed.

## D18 – LangGraph Adapter Architecture, Callback Provenance & Agent Registry (Phase 6a)
The LangGraph integration (`agentreplay/integrations/langgraph.py:AgentReplayHandler`) attaches to LangGraph runs via standard LangChain callback configuration:
```python
with record_trace(agent_name="lg_refund_agent", description="...") as rec:
    graph.invoke(inputs, config={"callbacks": [AgentReplayHandler(rec)]})
```
This enables zero-instrumentation tracing: agent code contains no `rec.start_step` or `rec.record_event` invocations.

### Callback Provenance in the LangGraph Demo
In the LangGraph refund agent execution:
- **`Step` Creation**: Produced at the graph node execution level via `on_chain_start` and finalized by `on_chain_end`. Tag filtering (`graph:step:*`) isolates true node executions from LangGraph's internal conditional routing sequences (`seq:step:*`).
- **`Event` Creation**: **Events are never produced by `on_chain_*` callbacks.**
  - `tool_call` events are generated by `@tool` invocations via `on_tool_start` and `on_tool_end` (or `on_tool_error`).
  - `llm_call` events are generated by LLM callbacks (`on_llm_start` / `on_llm_end` on `BaseLLM` wrappers, or `on_chat_model_start` / `on_llm_end` on LangChain ChatModels).
  - In `demo_agent/lg_agent.py`, the `llm_call` event specifically fired from `on_llm_start` -> `on_llm_end` on `RefundLLM(BaseLLM)` which delegates internally to our custom `demo_agent.llm.get_llm()` wrapper; it did not fire `on_chat_model_*` (since `RefundLLM` is a `BaseLLM`, not a LangChain `BaseChatModel`). Neither `demo_agent/tools.py` nor `demo_agent/llm.py` contains direct recording logic.
  - Token metrics (`input_tokens`, `output_tokens`, `total_tokens`) are read directly from `usage_metadata` on output `AIMessage` or `llm_output`.

### Agent Registry System (`register_agent`)
To decouple `fork.py` and `replay.py` from specific agent implementations, a registration system was introduced:
```python
register_agent(
    name="lg_refund_agent",
    runner=run_lg_refund_agent,
    runner_from_step=run_lg_refund_agent_from_step,
    patch_targets=[
        "demo_agent.tools.get_customer",
        "demo_agent.lg_agent.get_customer",
        ...
    ],
)
```
- Hardcoded `elif orig_trace.agent_name == ...` branches in `fork.py` were replaced with registry lookups via `get_agent(name)`.
- Hardcoded patch target lists in `replay.py` were replaced by dynamic collection across registered agents.
- Unknown agent names raise `UnknownAgentError` (subclass of `ValueError`) explaining how to call `register_agent`.

### Fork Overrides & State-Driven Nodes
Fork overrides for `lg_refund_agent` currently live inside node state (`start_step` and `overrides`):
- **Why `start_step` is in node state**: LangGraph `StateGraph` workflows always start execution at `START`. In the absence of an external LangGraph persistence checkpointer (e.g. `MemorySaver`), jumping to an intermediate fork node (such as Step 3) requires a router conditional edge from `START` that reads `start_step`.
- **Why prompt overrides are in node state**: Prompts in LangGraph nodes are typically formatted inline from node state. While tool outputs can be intercepted via monkeypatching at the tool layer, dynamically constructed prompt strings must either be passed in state or intercepted at the LLM wrapper level. For simplicity and reliability in the demo agent, `_extract_*_override` checks remain in the node functions.

### Unsupported LangGraph Features & Known Limitations
- **Streaming (`graph.stream()`, `graph.astream()`)**: Token-by-token or chunk-by-chunk streaming is not yet supported. Only completed node invocations and generation results are recorded.
- **Asynchronous Execution (`graph.ainvoke()`, `asyncio`)**: The callback handler and SDK use synchronous SQLAlchemy sessions. Asynchronous graph execution will block or cause concurrency conflicts on shared sessions.
- **Parallel Branches**: Graphs with parallel or concurrent branches that interleave out-of-order execution cannot be deterministically replayed.
- **Nested Subgraphs**: Subgraphs running as nodes emit child chain callbacks that are currently flattened or skipped rather than nested into hierarchical trace trees.
- **Arbitrary Third-Party Tools**: Tools that bypass standard Python callables (e.g. C extensions, direct raw HTTP clients embedded in custom classes) cannot be intercepted by `ReplayContext` without custom patch point registration.

## Known limitations (Phase 1-6a)
- Refund-domain template test generation: Test generation is currently a domain-specific template that
  directly imports `demo_agent` internals (mock orders, ledger reset, `issue_refund` tools) and synthesizes
  refund policy invariants. It does not generalise to arbitrary agents yet.
- SQLite Auto-Migration is dev-only: In-place column additions via SQLite PRAGMA checks do not handle table restructuring, column renaming, or constraints; production databases must use Alembic.
- No streaming ingest endpoint yet (SDK writes directly to DB).
- No auth on the FastAPI endpoints.
- Trace viewer has no pagination; will be slow with many traces.
- The demo agent is synchronous; LangGraph's async streaming is not yet hooked.
- Timestamps embedded inside strings: Regex normalization in `strip_volatile_fields` strips standard full ISO 8601 timestamps (`YYYY-MM-DDTHH:MM:SS...`) and UUIDs from strings. Non-standard date/time strings (e.g. localized dates like `18/11/2026`, epoch millisecond integers, or natural language times like "2 hours ago") embedded in prompt text will not be stripped and will alter the computed `input_hash`. Conversely, if an agent legitimately depends on the exact sub-second timestamp string to make a decision, regex substitution to `<VOLATILE_TIMESTAMP>` collapses those differences.
