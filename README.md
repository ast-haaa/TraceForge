# AgentReplay (TraceForge)

Deterministic recording, zero-LLM replay, fork & diff, and regression test generation for autonomous LLM agents (LangGraph).

---

## Important Notice: Regression Test Generator Scope

> **Domain-Specific Limitation:**  
> The current Phase 4 regression test generator (`agentreplay/test_gen.py`) is a **refund-domain template** specifically designed for the LangGraph Refund Agent. It imports `demo_agent` internals (such as `ORDERS`, `demo_agent.tools`, `issue_refund`, and `reset_ledger`) to construct invariant side-effect assertions around policy boundaries (`days_old`).  
> **It does not generalise to arbitrary agents yet.** Generalisation to arbitrary agent architectures and generic tool schemas will be addressed in future phases.

---

## Core Capabilities

- **Phase 1: Deterministic Trace Recording**  
  Captures agent executions (LLM calls, tool calls, state changes, latencies) into SQLite. Features automatic redaction of sensitive data (API keys, emails, phone numbers) and volatile field stripping (UUIDs, timestamps) for consistent hashing.
- **Phase 2: Deterministic Replay**  
  Replays recorded traces without live LLM or tool calls by matching events on `(node_name, input_hash, occurrence)`. Fails fast with `UnrecordedCall` on unrecorded operations.
- **Phase 3: Fork & Diff**  
  Branches executions at any step, applies mocks or corrected policy overrides, and runs the agent forward while redirecting side-effecting tools to an isolated sandbox ledger. Computes event-by-event diffs highlighting the first divergent step.
- **Phase 4: Regression Test Generation & Validation**  
  Synthesizes a standalone pytest regression test (`generated_tests/test_trace_<id>.py`) from trace diffs. Asserts on side effects (ledger unchanged, `issue_refund` not called) across boundary conditions (`days_old=8`, trace's `days_old`, `days_old=30`) plus positive controls (`days_old=7`). Validates in a subprocess against real agent variants (`buggy_agent` fails, `fixed_agent` passes, `always_deny_agent` fails).

---

## Quickstart

### Setup
```powershell
pip install -e ".[dev]"
```

### Run Tests
```powershell
pytest -q
```

### Run Code Quality Checks
```powershell
ruff check .
```

### Run Phase Demos
```powershell
# Phase 1: Recording demo
python examples/record_demo.py

# Phase 2: Replay demo
python examples/replay_demo.py

# Phase 3: Fork & diff demo
python examples/fork_demo.py

# Phase 4: Test generation & validation demo
python examples/testgen_demo.py
```

### Web Dashboard
```powershell
uvicorn agentreplay.api:app --reload --port 8000
```
Open `http://localhost:8000` to inspect traces, fork and diff executions, and view/download generated regression tests.
