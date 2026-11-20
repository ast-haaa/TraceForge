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
- **Phase 6a: LangGraph Adapter**  
  Zero-instrumentation recording and replay for LangGraph `StateGraph` workflows via `AgentReplayHandler(rec)`. Automatically maps graph nodes to `Step` rows, `@tool` calls to `TOOL_CALL`, and LLMs to `LLM_CALL`. Supports 100% fidelity plain-software replay and step-level forking with sandbox isolation.

---

## LangGraph Integration (Phase 6a)

Attach `AgentReplayHandler` directly to `graph.invoke` without modifying your agent's internal logic:

```python
from agentreplay.sdk import record_trace
from agentreplay.integrations.langgraph import AgentReplayHandler
from demo_agent.lg_agent import create_refund_graph

graph = create_refund_graph()

with record_trace(agent_name="lg_refund_agent", description="LangGraph execution") as rec:
    result = graph.invoke(
        {"customer_id": "C001", "order_id": "O100"},
        config={"callbacks": [AgentReplayHandler(rec)]},
    )
    rec.finalize(final_output=result)
```

### Supported vs. Unsupported LangGraph Features
| Feature | Status | Notes |
| :--- | :--- | :--- |
| **StateGraph Synchronous (`invoke`)** | Supported | Full step, state, tool, and LLM event capture |
| **Plain-Software Replay** | Supported | 100% fidelity matching on recorded inputs |
| **Step-Level Forking** | Supported | Clean branch with sandbox tool side-effect isolation |
| **Streaming (`stream`, `astream`)** | Unsupported | Chunk-level token streams are not yet assembled |
| **Asynchronous (`ainvoke`)** | Unsupported | SDK uses synchronous SQLite sessions |
| **Nested Subgraphs** | Unsupported | Subgraphs are not yet nested into trace hierarchies |
| **Arbitrary Unwrapped Tools** | Unsupported | Tools must be standard callables mockable by `ReplayContext` |

### Patch Points for Deterministic Replay
For offline replay without real side effects, tools and LLM callables are intercepted at:
- Tools: `demo_agent.tools.*` and `demo_agent.lg_agent.*` (`get_customer`, `get_orders`, `read_refund_policy`, `issue_refund`)
- LLM: `demo_agent.llm.get_llm` and `demo_agent.lg_agent.get_llm`

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

# Phase 6a: LangGraph integration demo
python examples/langgraph_demo.py
```

### Web Dashboard
```powershell
uvicorn agentreplay.api:app --reload --port 8000
```
Open `http://localhost:8000` to inspect traces, fork and diff executions, and view/download generated regression tests.
