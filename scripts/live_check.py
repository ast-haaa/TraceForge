# scripts/live_check.py
"""Live verification script for real LLM integration (Phase 4.5).

Runs the stale_policy scenario 10 times against the real LLM (Groq / OpenAI-compatible).
Prints:
1. Summary table of all 10 runs showing decisions, tokens, latencies, and reasons.
2. Aggregate metrics: counts (approve, deny, error), mean tokens, mean latency.
3. For the first approved run:
   - Replay fidelity using ReplayEngine.replay_agent (verifies zero live calls during replay).
   - Decision of a fork branched at step 3 with the corrected 7-day policy.
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from typing import Any

import dotenv

from agentreplay.db import SessionLocal, init_db
from agentreplay.fork import fork_trace
from agentreplay.models import EventKind, Step, Trace
from agentreplay.replay import ReplayEngine
from demo_agent.agent import run_refund_agent
from demo_agent.llm import clean_env_var
from demo_agent.tools import REFUND_POLICY, reset_ledger

# ---------------------------------------------------------------------------
# Formatting Helpers (safe against non-breaking hyphens / special characters)
# ---------------------------------------------------------------------------


def format_reason(reason: str, max_len: int = 60) -> str:
    """Format and truncate reason string safely."""
    r = str(reason or "")
    if len(r) > max_len:
        return r[:max_len] + "..."
    return r


def format_run_row(
    iteration: int,
    trace_id: int,
    decision: str,
    in_toks: int,
    out_toks: int,
    latency_ms: float,
    reason: str,
) -> str:
    """Format a summary row for tabular display."""
    total_toks = in_toks + out_toks
    toks_str = f"{in_toks}/{out_toks} ({total_toks})"
    lat_str = f"{latency_ms:.0f}ms"
    clean_reason = format_reason(reason, 60)
    return f"{iteration:4d} | #{trace_id:5d} | {decision.upper():8s} | {toks_str:>15s} | {lat_str:>9s} | {clean_reason}"


def format_iteration_log(
    iteration: int,
    trace_id: int,
    decision: str,
    total_toks: int,
    latency_ms: float,
    reason: str,
) -> str:
    """Format single iteration log line for progress display."""
    clean_reason = format_reason(reason, 40)
    return f"  Iteration {iteration:2d}/10: Trace #{trace_id:<4} -> {decision.upper():<7} ({total_toks:3d} toks, {latency_ms:5.0f}ms) | {clean_reason}"


def main() -> int:
    dotenv.load_dotenv()
    init_db()
    db = SessionLocal()

    api_key = clean_env_var(os.environ.get("LLM_API_KEY"), "")
    if not api_key:
        print("[ERROR] LLM_API_KEY is not set. Please set it in .env or the environment.")
        return 1

    # Ensure provider is openai_compatible
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    model = clean_env_var(os.environ.get("LLM_MODEL"), "openai/gpt-oss-120b")
    base_url = clean_env_var(os.environ.get("LLM_BASE_URL"), "https://api.groq.com/openai/v1")

    print("=" * 80)
    print(" AgentReplay Phase 4.5 Live LLM Check")
    print(f" Provider : openai_compatible ({base_url})")
    print(f" Model    : {model}")
    print(" Scenario : Stale 30-Day Policy on 12-Day-Old Order (10 Iterations)")
    print("=" * 80)

    runs_data: list[dict] = []
    first_approved_trace_id: int | None = None

    for i in range(1, 11):
        reset_ledger()
        desc = f"Live check run #{i}"

        # ONLY the LLM / agent call is inside the try/except that counts as an iteration ERROR
        call_error: Exception | None = None
        res: dict[str, Any] | None = None
        try:
            res = run_refund_agent(
                customer_id="C001",
                order_id="O103",
                stale_policy=True,
                description=desc,
            )
        except Exception as exc:
            call_error = exc

        # If LLM/agent call itself failed, record as error and continue
        if call_error is not None:
            runs_data.append({
                "iteration": i,
                "trace_id": -1,
                "decision": "error",
                "in_toks": 0,
                "out_toks": 0,
                "total_toks": 0,
                "latency_ms": 0.0,
                "reason": str(call_error),
                "status": "ERROR",
            })
            print(f"  Iteration {i:2d}/10: ERROR -> {call_error}")
            continue

        # Outside the try/except: DB query, data aggregation, formatting, printing
        assert res is not None
        tr = db.query(Trace).filter(Trace.description == desc).order_by(Trace.id.desc()).first()
        trace_id = tr.id if tr else -1

        # Find llm_decide step
        llm_step = (
            db.query(Step)
            .filter(Step.trace_id == trace_id, Step.node_name == "llm_decide")
            .first()
        )
        in_toks = llm_step.input_tokens if llm_step else 0
        out_toks = llm_step.output_tokens if llm_step else 0
        lat = llm_step.latency_ms if llm_step else 0.0

        # Extract reason from event or result
        reason = ""
        if llm_step and llm_step.events:
            ev = next((e for e in llm_step.events if e.kind == EventKind.LLM_CALL), None)
            if ev and isinstance(ev.output, dict):
                reason = str(ev.output.get("reason", ""))

        decision = res.get("result", "unknown").lower()
        if decision == "approve" and first_approved_trace_id is None and trace_id > 0:
            first_approved_trace_id = trace_id

        runs_data.append({
            "iteration": i,
            "trace_id": trace_id,
            "decision": decision,
            "in_toks": in_toks,
            "out_toks": out_toks,
            "total_toks": in_toks + out_toks,
            "latency_ms": lat,
            "reason": reason,
            "status": "OK" if "error" not in res else "ERROR",
        })

        # Progress log (printing and formatting outside the try/except)
        log_line = format_iteration_log(i, trace_id, decision, in_toks + out_toks, lat, reason)
        print(log_line)

    # Summary Table
    print("\n" + "-" * 80)
    print(f"{'Run':>4} | {'Trace':>6} | {'Decision':>8} | {'Tokens (In/Out)':>15} | {'Latency':>9} | {'Reason Summary'}")
    print("-" * 80)
    for r in runs_data:
        row_str = format_run_row(
            r["iteration"],
            r["trace_id"],
            r["decision"],
            r["in_toks"],
            r["out_toks"],
            r["latency_ms"],
            r["reason"],
        )
        print(row_str)
    print("-" * 80)

    # Aggregates
    approvals = sum(1 for r in runs_data if r["decision"] == "approve")
    denials = sum(1 for r in runs_data if r["decision"] == "deny")
    errors = sum(1 for r in runs_data if r["decision"] == "error")

    valid_runs = [r for r in runs_data if r["decision"] in ("approve", "deny")]
    mean_toks = sum(r["total_toks"] for r in valid_runs) / len(valid_runs) if valid_runs else 0.0
    mean_lat = sum(r["latency_ms"] for r in valid_runs) / len(valid_runs) if valid_runs else 0.0

    print("\n[Aggregate Metrics (Real LLM Variance)]")
    print("  Total Runs    : 10")
    print(f"  Approved      : {approvals:2d} ({approvals / 10 * 100:.0f}%)")
    print(f"  Denied        : {denials:2d} ({denials / 10 * 100:.0f}%)")
    print(f"  Errors        : {errors:2d} ({errors / 10 * 100:.0f}%)")
    print(f"  Mean Tokens   : {mean_toks:.1f} tokens")
    print(f"  Mean Latency  : {mean_lat:.1f} ms")

    # Replay & Fork on First Approved Run
    target_trace_id = first_approved_trace_id or (runs_data[0]["trace_id"] if runs_data else None)

    if target_trace_id and target_trace_id > 0:
        print(f"\n[Replay Verification: Trace #{target_trace_id}]")
        engine = ReplayEngine(db=db)
        replay_res = engine.replay_agent(
            trace_id=target_trace_id,
            agent_fn=run_refund_agent,
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
        )
        print(f"  Replayed Trace ID : #{replay_res.replayed_trace.id}")
        print(f"  Replay Fidelity   : {replay_res.fidelity.fidelity_percent:.1f}% ({replay_res.fidelity.matched_events}/{replay_res.fidelity.total_events} events matched)")
        print(f"  Replay Decision   : {replay_res.final_output.get('result', '').upper()}")

        print(f"\n[Fork Verification: Trace #{target_trace_id} -> 7-Day Policy]")
        fork_res = fork_trace(
            trace_id=target_trace_id,
            step_index=3,
            overrides={"tool_output": REFUND_POLICY},
            db=db,
        )
        fork_decision = (
            fork_res.final_output.get("result", "")
            if isinstance(fork_res.final_output, dict)
            else ""
        )
        print(f"  Fork Trace ID     : #{fork_res.id}")
        print("  Fork Step         : Step 3 (read_refund_policy override)")
        print(f"  Fork Decision     : {fork_decision.upper()} (expected: DENY for 12-day-old order)")
    else:
        print("\n[NOTICE] No successful trace recorded to replay or fork.")

    print("\n" + "=" * 80)
    print(" Live Check Completed Successfully")
    print("=" * 80)
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
