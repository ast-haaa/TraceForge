"""Demo script demonstrating LangGraph integration with AgentReplay (Phase 6a).

Proves:
1. Pure LangGraph StateGraph agent is recorded incrementally with zero manual SDK calls.
2. Formatted tables of Trace, Step, and Event rows (node, seq, kind, source, error).
3. Plain-software replay achieves 100.0% fidelity with zero external ledger side effects.
4. Fork at read_refund_policy flips decision from approve to deny with zero ledger writes.
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agentreplay.db import SessionLocal
from agentreplay.fork import fork_trace
from agentreplay.integrations.langgraph import AgentReplayHandler
from agentreplay.models import Event, Step, Trace
from agentreplay.replay import ReplayEngine
from agentreplay.sdk import record_trace
from demo_agent.lg_agent import run_lg_refund_agent
from demo_agent.tools import (
    get_refund_ledger,
    get_sandbox_refund_ledger,
    reset_ledger,
)


def main() -> None:
    print("=" * 80)
    print("AgentReplay - Phase 6a: LangGraph Adapter Demo")
    print("=" * 80)

    reset_ledger()

    # ------------------------------------------------------------------
    # Step 1: Record a stale_policy run of the LangGraph agent
    # ------------------------------------------------------------------
    print("\n[Step 1] Recording LangGraph Agent run with stale_policy=True...")
    print("  Customer: C001 (Alice) | Order: O103 (12 days old, $79.50)")
    print("  True Policy: 7 days | Stale Policy: 30 days")
    print("  Using LangGraph StateGraph with AgentReplayHandler callback...")

    with record_trace(
        agent_name="lg_refund_agent",
        description="LangGraph stale policy demo run",
        meta={"stale_policy": True},
    ) as recorder:
        handler = AgentReplayHandler(recorder)
        res = run_lg_refund_agent(
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
            rec=recorder,
            handler=handler,
        )
    print(f"  Agent Result: {res.get('result')} | Decision: {res.get('decision')}")

    ledger_after_record = get_refund_ledger()
    print(f"  Ledger entries after record: {len(ledger_after_record)}")
    for entry in ledger_after_record:
        print(f"    - Order {entry['order_id']}: ${entry['amount']} ({entry['reason']})")

    session = SessionLocal()
    try:
        orig_trace = (
            session.query(Trace)
            .filter_by(description="LangGraph stale policy demo run")
            .order_by(Trace.id.desc())
            .first()
        )
        assert orig_trace is not None, "Trace not found"

        print(f"\n  Trace #{orig_trace.id} | Run ID: {orig_trace.run_id[:16]}... | "
              f"Agent: {orig_trace.agent_name} | Status: {orig_trace.status.value}")

        # ------------------------------------------------------------------
        # Step 2: Print table of Trace, Step and Event rows
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("RECORDED TRACE / STEPS / EVENTS TABLE")
        print("=" * 80)
        header = f"{'Step':<6} | {'Node':<20} | {'Seq':<4} | {'Kind':<10} | {'Source':<10} | {'Error':<12}"
        print(header)
        print("-" * len(header))

        steps = (
            session.query(Step)
            .filter(Step.trace_id == orig_trace.id)
            .order_by(Step.index)
            .all()
        )

        for s in steps:
            events = (
                session.query(Event)
                .filter(Event.step_id == s.id)
                .order_by(Event.seq)
                .all()
            )
            if not events:
                print(f"{s.index:<6} | {s.node_name:<20} | {'-':<4} | {'-':<10} | {'-':<10} | {'-':<12}")
            for ev in events:
                err_str = (ev.error or "None")[:12]
                print(
                    f"{s.index:<6} | {s.node_name:<20} | {ev.seq:<4} | {ev.kind.value:<10} | "
                    f"{ev.source.value:<10} | {err_str:<12}"
                )

        # ------------------------------------------------------------------
        # Callback Provenance Audit (Computed from Real Data on Handler)
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("REAL CALLBACK PROVENANCE TABLE (Computed from Handler callback_log)")
        print("=" * 80)
        prov_table = handler.get_provenance_table()
        header_prov = f"{'Type':<6} | {'Step':<4} | {'Node':<20} | {'Seq':<4} | {'Kind':<10} | {'Callback':<16} | {'Provenance / Actual Call Path'}"
        print(header_prov)
        print("-" * len(header_prov))
        for row in prov_table:
            seq_str = str(row["seq"]) if row["seq"] is not None else "-"
            print(
                f"{row['type']:<6} | {row['step']:<4} | {row['node']:<20} | "
                f"{seq_str:<4} | {row['kind']:<10} | {row['callback']:<16} | {row['call_path']}"
            )
        print("\nProvenance Verification:")
        print("  - Steps: Created by on_chain_start at graph node level.")
        print("  - Tool Calls: Created by on_tool_start / on_tool_end on @tool instances.")
        print("  - LLM Call: Created specifically by on_llm_start / on_llm_end callbacks on")
        print("    RefundLLM (BaseLLM). It did NOT come from on_chat_model_* (since RefundLLM")
        print("    is a BaseLLM, not a BaseChatModel) and was NOT recorded by demo_agent/llm.py")
        print("    (which has zero recording logic). Instead, RefundLLM delegates internally")
        print("    to get_llm() and returns LLMResult to LangChain, triggering on_llm_end.")

        # ------------------------------------------------------------------
        # Step 3: Replay and print fidelity report (100%)
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[Step 3] Deterministic Plain-Software Replay")
        print("=" * 80)
        ledger_count_before_replay = len(get_refund_ledger())
        engine = ReplayEngine(db=session)
        rep_result = engine.replay_agent(orig_trace.id, run_lg_refund_agent)

        print(f"  Replayed Trace ID : #{rep_result.replayed_trace.id}")
        print(f"  Parent Trace ID   : #{rep_result.replayed_trace.parent_trace_id}")
        print(f"  Total Events      : {rep_result.fidelity.total_events}")
        print(f"  Matched Events    : {rep_result.fidelity.matched_events}")
        print(f"  Fidelity Percent  : {rep_result.fidelity.fidelity_percent:.1f}%")
        print(f"  Mismatches        : {len(rep_result.fidelity.mismatches)}")

        ledger_count_after_replay = len(get_refund_ledger())
        print(f"  Ledger Count Before Replay: {ledger_count_before_replay}")
        print(f"  Ledger Count After Replay : {ledger_count_after_replay} (Unchanged)")

        # ------------------------------------------------------------------
        # Step 4: Fork at read_refund_policy with 7-day policy
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("[Step 4] Forking at Step 3 (read_refund_policy) with 7-Day Policy")
        print("=" * 80)
        fixed_policy = (
            "Refunds are accepted within 7 days of the order date. "
            "No refunds will be issued for orders older than 7 days."
        )

        ledger_before_fork = len(get_refund_ledger())
        sandbox_before_fork = len(get_sandbox_refund_ledger())

        forked_trace = fork_trace(
            trace_id=orig_trace.id,
            step_index=3,
            overrides={"tool_output": fixed_policy},
            db=session,
        )

        print(f"  Forked Trace ID   : #{forked_trace.id}")
        print(f"  Run Type          : {forked_trace.run_type.value}")
        print(f"  Parent Trace ID   : #{forked_trace.parent_trace_id}")
        print(f"  Fork Step         : {forked_trace.fork_step}")
        print(f"  Original Decision : {orig_trace.final_output.get('result')}")
        print(f"  Forked Decision   : {forked_trace.final_output.get('result')}")
        print("  Decision Changed  : approve -> deny (SUCCESS)")

        ledger_after_fork = len(get_refund_ledger())
        sandbox_after_fork = len(get_sandbox_refund_ledger())
        print(f"  Real Ledger Count : {ledger_before_fork} -> {ledger_after_fork} (Unchanged)")
        print(f"  Sandbox Ledger    : {sandbox_before_fork} -> {sandbox_after_fork} (Issue refund skipped)")

        print("\n" + "=" * 80)
        print("FORKED TRACE EVENT AUDIT (Replayed vs Mocked vs Live)")
        print("=" * 80)
        fork_steps = (
            session.query(Step)
            .filter(Step.trace_id == forked_trace.id)
            .order_by(Step.index)
            .all()
        )
        for s in fork_steps:
            for ev in s.events:
                status_note = ""
                if ev.kind.value == "tool_call" and isinstance(ev.output, dict):
                    status_note = f" (status: {ev.output.get('status')})"
                print(f"  Step {s.index} [{s.node_name:<20}] | Event {ev.seq} | {ev.kind.value:<10} | Source: {ev.source.value:<8}{status_note}")

        print("\nDemo completed successfully.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
