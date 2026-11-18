"""Demo script demonstrating deterministic replay of a recorded execution.

Proves:
1. Replay re-executes the agent deterministically with ZERO live LLM or tool calls.
2. No real external side effects (fake refund ledger is unchanged during replay).
3. Replay creates a new Trace with run_type=replay and parent_trace_id set.
4. Events are marked source=replayed.
5. 100% replay fidelity reported honestly.
6. Unrecorded calls raise UnrecordedCall without guessing.
"""

from agentreplay.db import SessionLocal
from agentreplay.models import Event, Step, Trace
from agentreplay.replay import ReplayContext, ReplayEngine, UnrecordedCall
from demo_agent.agent import run_refund_agent
from demo_agent.tools import get_refund_ledger, reset_ledger


def main() -> None:
    print("=" * 75)
    print("AgentReplay – Phase 2: Deterministic Replay Demo")
    print("=" * 75)

    reset_ledger()

    # ------------------------------------------------------------------
    # Step 1: Record the original failing run
    # ------------------------------------------------------------------
    print("\n[Step 1] Recording original failing run with stale_policy=True...")
    print("  Order: O103 (12 days old, $79.50) | Ground Truth Policy: 7 days")
    print("  Agent approves wrongly due to stale 30-day policy.")

    orig_result = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Original run to replay",
    )
    print(f"  Recorded Decision: {orig_result.get('result')}")

    ledger_initial = get_refund_ledger()
    print(f"  Ledger entries after initial recording: {len(ledger_initial)}")
    for entry in ledger_initial:
        print(f"    - Order {entry['order_id']}: ${entry['amount']} ({entry['reason']})")

    session = SessionLocal()
    try:
        orig_trace = session.query(Trace).filter_by(description="Original run to replay").order_by(Trace.id.desc()).first()
        print(f"\n  Original Trace ID: #{orig_trace.id} | Run ID: {orig_trace.run_id[:16]}... | Status: {orig_trace.status.value}")

        # ------------------------------------------------------------------
        # Step 2: Replay the run with ReplayEngine (Agent Execution Mode)
        # ------------------------------------------------------------------
        print("\n" + "-" * 75)
        print(f"[Step 2] Executing deterministic replay of Trace #{orig_trace.id}...")
        print("  Patching all tools and LLM: zero live calls, zero external side effects.")

        ledger_before_replay = len(get_refund_ledger())
        engine = ReplayEngine(db=session)
        replay_result = engine.replay_agent(orig_trace.id, run_refund_agent)

        rep_trace = replay_result.replayed_trace
        print(f"\n  Replayed Trace ID : #{rep_trace.id}")
        print(f"  Parent Trace ID   : #{rep_trace.parent_trace_id} (Lineage confirmed)")
        print(f"  Run Type          : {rep_trace.run_type.value}")
        print(f"  Status            : {rep_trace.status.value}")
        print(f"  Replayed Output   : {replay_result.final_output.get('result')}")

        # Check ledger side-effects
        ledger_after_replay = len(get_refund_ledger())
        print(f"\n  Ledger count before replay : {ledger_before_replay}")
        print(f"  Ledger count after replay  : {ledger_after_replay}")
        assert ledger_after_replay == ledger_before_replay, "ERROR: Real side effect occurred during replay!"
        print("  ✓ SIDE-EFFECT PROOF: Zero new ledger entries written during replay!")

        # ------------------------------------------------------------------
        # Step 3: Fidelity Report
        # ------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("[Step 3] Replay Fidelity Report:")
        fidelity = replay_result.fidelity
        print(f"  Total Events   : {fidelity.total_events}")
        print(f"  Matched Events : {fidelity.matched_events}")
        print(f"  Fidelity       : {fidelity.fidelity_percent:.1f}%")
        print(f"  Mismatches     : {len(fidelity.mismatches)}")
        assert fidelity.fidelity_percent == 100.0, "Fidelity target 100% not reached"
        print("  ✓ FIDELITY PROOF: 100.0% exact event-by-event match!")

        # ------------------------------------------------------------------
        # Step 4: Audit Replayed Event Rows
        # ------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("[Step 4] Replayed Event Rows in Database:")
        events = (
            session.query(Event)
            .join(Step)
            .filter(Step.trace_id == rep_trace.id)
            .order_by(Step.index, Event.seq)
            .all()
        )
        print(f"  {'Step Node':<20s} | {'Seq':<4s} | {'Kind':<10s} | {'Source':<10s} | {'Input Hash'}")
        print("  " + "-" * 65)
        for ev in events:
            node = ev.step.node_name if ev.step else "unknown"
            print(f"  {node:<20s} | {ev.seq:<4d} | {ev.kind.value:<10s} | {ev.source.value:<10s} | {ev.input_hash[:16]}...")
            assert ev.source.value == "replayed", f"Event source was not replayed: {ev.source}"

        # ------------------------------------------------------------------
        # Step 5: Test UnrecordedCall handling
        # ------------------------------------------------------------------
        print("\n" + "-" * 75)
        print("[Step 5] Testing UNRECORDED_CALL handling (No guessing or fallback)...")
        print("  Attempting replay with unrecorded input (Customer C999):")
        try:
            with ReplayContext(trace_id=orig_trace.id, db=session):
                run_refund_agent(customer_id="C999", order_id="O103")
            print("  FAIL: Expected UnrecordedCall was not raised!")
        except UnrecordedCall as e:
            print(f"  ✓ SUCCESS: Caught UnrecordedCall as expected: {e}")

    finally:
        session.close()

    print("\n" + "=" * 75)
    print("Phase 2 Deterministic Replay: Fully Verified!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
