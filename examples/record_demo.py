"""Example script demonstrating recording a LangGraph Refund Agent execution."""

from agentreplay.db import SessionLocal
from agentreplay.models import Event, Step, Trace
from demo_agent.agent import run_refund_agent
from demo_agent.tools import get_refund_ledger, reset_ledger


def main() -> None:
    print("=" * 70)
    print("AgentReplay – Recording Demo (Failure -> Replay -> Fork Loop)")
    print("=" * 70)

    reset_ledger()

    # Record a FAILING run using stale_policy=True
    # Order O103 is 12 days old (outside 7-day policy), but stale policy (30 days) causes wrong approval.
    print("\n[Failing Run] Executing agent with stale_policy=True (Fault Injection)...")
    print("  Customer: C001 (Alice) | Order: O103 (12 days old | $79.50)")
    print("  Expected by Ground Truth Policy: DENY (older than 7 days)")
    print("  Actual with Stale Policy (30 days): APPROVE (fault!)")

    result = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Fault: stale 30-day policy wrongly approved 12-day-old order",
    )
    print(f"\nAgent Final Output Decision: {result.get('result')}")

    # Inspect fake ledger to show the real side-effect that occurred
    print("\nFake Ledger Entries (Side Effects):")
    for entry in get_refund_ledger():
        print(
            f"  - Order {entry['order_id']}: ${entry['amount']} | "
            f"{entry['reason']} | Issued at: {entry['issued_at']}"
        )

    # Print actual database records: Trace, Step, Event
    session = SessionLocal()
    try:
        trace = session.query(Trace).order_by(Trace.id.desc()).first()
        if not trace:
            print("No trace found in database!")
            return

        print("\n" + "=" * 70)
        print("DATABASE AUDIT: PERSISTED TRACE, STEP, AND EVENT ROWS")
        print("=" * 70)

        print("\n[TRACE ROW]")
        print(f"  ID           : {trace.id}")
        print(f"  Run ID       : {trace.run_id}")
        print(f"  Agent        : {trace.agent_name}")
        print(f"  Status       : {trace.status.value}")
        print(f"  Run Type     : {trace.run_type.value}")
        print(f"  Created At   : {trace.created_at}")
        print(f"  Description  : {trace.description}")
        print(f"  Final Output : {trace.final_output}")

        steps = session.query(Step).filter_by(trace_id=trace.id).order_by(Step.index).all()
        print(f"\n[STEP ROWS] (Total: {len(steps)})")
        for step in steps:
            latency = f"{step.latency_ms:.2f}ms" if step.latency_ms else "N/A"
            print(
                f"  Step #{step.index:<2d} | Node: {step.node_name:<20s} | "
                f"Model: {step.model or 'None':<12s} | Latency: {latency}"
            )

        events = (
            session.query(Event)
            .join(Step)
            .filter(Step.trace_id == trace.id)
            .order_by(Step.index, Event.seq)
            .all()
        )
        print(f"\n[EVENT ROWS] (Total: {len(events)})")
        print(
            f"  {'Step Node':<20s} | {'Seq':<4s} | {'Kind':<10s} | "
            f"{'Source':<10s} | {'Error':<12s} | {'Input Hash'}"
        )
        print("  " + "-" * 75)
        for ev in events:
            node_name = ev.step.node_name if ev.step else "unknown"
            error_str = ev.error if ev.error else "None"
            print(
                f"  {node_name:<20s} | {ev.seq:<4d} | {ev.kind.value:<10s} | "
                f"{ev.source.value:<10s} | {error_str:<12s} | {ev.input_hash[:16]}..."
            )

        print("\n" + "=" * 70)
        print(f"Trace Viewer URL: http://127.0.0.1:8000/trace/{trace.id}")
        print("To launch server: uvicorn agentreplay.api:app --reload")
        print("=" * 70 + "\n")
    finally:
        session.close()


if __name__ == "__main__":
    main()
