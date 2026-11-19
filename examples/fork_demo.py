"""AgentReplay Fork Demo: End-to-end loop demonstration.

Workflow:
1. Record failing run (order O103, 12 days old, wrongly approved due to stale policy).
2. Fork trace at Step 3 (`read_refund_policy`) with the corrected 7-day policy.
3. Observe agent running forward: decision changes from approve -> deny, refund skipped.
4. Verify safety: side effects write strictly to sandbox ledger, real ledger untouched.
5. Run event-by-event diff: highlights first divergent step and changed events.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agentreplay.db import SessionLocal
from agentreplay.diff import diff_traces
from agentreplay.fork import fork_trace
from agentreplay.models import Trace
from demo_agent.agent import run_refund_agent
from demo_agent.tools import (
    REFUND_POLICY,
    get_refund_ledger,
    get_sandbox_refund_ledger,
    reset_ledger,
)


def main() -> int:
    db = SessionLocal()
    reset_ledger()

    print("=" * 70)
    print(" AgentReplay Phase 3 Demo: Failure -> Fork -> Fix -> Diff")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Record failing run
    # -------------------------------------------------------------------------
    print("\n[Step 1] Recording original failing run (stale 30-day policy)...")
    orig_output = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Demo failing run: 12-day-old order wrongly approved",
    )

    orig_trace = (
        db.query(Trace)
        .filter_by(description="Demo failing run: 12-day-old order wrongly approved")
        .order_by(Trace.id.desc())
        .first()
    )
    assert orig_trace is not None, "Failed to record original trace"

    real_ledger_before = get_refund_ledger()
    print(f"  -> Original Trace ID : #{orig_trace.id} (run_id={orig_trace.run_id})")
    print(f"  -> Order              : O103 ({orig_output['days_old']} days old)")
    print(f"  -> Agent Decision     : {orig_output['result'].upper()} (BUGGY: exceeded 7-day policy window!)")
    print(f"  -> Real Ledger Count  : {len(real_ledger_before)} entry (refund amount=${orig_output['amount']})")

    # -------------------------------------------------------------------------
    # 2. Fork trace at Step 3 with corrected 7-day policy
    # -------------------------------------------------------------------------
    print("\n[Step 2] Forking Trace at Step 3 (read_refund_policy) with 7-day policy...")
    fixed_policy = REFUND_POLICY
    print(f"  -> Injected Policy: {fixed_policy[:60]}...")

    forked_trace = fork_trace(
        trace_id=orig_trace.id,
        step_index=3,
        overrides={"tool_output": fixed_policy},
        db=db,
    )

    print(f"  -> Fork Trace ID      : #{forked_trace.id}")
    print(f"  -> Lineage            : Parent #{forked_trace.parent_trace_id} @step {forked_trace.fork_step}")
    print(f"  -> Run Type           : {forked_trace.run_type.value}")
    print(f"  -> Status             : {forked_trace.status.value}")

    # -------------------------------------------------------------------------
    # 3. Observe agent running forward
    # -------------------------------------------------------------------------
    print("\n[Step 3] Fork Execution Results:")
    fork_output = forked_trace.final_output
    print(f"  -> New Agent Decision : {fork_output['result'].upper()} (FIXED: refund denied!)")
    print(f"  -> Refund Status      : {fork_output['refund']['status']} ({fork_output['refund']['reason']})")

    # -------------------------------------------------------------------------
    # 4. Verify safety sandbox
    # -------------------------------------------------------------------------
    print("\n[Step 4] Safety Sandbox Verification:")
    real_ledger_after = get_refund_ledger()
    sandbox_ledger_after = get_sandbox_refund_ledger()
    print(f"  -> Real Ledger Count    : {len(real_ledger_after)} (UNTOUCHED: {len(real_ledger_before)} before)")
    print(f"  -> Sandbox Ledger Count : {len(sandbox_ledger_after)} (isolated sandbox)")
    assert len(real_ledger_after) == len(real_ledger_before), "Real ledger was contaminated during fork!"

    # -------------------------------------------------------------------------
    # 5. Run Event-by-Event Diff
    # -------------------------------------------------------------------------
    print("\n[Step 5] Event-by-Event Diff Comparison:")
    diff = diff_traces(orig_trace.id, forked_trace.id, db=db)

    print(f"  -> Divergence Detected : {diff.diverged}")
    print(f"  -> First Divergent Step: Step {diff.first_divergent_step} ('{diff.first_divergent_node}')")
    print(f"  -> Total Changed Events: {len(diff.changed_events)}")
    print("\n  Step-by-Step Alignment:")
    for s in diff.steps:
        status_str = "[DIVERGED]" if s.diverged else "[MATCH]   "
        first_reason = s.events[0].reason if s.events and s.events[0].reason else "identical"
        print(f"    {status_str} Step {s.step_index}: {s.node_name:<20} | reason: {first_reason}")

    # -------------------------------------------------------------------------
    # 6. Web UI Links
    # -------------------------------------------------------------------------
    print("\n[Step 6] View in Web Dashboard:")
    print(f"  -> Original Trace : http://localhost:8000/trace/{orig_trace.id}")
    print(f"  -> Fork Trace     : http://localhost:8000/trace/{forked_trace.id}")
    print(f"  -> Visual Diff    : http://localhost:8000/fork-diff/{orig_trace.id}/{forked_trace.id}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Phase 3 Fork + Diff completed successfully!")
    print("=" * 70)

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
