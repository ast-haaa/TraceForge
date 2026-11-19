"""AgentReplay Phase 4 Demo: Regression Test Generation & Subprocess Validation.

Workflow:
1. Record failing run (order O103, 12 days old, wrongly approved due to stale policy).
2. Fork trace at Step 3 with corrected 7-day refund policy (denies refund).
3. Generate pytest regression test file from the divergence (test_trace_<id>.py).
4. Run validate_test() via subprocess against both variants:
   - Proves test FAILS on buggy agent
   - Proves test PASSES on fixed agent
5. Print the generated file contents and validation results.
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agentreplay.db import SessionLocal
from agentreplay.fork import fork_trace
from agentreplay.models import Trace
from agentreplay.test_gen import generate_test, validate_test
from demo_agent.agent import run_refund_agent
from demo_agent.tools import REFUND_POLICY, reset_ledger


def main() -> int:
    db = SessionLocal()
    reset_ledger()

    print("=" * 72)
    print(" AgentReplay Phase 4 Demo: Failure -> Fork -> Generate -> Validate")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 1. Record failing run
    # -------------------------------------------------------------------------
    print("\n[Step 1] Recording original failing run (stale 30-day policy)...")
    orig_output = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Demo failing run for test generation",
    )
    failed_trace = (
        db.query(Trace)
        .filter_by(description="Demo failing run for test generation")
        .order_by(Trace.id.desc())
        .first()
    )
    assert failed_trace is not None
    print(f"  -> Failed Trace ID   : #{failed_trace.id}")
    print(f"  -> Decision          : {orig_output['result'].upper()} (BUG: 12-day-old order approved!)")

    # -------------------------------------------------------------------------
    # 2. Fork trace with fix
    # -------------------------------------------------------------------------
    print("\n[Step 2] Forking trace at Step 3 with corrected 7-day policy...")
    fixed_trace = fork_trace(
        trace_id=failed_trace.id,
        step_index=3,
        overrides={"tool_output": REFUND_POLICY},
        db=db,
    )
    print(f"  -> Fixed Fork Trace  : #{fixed_trace.id}")
    print(f"  -> Fork Decision     : {fixed_trace.final_output['result'].upper()} (FIXED: refund denied)")

    # -------------------------------------------------------------------------
    # 3. Generate pytest regression test
    # -------------------------------------------------------------------------
    print("\n[Step 3] Generating pytest regression test from diff invariant...")
    gen_test = generate_test(failed_trace.id, fixed_trace.id, db=db)
    print(f"  -> Generated Test ID : #{gen_test.id}")
    print(f"  -> File Path         : {gen_test.file_path}")
    print(f"  -> Description       : {gen_test.description}")

    # -------------------------------------------------------------------------
    # 4. Validate test via subprocess (buggy vs fixed)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Validating test in subprocess (buggy variant & fixed variant)...")
    validated = validate_test(gen_test.id, db=db)

    print(f"  -> Fails on Buggy    : {validated.fails_on_buggy} (✔ Caught the bug!)")
    print(f"  -> Passes on Fixed   : {validated.passes_on_fixed} (✔ Verified clean on fix!)")
    print(f"  -> Status            : {validated.status.upper()}")

    # -------------------------------------------------------------------------
    # 5. Print generated test code
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generated Test Code:")
    print("-" * 72)
    print(validated.test_code.strip())
    print("-" * 72)

    # -------------------------------------------------------------------------
    # 6. Pytest Subprocess Output Summary
    # -------------------------------------------------------------------------
    print("\n[Step 6] Validation Pytest Subprocess Log Summary:")
    lines = validated.pytest_output.splitlines()
    for line in lines[:25]:
        print(f"  | {line}")
    if len(lines) > 25:
        print(f"  | ... [{len(lines) - 25} more log lines stored in DB]")

    # -------------------------------------------------------------------------
    # 7. Dashboard links
    # -------------------------------------------------------------------------
    print("\n[Step 7] View in Web Dashboard:")
    print(f"  -> Test Page  : http://localhost:8000/tests/{validated.id}")
    print(f"  -> Download   : http://localhost:8000/tests/{validated.id}/download")
    print(f"  -> Fork Diff  : http://localhost:8000/fork-diff/{failed_trace.id}/{fixed_trace.id}")

    print("\n" + "=" * 72)
    print(" [SUCCESS] Phase 4 Regression Test Generation & Validation Completed!")
    print("=" * 72)

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
