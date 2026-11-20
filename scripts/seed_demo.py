#!/usr/bin/env python3
"""scripts/seed_demo.py: Seed demo database with realistic offline agent runs.

Executes real agent workflows with the stub LLM to create:
1. Refund Agent: failing run (stale 30-day policy), its replay, and its fork (fixed 7-day policy).
2. LangGraph Agent: failing run (stale 30-day policy), its replay, and its fork (fixed 7-day policy).
3. Generated & Validated Regression Test: synthesized from diff and validated in subprocess.

All records are created via genuine code paths (zero hand-crafted database inserts).
Target database is specified via --db or DATABASE_URL (defaults to demo.db, never agentreplay.db).
Refuses to run against a non-empty database unless --force is provided.
"""
import argparse
import os
import sys

# Ensure UTF-8 output across all platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Force offline stub LLM mode
os.environ["LLM_PROVIDER"] = "stub"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed demo database with realistic offline agent runs.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Target SQLite database file (defaults to demo.db in cwd, never developer's agentreplay.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if the database already contains records (resets demo tables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine target database URL
    if args.db:
        db_path = os.path.abspath(args.db).replace("\\", "/")
        target_db_url = f"sqlite:///{db_path}"
    elif os.environ.get("DATABASE_URL"):
        target_db_url = os.environ["DATABASE_URL"]
    else:
        # Strict default: demo.db in cwd, never touching the developer's agentreplay.db!
        db_path = os.path.abspath("demo.db").replace("\\", "/")
        target_db_url = f"sqlite:///{db_path}"

    os.environ["DATABASE_URL"] = target_db_url

    from agentreplay.db import SessionLocal, init_db, set_database_url
    from agentreplay.fork import fork_trace
    from agentreplay.integrations.langgraph import AgentReplayHandler
    from agentreplay.models import (
        EvalLabel,
        Event,
        EventKind,
        GeneratedTest,
        Step,
        Trace,
        TraceRunType,
    )
    from agentreplay.replay import ReplayEngine
    from agentreplay.sdk import record_trace
    from agentreplay.test_gen import generate_test, validate_test
    from demo_agent.agent import run_refund_agent
    from demo_agent.lg_agent import run_lg_refund_agent
    from demo_agent.tools import REFUND_POLICY, reset_ledger

    set_database_url(target_db_url)
    init_db()

    session = SessionLocal()

    # Refusal check: non-empty database check
    try:
        existing_traces = session.query(Trace).count()
    except Exception:
        existing_traces = 0

    if existing_traces > 0:
        if not args.force:
            print(
                f"Error: Database at {target_db_url} is not empty (contains {existing_traces} trace(s)). "
                f"Refusing to overwrite without --force.",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            print(f"Notice: Database contains {existing_traces} trace(s). --force specified: resetting demo tables...")
            try:
                session.query(GeneratedTest).delete()
                session.query(Event).delete()
                session.query(Step).delete()
                session.query(Trace).delete()
                session.query(EvalLabel).delete()
                session.commit()
            except Exception as exc:
                session.rollback()
                print(f"Warning: Failed to clean tables: {exc}", file=sys.stderr)

    print("=" * 76)
    print("AgentReplay - Seed Demo Database (Offline / Stub LLM)")
    print(f"Target Database: {target_db_url}")
    print("=" * 76)

    engine = ReplayEngine(db=session)

    # -------------------------------------------------------------------------
    # 1. Refund Agent: Failing Run -> Replay -> Fork
    # -------------------------------------------------------------------------
    print("\n[1/3] Seeding Refund Agent Workflows...")
    reset_ledger()

    # 1a. Failing run
    print("  -> Executing failing run with stale_policy=True...")
    run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Refund Agent: failing run (12-day order approved under stale policy)",
    )
    trace_refund_fail = (
        session.query(Trace)
        .filter_by(description="Refund Agent: failing run (12-day order approved under stale policy)")
        .order_by(Trace.id.desc())
        .first()
    )
    assert trace_refund_fail is not None
    print(f"     Recorded Trace #{trace_refund_fail.id} ({trace_refund_fail.run_type.value})")

    # 1b. Deterministic Replay
    print(f"  -> Replaying Trace #{trace_refund_fail.id} (zero external calls)...")
    res_refund_replay = engine.replay_agent(trace_refund_fail.id, run_refund_agent)
    trace_refund_replay = res_refund_replay.replayed_trace
    print(f"     Replayed Trace #{trace_refund_replay.id} ({trace_refund_replay.run_type.value})")

    # 1c. Fork & Fix
    print(f"  -> Forking Trace #{trace_refund_fail.id} at Step 3 with 7-day policy...")
    trace_refund_fork = fork_trace(
        trace_id=trace_refund_fail.id,
        step_index=3,
        overrides={"tool_output": REFUND_POLICY},
        db=session,
    )
    print(f"     Forked Trace #{trace_refund_fork.id} ({trace_refund_fork.run_type.value})")

    # -------------------------------------------------------------------------
    # 2. LangGraph Agent: Failing Run -> Replay -> Fork
    # -------------------------------------------------------------------------
    print("\n[2/3] Seeding LangGraph StateGraph Agent Workflows...")
    reset_ledger()

    # 2a. Failing LangGraph run
    print("  -> Executing LangGraph failing run with AgentReplayHandler...")
    with record_trace(
        agent_name="lg_refund_agent",
        description="LangGraph Agent: failing run (12-day order approved under stale policy)",
        meta={"stale_policy": True},
    ) as rec:
        handler = AgentReplayHandler(rec)
        run_lg_refund_agent(
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
            rec=rec,
            handler=handler,
        )

    trace_lg_fail = (
        session.query(Trace)
        .filter_by(description="LangGraph Agent: failing run (12-day order approved under stale policy)")
        .order_by(Trace.id.desc())
        .first()
    )
    assert trace_lg_fail is not None
    print(f"     Recorded Trace #{trace_lg_fail.id} ({trace_lg_fail.run_type.value})")

    # 2b. LangGraph Replay
    print(f"  -> Replaying LangGraph Trace #{trace_lg_fail.id}...")
    res_lg_replay = engine.replay_agent(trace_lg_fail.id, run_lg_refund_agent)
    trace_lg_replay = res_lg_replay.replayed_trace
    print(f"     Replayed Trace #{trace_lg_replay.id} ({trace_lg_replay.run_type.value})")

    # 2c. LangGraph Fork & Fix
    print(f"  -> Forking LangGraph Trace #{trace_lg_fail.id} at Step 3 with 7-day policy...")
    trace_lg_fork = fork_trace(
        trace_id=trace_lg_fail.id,
        step_index=3,
        overrides={"tool_output": REFUND_POLICY},
        db=session,
    )
    print(f"     Forked Trace #{trace_lg_fork.id} ({trace_lg_fork.run_type.value})")

    # -------------------------------------------------------------------------
    # 3. Regression Test Generation & Validation
    # -------------------------------------------------------------------------
    print("\n[3/3] Generating and Validating Standalone Pytest Regression Test...")
    gen_test = generate_test(
        failed_trace_id=trace_refund_fail.id,
        fixed_trace_id=trace_refund_fork.id,
        db=session,
    )
    print(f"  -> Generated Test #{gen_test.id} at {gen_test.file_path}")

    val_test = validate_test(generated_test_id=gen_test.id, db=session)
    print(f"  -> Validation Status : {val_test.status.upper()}")
    print(f"  -> Fails on Buggy    : {val_test.fails_on_buggy}")
    print(f"  -> Passes on Fixed   : {val_test.passes_on_fixed}")

    # -------------------------------------------------------------------------
    # 4. Summary Table of Created Database Records
    # -------------------------------------------------------------------------
    total_traces = session.query(Trace).count()
    orig_traces = session.query(Trace).filter_by(run_type=TraceRunType.ORIGINAL).count()
    replay_traces = session.query(Trace).filter_by(run_type=TraceRunType.REPLAY).count()
    fork_traces = session.query(Trace).filter_by(run_type=TraceRunType.FORK).count()

    total_steps = session.query(Step).count()
    total_events = session.query(Event).count()
    llm_events = session.query(Event).filter_by(kind=EventKind.LLM_CALL).count()
    tool_events = session.query(Event).filter_by(kind=EventKind.TOOL_CALL).count()
    state_events = session.query(Event).filter_by(kind=EventKind.STATE).count()

    total_tests = session.query(GeneratedTest).count()
    total_eval_labels = session.query(EvalLabel).count()

    print("\n" + "=" * 76)
    print("DEMO DATABASE SEED SUMMARY (ALL ROWS CREATED VIA REAL CODE PATHS)")
    print("=" * 76)
    print(f"{'Table / Entity':<35} | {'Metric / Type':<25} | {'Count':<10}")
    print("-" * 76)
    print(f"{'Trace (Total)':<35} | {'All execution runs':<25} | {total_traces:<10}")
    print(f"{'  ├─ Original Traces':<35} | {'run_type=original':<25} | {orig_traces:<10}")
    print(f"{'  ├─ Replay Traces':<35} | {'run_type=replay':<25} | {replay_traces:<10}")
    print(f"{'  └─ Fork Traces':<35} | {'run_type=fork':<25} | {fork_traces:<10}")
    print(f"{'Step (Total)':<35} | {'Graph nodes executed':<25} | {total_steps:<10}")
    print(f"{'Event (Total)':<35} | {'Granular event records':<25} | {total_events:<10}")
    print(f"{'  ├─ LLM Calls':<35} | {'kind=llm_call':<25} | {llm_events:<10}")
    print(f"{'  ├─ Tool Calls':<35} | {'kind=tool_call':<25} | {tool_events:<10}")
    print(f"{'  └─ State Changes':<35} | {'kind=state':<25} | {state_events:<10}")
    print(f"{'GeneratedTest (Total)':<35} | {'Validated regression tests':<25} | {total_tests:<10}")
    print(f"{'EvalLabel (Total)':<35} | {'Ground truth labels':<25} | {total_eval_labels:<10}")
    print("=" * 76)
    print("Demo database seeding complete.\n")


if __name__ == "__main__":
    main()
