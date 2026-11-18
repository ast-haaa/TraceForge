"""End-to-end Phase 1 smoke test.

Runs the Refund Agent against a real SQLite file, queries the DB, and
prints a summary to stdout. Run with the venv python from the project root.
"""

import json
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import agentreplay.db as db_mod
import agentreplay.sdk as sdk_mod
from agentreplay.models import Base, Event, Step, Trace
from demo_agent.agent import run_refund_agent
from demo_agent.tools import get_refund_ledger, reset_ledger

DB_PATH = "e:/Projects/TraceForge/phase1_verify.db"

RUNS = [
    {"customer_id": "C001", "order_id": "O100", "description": "Happy path – within 7 days"},
    {"customer_id": "C001", "order_id": "O101", "description": "Denied – 10 days old"},
    {
        "customer_id": "C001",
        "order_id": "O101",
        "stale_policy": True,
        "description": "Fault: stale policy",
    },
    {"customer_id": "INVALID", "order_id": "O100", "description": "Fault: invalid customer"},
]


def setup_temp_db() -> sessionmaker:
    os.environ["DATABASE_URL"] = f"sqlite:///{DB_PATH}"
    db_file = DB_PATH.replace("/", os.sep)
    if os.path.exists(db_file):
        os.remove(db_file)

    engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    db_mod.engine = engine
    db_mod.SessionLocal = session_factory
    sdk_mod.SessionLocal = session_factory

    return session_factory


def main() -> None:
    session_factory = setup_temp_db()

    print("=" * 65)
    print("AgentReplay – Phase 1 End-to-End Verification")
    print("=" * 65)

    for kwargs in RUNS:
        reset_ledger()
        result = run_refund_agent(**kwargs)
        desc = kwargs.get("description", "")
        print(f"\n▶  {desc}")
        print(f"   result={result.get('result')}  days_old={result.get('days_old', 'N/A')}")
        ledger = get_refund_ledger()
        if ledger:
            print(f"   Ledger entry: {ledger[0]}")

    # Query the DB
    session = session_factory()
    try:
        traces = session.query(Trace).order_by(Trace.id).all()
        print("\n" + "=" * 65)
        print(f"Traces in SQLite: {len(traces)}")
        print("-" * 65)
        for t in traces:
            steps = session.query(Step).filter_by(trace_id=t.id).order_by(Step.index).all()
            events = session.query(Event).join(Step).filter(Step.trace_id == t.id).all()
            print(
                f"  ID={t.id:3d}  status={t.status.value:9s}  steps={len(steps):2d}  events={len(events):2d}  run_id={t.run_id[:20]}…"
            )
            # Verify no PII in stored events
            pii_found = False
            for ev in events:
                raw = json.dumps(ev.output or {}) + (ev.raw_input or "") + (ev.raw_output or "")
                if "alice@example.com" in raw or "bob@example.com" in raw:
                    pii_found = True
            if pii_found:
                print("  ⚠  PII LEAK DETECTED in event data!")
            else:
                print("  ✓  No PII found in event data")

        events_no_hash = session.query(Event).filter(Event.input_hash.is_(None)).count()
        print(f"\nEvents without input_hash: {events_no_hash}  (should be 0)")
    finally:
        session.close()

    print("\n✓ Phase 1 verification complete.")
    print(f"  SQLite DB: {DB_PATH}")
    print("  Open the trace viewer: uvicorn agentreplay.api:app --reload")


if __name__ == "__main__":
    main()
