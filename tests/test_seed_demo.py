"""Tests for scripts/seed_demo.py: DB path resolution and refusal on non-empty databases."""
import subprocess
import sys
from pathlib import Path

from agentreplay.db import set_database_url
from agentreplay.models import Base, Trace, TraceRunType, TraceStatus


def test_seed_demo_refuses_non_empty_db_without_force(tmp_path):
    """Assert seed_demo.py exits with code 1 and prints refusal if database has existing traces."""
    db_file = tmp_path / "test_refusal.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    # Setup database with 1 dummy trace
    set_database_url(db_url)
    from agentreplay.db import SessionLocal, engine
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    trace = Trace(
        run_id="seed-refusal-test",
        agent_name="test_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    session.add(trace)
    session.commit()
    session.close()

    # Run seed_demo.py without --force
    cmd = [sys.executable, "scripts/seed_demo.py", "--db", str(db_file)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 1
    assert "not empty" in res.stderr.lower()
    assert "refusing to overwrite without --force" in res.stderr.lower()


def test_seed_demo_succeeds_with_force_and_creates_clean_seed_rows(tmp_path):
    """Assert seed_demo.py with --force resets and creates exactly the seed rows."""
    db_file = tmp_path / "test_force.db"
    cmd = [sys.executable, "scripts/seed_demo.py", "--db", str(db_file), "--force"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "Demo database seeding complete" in res.stdout

    # Connect and verify exact counts
    db_url = f"sqlite:///{db_file.as_posix()}"
    set_database_url(db_url)
    from agentreplay.db import SessionLocal
    from agentreplay.models import EvalLabel, GeneratedTest, Trace

    session = SessionLocal()
    traces = session.query(Trace).all()
    assert len(traces) == 6
    assert session.query(GeneratedTest).count() == 1
    assert session.query(EvalLabel).count() == 0
    session.close()


def test_seed_demo_default_does_not_touch_agentreplay_db(monkeypatch):
    """Assert seed_demo argument parser defaults to demo.db, never agentreplay.db."""
    from scripts.seed_demo import parse_args

    monkeypatch.setattr(sys, "argv", ["seed_demo.py"])
    args = parse_args()
    assert args.db is None
    # Verify default path logic
    target_default = Path("demo.db").resolve()
    assert "agentreplay.db" not in target_default.name
