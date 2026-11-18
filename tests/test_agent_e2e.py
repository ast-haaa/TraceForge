# tests/test_agent_e2e.py
"""End-to-end integration test: run the demo Refund Agent and verify the recorded trace."""
import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite://")  # keep tests isolated

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import agentreplay.db as db_mod
import agentreplay.sdk as sdk_mod
from agentreplay.models import (
    Base,
    Event,
    EventKind,
    Step,
    Trace,
    TraceRunType,
    TraceStatus,
)


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch):
    """Each test gets a fresh in-memory SQLite DB."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    monkeypatch.setattr(sdk_mod, "SessionLocal", Session)
    monkeypatch.setattr(db_mod, "SessionLocal", Session)

    yield Session

    engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_agent(session_factory, **kwargs):
    """Run the refund agent inside the patched DB context."""
    from demo_agent.agent import run_refund_agent
    result = run_refund_agent(**kwargs)
    session = session_factory()
    traces = session.query(Trace).order_by(Trace.id.desc()).all()
    session.close()
    return result, traces


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRefundAgentHappyPath:
    def test_trace_is_recorded(self, isolated_db):
        result, traces = _run_agent(isolated_db, customer_id="C001", order_id="O100")
        assert len(traces) >= 1
        trace = traces[0]
        assert trace.agent_name == "refund_agent"
        assert trace.status == TraceStatus.COMPLETED
        assert trace.run_type == TraceRunType.ORIGINAL

    def test_trace_has_steps(self, isolated_db):
        _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        trace = s.query(Trace).order_by(Trace.id.desc()).first()
        steps = s.query(Step).filter_by(trace_id=trace.id).order_by(Step.index).all()
        s.close()
        assert len(steps) == 5, f"Expected 5 steps, got {len(steps)}: {[st.node_name for st in steps]}"

    def test_step_names_are_correct(self, isolated_db):
        _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        trace = s.query(Trace).order_by(Trace.id.desc()).first()
        steps = s.query(Step).filter_by(trace_id=trace.id).order_by(Step.index).all()
        s.close()
        names = [st.node_name for st in steps]
        assert names == ["get_customer", "get_orders", "read_refund_policy", "llm_decide", "issue_refund"]

    def test_events_are_recorded(self, isolated_db):
        _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        events = s.query(Event).all()
        s.close()
        assert len(events) >= 5

    def test_events_have_input_hash(self, isolated_db):
        _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        events = s.query(Event).all()
        s.close()
        for ev in events:
            assert ev.input_hash and len(ev.input_hash) == 64

    def test_state_before_after_populated(self, isolated_db):
        _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        trace = s.query(Trace).order_by(Trace.id.desc()).first()
        step = s.query(Step).filter_by(trace_id=trace.id, index=1).first()
        s.close()
        assert step.state_before is not None
        assert step.state_after is not None

    def test_final_output_stored(self, isolated_db):
        result, _ = _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        trace = s.query(Trace).order_by(Trace.id.desc()).first()
        s.close()
        assert trace.final_output is not None
        assert trace.final_output["result"] in ("approve", "deny")

    def test_refund_approved_within_7_days(self, isolated_db):
        """O100 is 3 days old → should be approved."""
        result, _ = _run_agent(isolated_db, customer_id="C001", order_id="O100")
        assert result["result"] == "approve"

    def test_refund_denied_outside_7_days(self, isolated_db):
        """O101 is 10 days old → should be denied."""
        result, _ = _run_agent(isolated_db, customer_id="C001", order_id="O101")
        assert result["result"] == "deny"


class TestRefundAgentFaultScenarios:
    def test_stale_policy_causes_wrong_approval(self, isolated_db):
        """Stale policy says 30 days → 10-day-old order incorrectly approved."""
        result, _ = _run_agent(
            isolated_db,
            customer_id="C001", order_id="O101",
            stale_policy=True,
        )
        # With stale policy (30 days), a 10-day-old order should be approved
        assert result["result"] == "approve", (
            "Expected wrong approval due to stale policy"
        )

    def test_invalid_customer_records_error(self, isolated_db):
        result, traces = _run_agent(isolated_db, customer_id="INVALID", order_id="O100")
        assert result["result"] == "error"
        assert traces[0].status == TraceStatus.COMPLETED  # error returned gracefully


class TestRedactionInTrace:
    def test_pii_not_stored_in_state(self, isolated_db):
        """Customer email must NOT appear verbatim in any step state column."""
        _run_agent(isolated_db, customer_id="C001", order_id="O100")
        s = isolated_db()
        steps = s.query(Step).all()
        s.close()
        import json
        for step in steps:
            before_str = json.dumps(step.state_before or {})
            after_str = json.dumps(step.state_after or {})
            # alice@example.com is in CUSTOMERS – must be redacted
            assert "alice@example.com" not in before_str, f"Email leaked in state_before of step {step.index}"
            assert "alice@example.com" not in after_str, f"Email leaked in state_after of step {step.index}"

    def test_no_api_keys_in_events(self, isolated_db):
        """Events with api_key in input must have that key stripped."""
        import json

        from agentreplay.sdk import Recorder
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        rec = Recorder(db=session, run_id="redact-test", agent_name="test")
        rec.start_step("n", state_before={})
        ev = rec.record_event(
            EventKind.TOOL_CALL,
            raw_input={"name": "call_api", "args": {"api_key": "sk-secret", "query": "hello"}},
            output={"data": "result"},
        )
        session.close()
        assert "sk-secret" not in (ev.raw_input or "")
        assert "api_key" not in json.loads(ev.raw_input or "{}")


class TestRunIdIdempotency:
    def test_same_run_id_does_not_duplicate_trace(self, isolated_db):
        from demo_agent.agent import run_refund_agent
        run_refund_agent(customer_id="C001", order_id="O100", run_id="fixed-id-1")
        run_refund_agent(customer_id="C001", order_id="O100", run_id="fixed-id-1")

        s = isolated_db()
        count = s.query(Trace).filter_by(run_id="fixed-id-1").count()
        s.close()
        assert count == 1
