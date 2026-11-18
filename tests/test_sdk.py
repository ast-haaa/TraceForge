# tests/test_sdk.py
"""Tests for the AgentReplay SDK – incremental writes, idempotency, crash recording."""
import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite://")  # in-memory

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agentreplay.hash import compute_input_hash
from agentreplay.models import (
    Base,
    Event,
    EventKind,
    Step,
    Trace,
    TraceStatus,
)
from agentreplay.redact import redact_sensitive

# ---------------------------------------------------------------------------
# Minimal Recorder exercised directly (no context manager overhead)
# ---------------------------------------------------------------------------

def _make_recorder(session, run_id="run-sdk-001", agent_name="test_agent"):
    """Build a Recorder backed by *session*."""
    from agentreplay.sdk import Recorder
    return Recorder(db=session, run_id=run_id, agent_name=agent_name)


def _fresh_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecorderIncremental:
    def setup_method(self):
        self.session = _fresh_session()

    def teardown_method(self):
        self.session.close()

    def test_trace_row_created_on_init(self):
        _make_recorder(self.session)
        trace = self.session.query(Trace).filter_by(run_id="run-sdk-001").first()
        assert trace is not None
        assert trace.status == TraceStatus.RECORDING

    def test_start_step_flushes_immediately(self):
        rec = _make_recorder(self.session)
        step = rec.start_step("my_node", state_before={"key": "val"})
        # Row must be in DB without an explicit commit
        fetched = self.session.query(Step).filter_by(id=step.id).first()
        assert fetched is not None
        assert fetched.node_name == "my_node"

    def test_record_event_flushes_immediately(self):
        rec = _make_recorder(self.session)
        rec.start_step("node", state_before={})
        ev = rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input={"name": "get_customer", "args": {"customer_id": "C001"}},
            output={"name": "Alice"},
        )
        fetched = self.session.query(Event).filter_by(id=ev.id).first()
        assert fetched is not None
        assert fetched.kind == EventKind.TOOL_CALL
        assert fetched.seq == 1

    def test_event_seq_increments(self):
        rec = _make_recorder(self.session)
        rec.start_step("n", state_before={})
        e1 = rec.record_event(EventKind.TOOL_CALL, raw_input={"a": 1}, output={})
        e2 = rec.record_event(EventKind.LLM_CALL, raw_input={"b": 2}, output={})
        assert e1.seq == 1
        assert e2.seq == 2

    def test_event_seq_resets_per_step(self):
        rec = _make_recorder(self.session)
        rec.start_step("step1", state_before={})
        e1 = rec.record_event(EventKind.TOOL_CALL, raw_input={"x": 1}, output={})
        rec.end_step(state_after={})

        rec.start_step("step2", state_before={})
        e2 = rec.record_event(EventKind.TOOL_CALL, raw_input={"y": 1}, output={})
        assert e1.seq == 1
        assert e2.seq == 1  # resets for new step

    def test_input_hash_computed_after_redaction(self):
        rec = _make_recorder(self.session)
        rec.start_step("n", state_before={})
        raw = {"name": "get_customer", "args": {"customer_id": "C001", "api_key": "sk-secret"}}
        ev = rec.record_event(EventKind.TOOL_CALL, raw_input=raw, output={})
        # Hash must be over redacted input (api_key removed)
        expected_hash = compute_input_hash(redact_sensitive(raw))
        assert ev.input_hash == expected_hash

    def test_redaction_applied_to_stored_raw_input(self):
        rec = _make_recorder(self.session)
        rec.start_step("n", state_before={})
        ev = rec.record_event(
            EventKind.TOOL_CALL,
            raw_input={"name": "login", "args": {"password": "hunter2"}},
            output={},
        )
        assert "hunter2" not in (ev.raw_input or "")

    def test_error_field_recorded(self):
        rec = _make_recorder(self.session)
        rec.start_step("n", state_before={})
        ev = rec.record_event(
            EventKind.LLM_CALL,
            raw_input={"prompt": "hi"},
            output={},
            error="TimeoutError: LLM did not respond",
        )
        assert ev.error == "TimeoutError: LLM did not respond"

    def test_finalize_sets_status_completed(self):
        rec = _make_recorder(self.session)
        rec.start_step("n", state_before={})
        rec.record_event(EventKind.TOOL_CALL, raw_input={}, output={})
        rec.end_step(state_after={})
        rec.finalize(final_output={"result": "ok"})
        assert rec.trace.status == TraceStatus.COMPLETED

    def test_idempotent_ingest_same_run_id(self):
        """Two Recorders with the same run_id must share the same Trace row."""
        rec1 = _make_recorder(self.session, run_id="dup-run")
        trace_id_1 = rec1.trace.id

        rec2 = _make_recorder(self.session, run_id="dup-run")
        trace_id_2 = rec2.trace.id

        assert trace_id_1 == trace_id_2
        # Only one Trace row
        count = self.session.query(Trace).filter_by(run_id="dup-run").count()
        assert count == 1

    def test_state_before_is_stored(self):
        rec = _make_recorder(self.session)
        step = rec.start_step("n", state_before={"order_id": "O100", "amount": 49.99})
        fetched = self.session.query(Step).filter_by(id=step.id).one()
        assert fetched.state_before["order_id"] == "O100"

    def test_state_after_is_stored(self):
        rec = _make_recorder(self.session)
        rec.start_step("n", state_before={})
        rec.end_step(state_after={"result": "done"})
        # Find the step
        step = self.session.query(Step).filter_by(trace_id=rec.trace.id).first()
        assert step.state_after["result"] == "done"


class TestContextManager:
    def test_clean_exit_marks_completed(self):
        from agentreplay.sdk import record_trace
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        # Monkeypatch SessionLocal to use our engine
        import agentreplay.sdk as sdk_mod
        original = sdk_mod.SessionLocal
        sdk_mod.SessionLocal = Session
        try:
            with record_trace(run_id="ctx-ok", agent_name="test") as rec:
                rec.start_step("n", state_before={})
                rec.record_event(EventKind.TOOL_CALL, raw_input={}, output={"x": 1})
                rec.end_step(state_after={})
            # After context exit, status must be COMPLETED
            s = Session()
            trace = s.query(Trace).filter_by(run_id="ctx-ok").one()
            assert trace.status == TraceStatus.COMPLETED
            s.close()
        finally:
            sdk_mod.SessionLocal = original

    def test_exception_marks_failed(self):
        from agentreplay.sdk import record_trace
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        import agentreplay.sdk as sdk_mod
        original = sdk_mod.SessionLocal
        sdk_mod.SessionLocal = Session
        try:
            with pytest.raises(ValueError):
                with record_trace(run_id="ctx-fail", agent_name="test") as rec:
                    rec.start_step("n", state_before={})
                    raise ValueError("agent crashed")

            s = Session()
            trace = s.query(Trace).filter_by(run_id="ctx-fail").one()
            assert trace.status == TraceStatus.FAILED
            s.close()
        finally:
            sdk_mod.SessionLocal = original

    def test_partial_trace_survives_crash(self):
        """Steps flushed before the crash must be in the DB."""
        from agentreplay.sdk import record_trace
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        import agentreplay.sdk as sdk_mod
        original = sdk_mod.SessionLocal
        sdk_mod.SessionLocal = Session
        try:
            with pytest.raises(RuntimeError):
                with record_trace(run_id="ctx-partial", agent_name="test") as rec:
                    # This step is flushed before the crash
                    rec.start_step("step1", state_before={"x": 1})
                    rec.record_event(EventKind.TOOL_CALL, raw_input={}, output={"y": 2})
                    rec.end_step(state_after={})
                    raise RuntimeError("mid-run crash")

            s = Session()
            trace = s.query(Trace).filter_by(run_id="ctx-partial").one()
            assert trace.status == TraceStatus.FAILED
            steps = s.query(Step).filter_by(trace_id=trace.id).all()
            assert len(steps) == 1, "step recorded before crash must be persisted"
            s.close()
        finally:
            sdk_mod.SessionLocal = original
