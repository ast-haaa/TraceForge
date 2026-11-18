# tests/test_models.py
"""Verify DB schema creation and basic ORM operations."""
import datetime

from agentreplay.models import (
    EvalLabel,
    Event,
    EventKind,
    Step,
    Trace,
    TraceRunType,
    TraceStatus,
)


def test_schema_creates_all_tables(db_session):
    """All models must be mappable – just committing an empty transaction is enough."""
    db_session.commit()


def test_trace_creation_and_retrieval(db_session):
    t = Trace(
        run_id="run-test-001",
        agent_name="test_agent",
        status=TraceStatus.RECORDING,
        run_type=TraceRunType.ORIGINAL,
        created_at=datetime.datetime.now(datetime.UTC),
    )
    db_session.add(t)
    db_session.commit()

    fetched = db_session.query(Trace).filter_by(run_id="run-test-001").one()
    assert fetched.agent_name == "test_agent"
    assert fetched.status == TraceStatus.RECORDING


def test_step_unique_index_constraint(db_session):
    """Two Steps with the same (trace_id, index) must raise an IntegrityError."""
    import pytest
    from sqlalchemy.exc import IntegrityError

    t = Trace(run_id="run-dup", agent_name="x", status=TraceStatus.RECORDING, run_type=TraceRunType.ORIGINAL)
    db_session.add(t)
    db_session.flush()

    s1 = Step(trace_id=t.id, index=1, node_name="n1")
    s2 = Step(trace_id=t.id, index=1, node_name="n2")
    db_session.add_all([s1, s2])
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_event_unique_seq_constraint(db_session):
    """Two Events with the same (step_id, seq) must raise an IntegrityError."""
    import pytest
    from sqlalchemy.exc import IntegrityError

    t = Trace(run_id="run-evdup", agent_name="x", status=TraceStatus.RECORDING, run_type=TraceRunType.ORIGINAL)
    db_session.add(t)
    db_session.flush()
    s = Step(trace_id=t.id, index=1, node_name="n")
    db_session.add(s)
    db_session.flush()

    e1 = Event(step_id=s.id, kind=EventKind.TOOL_CALL, seq=1, input_hash="h", output={})
    e2 = Event(step_id=s.id, kind=EventKind.TOOL_CALL, seq=1, input_hash="h2", output={})
    db_session.add_all([e1, e2])
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_event_has_error_field(db_session):
    t = Trace(run_id="run-err", agent_name="x", status=TraceStatus.FAILED, run_type=TraceRunType.ORIGINAL)
    db_session.add(t)
    db_session.flush()
    s = Step(trace_id=t.id, index=1, node_name="n")
    db_session.add(s)
    db_session.flush()
    e = Event(
        step_id=s.id, kind=EventKind.LLM_CALL, seq=1,
        input_hash="abc", output={}, error="TimeoutError: request timed out",
    )
    db_session.add(e)
    db_session.commit()

    fetched = db_session.query(Event).filter_by(id=e.id).one()
    assert fetched.error == "TimeoutError: request timed out"


def test_eval_label_separate_from_trace(db_session):
    t = Trace(run_id="run-eval", agent_name="x", status=TraceStatus.FAILED, run_type=TraceRunType.ORIGINAL)
    db_session.add(t)
    db_session.flush()
    el = EvalLabel(trace_id=t.id, injected_fault="stale_policy", ground_truth_step=3)
    db_session.add(el)
    db_session.commit()

    fetched = db_session.query(EvalLabel).filter_by(trace_id=t.id).one()
    assert fetched.injected_fault == "stale_policy"
    assert fetched.ground_truth_step == 3

    # Confirm the Trace object itself does NOT have injected_fault column
    assert not hasattr(t, "injected_fault") or not hasattr(Trace, "injected_fault")


def test_run_id_unique_constraint(db_session):
    """Inserting two Traces with the same run_id must raise IntegrityError."""
    import pytest
    from sqlalchemy.exc import IntegrityError

    t1 = Trace(run_id="same-id", agent_name="a", status=TraceStatus.RECORDING, run_type=TraceRunType.ORIGINAL)
    t2 = Trace(run_id="same-id", agent_name="b", status=TraceStatus.RECORDING, run_type=TraceRunType.ORIGINAL)
    db_session.add_all([t1, t2])
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_trace_fork_lineage(db_session):
    parent = Trace(run_id="parent", agent_name="a", status=TraceStatus.COMPLETED, run_type=TraceRunType.ORIGINAL)
    db_session.add(parent)
    db_session.flush()

    fork = Trace(
        run_id="fork-1", agent_name="a", status=TraceStatus.RECORDING,
        run_type=TraceRunType.FORK,
        parent_trace_id=parent.id,
        fork_step=3,
    )
    db_session.add(fork)
    db_session.commit()

    fetched = db_session.query(Trace).filter_by(run_id="fork-1").one()
    assert fetched.parent_trace_id == parent.id
    assert fetched.fork_step == 3
