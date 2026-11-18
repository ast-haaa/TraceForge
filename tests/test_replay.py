"""Unit and integration tests for deterministic replay engine."""

import pytest

from agentreplay.hash import compute_input_hash
from agentreplay.models import Event, EventKind, EventSource, Step, Trace, TraceRunType, TraceStatus
from agentreplay.replay import (
    ReplayContext,
    ReplayEngine,
    ReplayEventMatcher,
    UnrecordedCall,
    calculate_replay_fidelity,
)
from demo_agent.agent import run_refund_agent
from demo_agent.tools import get_refund_ledger, reset_ledger


def test_event_matcher_same_tool_called_twice_in_order(db_session):
    """The same tool called twice with identical args must replay strictly in order by occurrence."""
    trace = Trace(
        run_id="run-twice",
        agent_name="test_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.flush()

    step = Step(trace_id=trace.id, index=1, node_name="retry_node")
    db_session.add(step)
    db_session.flush()

    raw_input = {"name": "get_customer", "args": {"customer_id": "C001"}}
    h = compute_input_hash(raw_input)

    # Event 1: first attempt returns pending
    e1 = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=1,
        input_hash=h,
        output={"status": "pending", "attempt": 1},
        source=EventSource.RECORDED,
    )
    # Event 2: second attempt with identical input returns confirmed
    e2 = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=2,
        input_hash=h,
        output={"status": "confirmed", "attempt": 2},
        source=EventSource.RECORDED,
    )
    db_session.add_all([e1, e2])
    db_session.commit()

    matcher = ReplayEventMatcher([step])

    # First match must return occurrence 0 (Event 1)
    matched1 = matcher.match("retry_node", h, EventKind.TOOL_CALL)
    assert matched1.output == {"status": "pending", "attempt": 1}

    # Second match must return occurrence 1 (Event 2)
    matched2 = matcher.match("retry_node", h, EventKind.TOOL_CALL)
    assert matched2.output == {"status": "confirmed", "attempt": 2}

    # Third call has no recorded occurrence 2 -> must raise UnrecordedCall
    with pytest.raises(UnrecordedCall) as exc_info:
        matcher.match("retry_node", h, EventKind.TOOL_CALL)
    assert exc_info.value.node == "retry_node"
    assert exc_info.value.input_hash == h


def test_changed_input_raises_unrecorded_call(db_session):
    """A changed input (e.g. different prompt or arguments) must raise UnrecordedCall."""
    trace = Trace(
        run_id="run-changed-input",
        agent_name="test_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.flush()

    step = Step(trace_id=trace.id, index=1, node_name="llm_decide")
    db_session.add(step)
    db_session.flush()

    recorded_prompt = {"prompt": "Order O100 DAYS_OLD: 3"}
    recorded_hash = compute_input_hash(recorded_prompt)

    ev = Event(
        step_id=step.id,
        kind=EventKind.LLM_CALL,
        seq=1,
        input_hash=recorded_hash,
        output={"response": "DECISION: approve"},
        source=EventSource.RECORDED,
    )
    db_session.add(ev)
    db_session.commit()

    # Replayer has recorded_hash, but execution requests changed_hash
    changed_prompt = {"prompt": "Order O100 DAYS_OLD: 99"}  # changed days_old!
    changed_hash = compute_input_hash(changed_prompt)
    assert changed_hash != recorded_hash

    matcher = ReplayEventMatcher([step])
    with pytest.raises(UnrecordedCall) as exc_info:
        matcher.match("llm_decide", changed_hash, EventKind.LLM_CALL)

    assert exc_info.value.node == "llm_decide"
    assert exc_info.value.input_hash == changed_hash


def test_replay_stale_policy_failing_run(db_session, monkeypatch):
    """Replay a recorded stale_policy failing run:
    - Replayed outputs == recorded outputs
    - Replayed trace created with run_type=replay and parent_trace_id set
    - Events copied with source=replayed
    - Fidelity is 100%
    - Refund ledger is untouched (ZERO live side effects)
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod
    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    reset_ledger()
    # Record original failing run: order O103 (12 days old) wrongly approved due to stale policy
    orig_output = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Original failing run",
    )
    orig_trace = db_session.query(Trace).filter_by(description="Original failing run").one()
    orig_trace_id = orig_trace.id
    assert orig_output["result"] == "approve"

    ledger_after_original = get_refund_ledger()
    assert len(ledger_after_original) == 1
    orig_entry = ledger_after_original[0]
    assert orig_entry["order_id"] == "O103"

    # Now replay the trace using ReplayEngine
    engine = ReplayEngine(db=db_session)
    result = engine.replay_agent(orig_trace_id, run_refund_agent)

    # 1. Replayed output == recorded output
    assert result.final_output == orig_output
    assert result.final_output["result"] == "approve"
    assert result.final_output["order_id"] == "O103"

    # 2. Replayed Trace metadata
    rep_trace = result.replayed_trace
    assert rep_trace.run_type == TraceRunType.REPLAY
    assert rep_trace.parent_trace_id == orig_trace_id
    assert rep_trace.status == TraceStatus.COMPLETED

    # 3. Events copied with source=replayed
    rep_events = (
        db_session.query(Event)
        .join(Step)
        .filter(Step.trace_id == rep_trace.id)
        .all()
    )
    assert len(rep_events) == 5
    for ev in rep_events:
        assert ev.source == EventSource.REPLAYED

    # 4. Fidelity is 100%
    assert result.fidelity.fidelity_percent == 100.0
    assert result.fidelity.matched_events == 5
    assert len(result.fidelity.mismatches) == 0

    # 5. Ledger is UNCHANGED after replay (no live refund issued!)
    ledger_after_replay = get_refund_ledger()
    assert len(ledger_after_replay) == 1  # exactly 1, no new entry added!
    assert ledger_after_replay == ledger_after_original


def test_replay_trace_deterministic_software(db_session, monkeypatch):
    """ReplayEngine.replay_trace executes deterministic plain software replay from DB.
    Restores state from Step.state_before and produces 100% fidelity.
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod
    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    reset_ledger()
    run_refund_agent(customer_id="C001", order_id="O100", description="Happy path for trace replay")
    orig_trace = db_session.query(Trace).filter_by(description="Happy path for trace replay").one()

    ledger_before = get_refund_ledger()

    engine = ReplayEngine(db=db_session)
    result = engine.replay_trace(orig_trace.id)

    assert result.replayed_trace.run_type == TraceRunType.REPLAY
    assert result.replayed_trace.parent_trace_id == orig_trace.id
    assert result.fidelity.fidelity_percent == 100.0
    assert result.final_output == orig_trace.final_output

    # Ledger unchanged
    assert get_refund_ledger() == ledger_before


def test_replay_of_crashed_run_reproduces_same_error(db_session):
    """Replaying a crashed run reproduces the same error and preserves failed status."""
    trace = Trace(
        run_id="run-crashed",
        agent_name="refund_agent",
        status=TraceStatus.FAILED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.flush()

    step = Step(
        trace_id=trace.id,
        index=1,
        node_name="get_orders",
        state_before={"customer_id": "C001"},
        state_after={},
    )
    db_session.add(step)
    db_session.flush()

    error_msg = "DatabaseConnectionTimeout: upstream database did not respond"
    raw_input = {"name": "get_orders", "args": {"customer_id": "C001"}}
    h = compute_input_hash(raw_input)

    ev = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=1,
        input_hash=h,
        output={},
        error=error_msg,
        source=EventSource.RECORDED,
    )
    db_session.add(ev)
    db_session.commit()

    engine = ReplayEngine(db=db_session)
    result = engine.replay_trace(trace.id)

    # Replayed trace inherits status FAILED and parent lineage
    assert result.replayed_trace.status == TraceStatus.FAILED
    assert result.replayed_trace.parent_trace_id == trace.id

    # The replayed event preserves the exact same error
    rep_event = (
        db_session.query(Event)
        .join(Step)
        .filter(Step.trace_id == result.replayed_trace.id)
        .one()
    )
    assert rep_event.error == error_msg
    assert rep_event.source == EventSource.REPLAYED
    assert result.fidelity.fidelity_percent == 100.0


def test_replay_context_blocks_and_patches_every_tool(db_session):
    """Every tool in demo_agent is patched during replay so real calls cannot occur."""
    trace = Trace(
        run_id="run-patch-test",
        agent_name="refund_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.flush()

    step = Step(trace_id=trace.id, index=1, node_name="issue_refund")
    db_session.add(step)
    db_session.flush()

    tool_input = {
        "name": "issue_refund",
        "args": {"order_id": "O100", "amount": 49.99, "reason": "approved"},
    }
    h = compute_input_hash(tool_input)
    ev = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=1,
        input_hash=h,
        output={"status": "refund_issued", "order_id": "O100", "amount": 49.99},
        source=EventSource.RECORDED,
    )
    db_session.add(ev)
    db_session.commit()

    reset_ledger()
    assert len(get_refund_ledger()) == 0

    import demo_agent.tools as tools

    with ReplayContext(trace_id=trace.id, db=db_session):
        # Calling issue_refund inside ReplayContext must return the recorded output
        res = tools.issue_refund("O100", 49.99, reason="approved")
        assert res["status"] == "refund_issued"
        # And fake ledger must NOT have been called/written to!
        assert len(get_refund_ledger()) == 0


def test_calculate_replay_fidelity_honest_reporting(db_session):
    """Fidelity report accurately measures match rate and lists mismatches."""
    t1 = Trace(run_id="t1", agent_name="a", status=TraceStatus.COMPLETED)
    t2 = Trace(run_id="t2", agent_name="a", status=TraceStatus.COMPLETED)
    db_session.add_all([t1, t2])
    db_session.flush()

    s1 = Step(trace_id=t1.id, index=1, node_name="n1")
    s2 = Step(trace_id=t2.id, index=1, node_name="n1")
    db_session.add_all([s1, s2])
    db_session.flush()

    # Event 1 matches
    e1_orig = Event(step_id=s1.id, kind=EventKind.TOOL_CALL, seq=1, input_hash="h1", output={"a": 1})
    e1_rep = Event(step_id=s2.id, kind=EventKind.TOOL_CALL, seq=1, input_hash="h1", output={"a": 1})

    # Event 2 differs in output
    e2_orig = Event(step_id=s1.id, kind=EventKind.TOOL_CALL, seq=2, input_hash="h2", output={"b": 1})
    e2_rep = Event(step_id=s2.id, kind=EventKind.TOOL_CALL, seq=2, input_hash="h2", output={"b": 999})

    db_session.add_all([e1_orig, e1_rep, e2_orig, e2_rep])
    db_session.commit()

    report = calculate_replay_fidelity(t1.id, t2.id, db_session)
    assert report.total_events == 2
    assert report.matched_events == 1
    assert report.fidelity_percent == 50.0
    assert len(report.mismatches) == 1
