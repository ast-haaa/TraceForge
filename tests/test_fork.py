"""Phase 3 Tests: Forking, Diffing, Sandbox Ledger, and HTML Escaping."""

import re

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agentreplay import api
from agentreplay.api import app
from agentreplay.diff import diff_traces
from agentreplay.fork import fork_trace
from agentreplay.models import (
    Event,
    EventKind,
    EventSource,
    Step,
    Trace,
    TraceRunType,
    TraceStatus,
)
from agentreplay.replay import UnrecordedCall
from demo_agent.agent import run_refund_agent
from demo_agent.tools import (
    get_refund_ledger,
    get_sandbox_refund_ledger,
    reset_ledger,
)


@pytest.fixture
def client(db_session):
    engine = db_session.get_bind()
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[api.get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def recorded_stale_policy_trace(db_session, monkeypatch):
    """Fixture that records a failing run (order O103, 12 days old, wrongly approved due to stale policy)."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    reset_ledger()
    output = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Original stale policy run (failing)",
    )
    assert output["result"] == "approve"
    trace = (
        db_session.query(Trace)
        .filter_by(description="Original stale policy run (failing)")
        .one()
    )
    return trace


def test_fork_stale_policy_changes_decision_to_deny_and_skips_refund(
    db_session, recorded_stale_policy_trace, monkeypatch
):
    """Fork at read_refund_policy with a 7-day policy:
    - Decision changes from approve to deny
    - issue_refund is NOT called
    - Event at step 3 is marked source=mocked
    - Subsequent events marked source=live
    - Trace metadata has run_type=fork, parent_trace_id, fork_step set
    """
    orig_trace = recorded_stale_policy_trace
    fixed_policy = (
        "Refunds are accepted within 7 days of the order date. "
        "No refunds will be issued for orders older than 7 days. "
        "Refunds are credited to the original payment method within 3-5 business days."
    )

    ledger_before_fork = len(get_refund_ledger())
    sandbox_before_fork = len(get_sandbox_refund_ledger())

    # Fork at step 3 (read_refund_policy)
    forked_trace = fork_trace(
        trace_id=orig_trace.id,
        step_index=3,
        overrides={"tool_output": fixed_policy},
        db=db_session,
    )

    # 1. Trace lineage & status
    assert forked_trace.run_type == TraceRunType.FORK
    assert forked_trace.parent_trace_id == orig_trace.id
    assert forked_trace.fork_step == 3
    assert forked_trace.status == TraceStatus.COMPLETED

    # 2. Final output decision changed to deny
    assert forked_trace.final_output["result"] == "deny"
    assert forked_trace.final_output["refund"]["status"] == "skipped"

    # 3. issue_refund was not called
    assert len(get_refund_ledger()) == ledger_before_fork
    assert len(get_sandbox_refund_ledger()) == sandbox_before_fork

    # 4. Step 3 event has source=mocked
    step3 = (
        db_session.query(Step)
        .filter(Step.trace_id == forked_trace.id, Step.index == 3)
        .one()
    )
    assert len(step3.events) == 1
    assert step3.events[0].source == EventSource.MOCKED
    assert step3.events[0].output == {"policy": fixed_policy}

    # 5. Subsequent steps run live
    step4 = (
        db_session.query(Step)
        .filter(Step.trace_id == forked_trace.id, Step.index == 4)
        .one()
    )
    assert step4.events[0].source == EventSource.LIVE
    assert "DECISION: deny" in step4.events[0].output["response"]

    step5 = (
        db_session.query(Step)
        .filter(Step.trace_id == forked_trace.id, Step.index == 5)
        .one()
    )
    assert step5.events[0].source == EventSource.LIVE
    assert step5.events[0].output["status"] == "skipped"


def test_events_before_fork_step_are_identical_to_original(
    db_session, recorded_stale_policy_trace
):
    """Events before the fork step (steps 1 and 2) must be identical to the original,
    with source=replayed.
    """
    orig_trace = recorded_stale_policy_trace
    fixed_policy = "Refunds are accepted within 7 days of the order date."

    forked_trace = fork_trace(
        trace_id=orig_trace.id,
        step_index=3,
        overrides={"tool_output": fixed_policy},
        db=db_session,
    )

    orig_steps = (
        db_session.query(Step)
        .filter(Step.trace_id == orig_trace.id)
        .order_by(Step.index)
        .all()
    )
    fork_steps = (
        db_session.query(Step)
        .filter(Step.trace_id == forked_trace.id)
        .order_by(Step.index)
        .all()
    )

    # Check Steps 1 and 2 (before fork step 3)
    for idx in (0, 1):
        os = orig_steps[idx]
        fs = fork_steps[idx]
        assert os.node_name == fs.node_name
        assert os.index == fs.index
        assert os.state_before == fs.state_before
        assert os.state_after == fs.state_after

        assert len(os.events) == len(fs.events)
        for oe, fe in zip(os.events, fs.events, strict=False):
            assert oe.kind == fe.kind
            assert oe.seq == fe.seq
            assert oe.input_hash == fe.input_hash
            assert oe.output == fe.output
            assert oe.raw_input == fe.raw_input
            assert oe.raw_output == fe.raw_output
            assert fe.source == EventSource.REPLAYED


def test_diff_returns_read_refund_policy_as_first_divergent_step(
    db_session, recorded_stale_policy_trace
):
    """Diff compares original and forked traces:
    - diff returns read_refund_policy as first divergent step
    - first_divergent_step == 3
    - changed_events contains the divergent events
    """
    orig_trace = recorded_stale_policy_trace
    fixed_policy = "Refunds are accepted within 7 days of the order date."

    forked_trace = fork_trace(
        trace_id=orig_trace.id,
        step_index=3,
        overrides={"tool_output": fixed_policy},
        db=db_session,
    )

    diff = diff_traces(orig_trace.id, forked_trace.id, db=db_session)

    assert diff.diverged is True
    assert diff.first_divergent_step == 3
    assert diff.first_divergent_node == "read_refund_policy"
    assert len(diff.changed_events) >= 3

    # Steps 1 and 2 did not diverge
    step_diff1 = next(s for s in diff.steps if s.step_index == 1)
    assert step_diff1.diverged is False

    step_diff2 = next(s for s in diff.steps if s.step_index == 2)
    assert step_diff2.diverged is False

    # Step 3 diverged
    step_diff3 = next(s for s in diff.steps if s.step_index == 3)
    assert step_diff3.diverged is True


def test_unrecorded_call_stops_fork_with_unrecorded_call(
    db_session, recorded_stale_policy_trace
):
    """UNRECORDED_CALL policy: if a forked run makes a call that has no recording
    and no user-supplied mock, stop the fork, mark the trace status=failed with
    a clear UNRECORDED_CALL error, and list the call.
    """
    orig_trace = recorded_stale_policy_trace

    def failing_agent(rec, start_step, restored_state, overrides):
        # Attempts an unexpected unrecorded call
        raise UnrecordedCall(
            node="external_shipping_service",
            kind=EventKind.TOOL_CALL,
            input_hash="unrecorded_hash_12345",
        )

    # 1. With raise_on_error=True (default), it raises UnrecordedCall
    with pytest.raises(UnrecordedCall) as exc_info:
        fork_trace(
            trace_id=orig_trace.id,
            step_index=3,
            agent_fn=failing_agent,
            db=db_session,
            raise_on_error=True,
        )

    assert exc_info.value.node == "external_shipping_service"
    assert "UNRECORDED_CALL" in str(exc_info.value)

    # 2. Check that the trace was saved as FAILED in the DB with the error listed
    failed_trace = (
        db_session.query(Trace)
        .filter(Trace.run_type == TraceRunType.FORK)
        .order_by(Trace.id.desc())
        .first()
    )
    assert failed_trace is not None
    assert failed_trace.status == TraceStatus.FAILED
    assert "UNRECORDED_CALL" in failed_trace.description
    assert "external_shipping_service" in failed_trace.description

    # 3. With raise_on_error=False, it marks failed and returns the trace
    returned_trace = fork_trace(
        trace_id=orig_trace.id,
        step_index=3,
        agent_fn=failing_agent,
        db=db_session,
        raise_on_error=False,
    )
    assert returned_trace.status == TraceStatus.FAILED
    assert "UNRECORDED_CALL" in returned_trace.description


def test_sandbox_ledger_separate_from_real_ledger(
    db_session, recorded_stale_policy_trace
):
    """Safety: during a fork every side-effecting tool (issue_refund) must write
    to a sandbox ledger, never the real one.
    """
    orig_trace = recorded_stale_policy_trace
    reset_ledger()

    assert len(get_refund_ledger()) == 0
    assert len(get_sandbox_refund_ledger()) == 0

    # Fork and approve a refund: fork at step 5 and force issue_refund
    # or fork at step 4 with an approved decision
    def approving_fork_agent(rec, start_step, restored_state, overrides):
        from demo_agent.tools import issue_refund

        rec.start_step("issue_refund", state_before={"decision": "approve"})
        res = issue_refund("O103", 79.50, reason="fork test approval")
        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input={"order_id": "O103", "amount": 79.50},
            output=res,
            source=EventSource.LIVE,
        )
        rec.end_step(state_after={"refund_result": res})
        rec.finalize(final_output={"result": "approve", "refund": res})

    forked = fork_trace(
        trace_id=orig_trace.id,
        step_index=5,
        agent_fn=approving_fork_agent,
        db=db_session,
    )

    assert forked.status == TraceStatus.COMPLETED

    # Real ledger MUST be untouched (0 entries)
    real_ledger = get_refund_ledger()
    assert len(real_ledger) == 0

    # Sandbox ledger MUST have received the write
    sandbox_ledger = get_sandbox_refund_ledger()
    assert len(sandbox_ledger) == 1
    assert sandbox_ledger[0]["order_id"] == "O103"
    assert sandbox_ledger[0]["amount"] == 79.50


def test_html_escaping_holds_on_diff_and_fork_pages(
    client, db_session, recorded_stale_policy_trace
):
    """Confirm XSS prevention: HTML characters in inputs/outputs are escaped,
    and no |safe filter is used in fork_diff.html.
    """
    orig_trace = recorded_stale_policy_trace

    # Inject XSS payload into an event
    xss_payload = "<script>alert(1)</script>"
    step = orig_trace.steps[0]
    ev = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=99,
        input_hash="xss_hash_1",
        output={"xss": xss_payload},
        raw_input=f'{{"xss": "{xss_payload}"}}',
        raw_output=f'{{"xss": "{xss_payload}"}}',
        source=EventSource.RECORDED,
    )
    db_session.add(ev)
    db_session.commit()
    db_session.expire_all()

    forked_trace = fork_trace(
        trace_id=orig_trace.id,
        step_index=3,
        overrides={"tool_output": "7-day policy"},
        db=db_session,
    )

    response = client.get(f"/fork-diff/{orig_trace.id}/{forked_trace.id}")

    assert response.status_code == 200
    body = response.text

    # Unescaped raw script tag must NEVER appear
    assert "<script>alert(1)" not in body
    # Escaped entity MUST appear
    assert "&lt;script&gt;" in body

    # Confirm no |safe filter exists in fork_diff.html
    import os

    template_path = os.path.join(
        os.path.dirname(__file__), "..", "templates", "fork_diff.html"
    )
    with open(template_path, encoding="utf-8") as f:
        content = f.read()
    assert not re.search(r"\|\s*safe", content), (
        "Found |safe filter in templates/fork_diff.html"
    )


def test_api_fork_endpoints(client, db_session, recorded_stale_policy_trace):
    """Test POST /traces/{id}/fork with both Form redirect and JSON API."""
    orig_trace = recorded_stale_policy_trace

    # 1. JSON POST
    res_json = client.post(
        f"/traces/{orig_trace.id}/fork",
        json={"step_index": 3, "overrides": {"tool_output": "7-day policy"}},
    )
    assert res_json.status_code == 200
    data = res_json.json()
    assert "fork_trace_id" in data
    assert "diff_url" in data
    assert data["status"] == "completed"

    # 2. Form POST with redirect
    res_form = client.post(
        f"/traces/{orig_trace.id}/fork",
        data={
            "step_index": "3",
            "override_type": "tool_output",
            "override_value": "7-day policy",
        },
        follow_redirects=False,
    )
    assert res_form.status_code == 303
    assert "/fork-diff/" in res_form.headers["location"]
