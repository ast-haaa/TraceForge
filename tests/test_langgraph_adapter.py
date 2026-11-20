# tests/test_langgraph_adapter.py
"""Tests for AgentReplay LangGraph Adapter (Phase 6a).

Verifies that LangGraph StateGraph agents are recorded, replayed, and forked
with 100% fidelity using AgentReplayHandler with zero manual recording calls.
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from agentreplay.fork import fork_trace
from agentreplay.integrations.langgraph import AgentReplayHandler
from agentreplay.models import (
    EventKind,
    EventSource,
    Step,
    Trace,
    TraceRunType,
    TraceStatus,
)
from agentreplay.replay import ReplayEngine
from agentreplay.sdk import record_trace
from demo_agent.agent import run_refund_agent
from demo_agent.lg_agent import create_refund_graph, run_lg_refund_agent
from demo_agent.tools import (
    get_refund_ledger,
    get_sandbox_refund_ledger,
    reset_ledger,
)


@pytest.fixture(autouse=True)
def clean_test_ledger():
    """Ensure clean ledgers before each test."""
    reset_ledger()
    yield
    reset_ledger()


# ---------------------------------------------------------------------------
# 1. Handler maps nodes, LLM calls and tool calls correctly
# ---------------------------------------------------------------------------


def test_handler_maps_nodes_llm_and_tools_correctly(db_session, monkeypatch):
    """AgentReplayHandler maps graph nodes to Steps, tool invocations to TOOL_CALL,
    and LLM invocations to LLM_CALL with state_before, state_after, and token metrics.
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    res = run_lg_refund_agent(
        "C001",
        "O100",
        description="Handler mapping verification run",
    )
    assert res["result"] == "approve"

    tr = (
        db_session.query(Trace)
        .filter_by(description="Handler mapping verification run")
        .one()
    )
    assert tr.status == TraceStatus.COMPLETED
    assert tr.agent_name == "lg_refund_agent"

    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) == 5
    expected_nodes = [
        "get_customer",
        "get_orders",
        "read_refund_policy",
        "llm_decide",
        "issue_refund",
    ]
    assert [s.node_name for s in steps] == expected_nodes

    for s in steps:
        assert s.state_before is not None
        assert s.state_after is not None
        assert s.latency_ms is not None and s.latency_ms >= 0

    # Step 1: get_customer tool_call
    assert len(steps[0].events) == 1
    assert steps[0].events[0].kind == EventKind.TOOL_CALL
    assert steps[0].events[0].output["name"] == "Alice Smith"

    # Step 2: get_orders tool_call
    assert len(steps[1].events) == 1
    assert steps[1].events[0].kind == EventKind.TOOL_CALL
    assert "orders" in steps[1].events[0].output

    # Step 3: read_refund_policy tool_call
    assert len(steps[2].events) == 1
    assert steps[2].events[0].kind == EventKind.TOOL_CALL
    assert "policy" in steps[2].events[0].output

    # Step 4: llm_decide llm_call
    assert len(steps[3].events) == 1
    assert steps[3].events[0].kind == EventKind.LLM_CALL
    llm_out = steps[3].events[0].output
    assert llm_out["decision"] == "approve"
    assert "input_tokens" in llm_out
    assert "output_tokens" in llm_out
    assert "model" in llm_out

    # Step 5: issue_refund tool_call
    assert len(steps[4].events) == 1
    assert steps[4].events[0].kind == EventKind.TOOL_CALL
    assert steps[4].events[0].output["status"] == "refund_issued"


# ---------------------------------------------------------------------------
# 2. Step and event counts and order match the manual agent
# ---------------------------------------------------------------------------


def test_step_and_event_counts_and_order_match_manual_agent(db_session, monkeypatch):
    """A LangGraph execution records step and event sequences matching the manual agent."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Manual agent execution
    manual_res = run_refund_agent(
        "C001",
        "O100",
        description="Manual agent benchmark trace",
    )
    manual_tr = (
        db_session.query(Trace)
        .filter_by(description="Manual agent benchmark trace")
        .one()
    )
    manual_tr_id = manual_tr.id
    manual_res_result = manual_res["result"]

    # 2. LangGraph agent execution
    lg_res = run_lg_refund_agent(
        "C001",
        "O100",
        description="LangGraph agent parity trace",
    )
    lg_tr = (
        db_session.query(Trace)
        .filter_by(description="LangGraph agent parity trace")
        .one()
    )
    lg_tr_id = lg_tr.id

    assert manual_res_result == lg_res["result"]

    manual_steps = (
        db_session.query(Step)
        .filter(Step.trace_id == manual_tr_id)
        .order_by(Step.index)
        .all()
    )
    lg_steps = (
        db_session.query(Step)
        .filter(Step.trace_id == lg_tr_id)
        .order_by(Step.index)
        .all()
    )

    assert len(manual_steps) == len(lg_steps) == 5

    for m_s, lg_s in zip(manual_steps, lg_steps, strict=False):
        assert m_s.index == lg_s.index
        assert m_s.node_name == lg_s.node_name
        assert len(m_s.events) == len(lg_s.events)
        for m_ev, lg_ev in zip(m_s.events, lg_s.events, strict=False):
            assert m_ev.kind == lg_ev.kind
            assert m_ev.seq == lg_ev.seq


# ---------------------------------------------------------------------------
# 3. Tool exception leaves a failed trace with the error
# ---------------------------------------------------------------------------


def test_tool_exception_leaves_failed_trace_with_error(db_session, monkeypatch):
    """An unhandled tool exception marks the trace FAILED and records the error."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod
    from demo_agent import tools as tools_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # Patch get_orders to raise an unexpected runtime error
    def faulty_get_orders(customer_id: str) -> list[dict[str, Any]]:
        raise ConnectionResetError("Database connection lost during order fetch")

    monkeypatch.setattr(tools_mod, "get_orders", faulty_get_orders)

    with pytest.raises(ConnectionResetError, match="Database connection lost"):
        run_lg_refund_agent(
            "C001",
            "O100",
            description="Failing tool exception trace",
        )

    tr = (
        db_session.query(Trace)
        .filter_by(description="Failing tool exception trace")
        .one()
    )
    assert tr.status == TraceStatus.FAILED

    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) >= 2
    failing_step = steps[1]  # get_orders
    assert failing_step.node_name == "get_orders"

    # Event has error recorded
    ev = failing_step.events[0]
    assert ev.kind == EventKind.TOOL_CALL
    assert "Database connection lost" in (ev.error or "")


# ---------------------------------------------------------------------------
# 4. Interleaved callbacks are matched by run_id
# ---------------------------------------------------------------------------


def test_interleaved_callbacks_matched_by_run_id(db_session, monkeypatch):
    """Callbacks interleaved between parallel or nested nodes correctly match by run_id."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    with record_trace(
        agent_name="interleave_test_agent",
        description="Interleaved callbacks verification",
    ) as rec:
        handler = AgentReplayHandler(rec)

        # 1. Start Node A
        run_a = uuid.uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={"task": "A"},
            run_id=run_a,
            metadata={"langgraph_node": "node_a"},
            tags=["graph:step:1"],
        )

        # 2. Start Node B (interleaved before Node A ends)
        run_b = uuid.uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={"task": "B"},
            run_id=run_b,
            metadata={"langgraph_node": "node_b"},
            tags=["graph:step:2"],
        )

        # 3. Tool call inside Node A
        tool_a_run = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "tool_for_a"},
            input_str="{}",
            run_id=tool_a_run,
            parent_run_id=run_a,
            inputs={"param": "A"},
        )

        # 4. Tool call inside Node B
        tool_b_run = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "tool_for_b"},
            input_str="{}",
            run_id=tool_b_run,
            parent_run_id=run_b,
            inputs={"param": "B"},
        )

        # 5. End Tool B before Tool A finishes
        handler.on_tool_end(
            output={"result": "done_B"},
            run_id=tool_b_run,
            parent_run_id=run_b,
        )

        # 6. End Tool A
        handler.on_tool_end(
            output={"result": "done_A"},
            run_id=tool_a_run,
            parent_run_id=run_a,
        )

        # 7. End Node B, then Node A
        handler.on_chain_end(
            outputs={"status_b": "ok"},
            run_id=run_b,
        )
        handler.on_chain_end(
            outputs={"status_a": "ok"},
            run_id=run_a,
        )

    tr = (
        db_session.query(Trace)
        .filter_by(description="Interleaved callbacks verification")
        .one()
    )
    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) == 2

    step_a = next(s for s in steps if s.node_name == "node_a")
    step_b = next(s for s in steps if s.node_name == "node_b")

    # Step A must contain Tool A, not Tool B
    assert len(step_a.events) == 1
    assert step_a.events[0].output == {"result": "done_A"}

    # Step B must contain Tool B, not Tool A
    assert len(step_b.events) == 1
    assert step_b.events[0].output == {"result": "done_B"}


# ---------------------------------------------------------------------------
# 5. Replay fidelity 100% and ledger unchanged
# ---------------------------------------------------------------------------


def test_replay_fidelity_100_percent_and_ledger_unchanged(db_session, monkeypatch):
    """Replaying a recorded LangGraph execution yields 100% fidelity with 0 ledger writes."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Record original failing stale policy run
    orig_output = run_lg_refund_agent(
        "C001",
        "O103",
        stale_policy=True,
        description="Original stale policy LangGraph run",
    )
    assert orig_output["result"] == "approve"
    ledger_after_record = len(get_refund_ledger())
    assert ledger_after_record == 1

    orig_tr = (
        db_session.query(Trace)
        .filter_by(description="Original stale policy LangGraph run")
        .one()
    )

    # 2. Deterministic Plain-Software Replay
    engine = ReplayEngine(db=db_session)
    rep_res = engine.replay_agent(orig_tr.id, run_lg_refund_agent)

    # 3. Assertions
    assert rep_res.fidelity.fidelity_percent == 100.0
    assert rep_res.fidelity.matched_events == rep_res.fidelity.total_events == 5
    assert rep_res.fidelity.mismatches == []
    assert len(get_refund_ledger()) == ledger_after_record  # Zero ledger changes!


# ---------------------------------------------------------------------------
# 6. Fork decision flips and issue_refund is not called
# ---------------------------------------------------------------------------


def test_fork_decision_flips_and_issue_refund_not_called(db_session, monkeypatch):
    """Forking at read_refund_policy with a 7-day policy flips decision to deny."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Record failing stale policy run (order O103 is 12 days old)
    orig_output = run_lg_refund_agent(
        "C001",
        "O103",
        stale_policy=True,
        description="Original failing run to fork",
    )
    assert orig_output["result"] == "approve"
    ledger_before_fork = len(get_refund_ledger())
    sandbox_before_fork = len(get_sandbox_refund_ledger())

    orig_tr = (
        db_session.query(Trace)
        .filter_by(description="Original failing run to fork")
        .one()
    )

    fixed_policy = (
        "Refunds are accepted within 7 days of the order date. "
        "No refunds will be issued for orders older than 7 days."
    )

    # 2. Fork at step 3 (read_refund_policy)
    forked_tr = fork_trace(
        trace_id=orig_tr.id,
        step_index=3,
        overrides={"tool_output": fixed_policy},
        db=db_session,
    )

    # Lineage assertions
    assert forked_tr.run_type == TraceRunType.FORK
    assert forked_tr.parent_trace_id == orig_tr.id
    assert forked_tr.fork_step == 3
    assert forked_tr.status == TraceStatus.COMPLETED

    # Decision flipped to deny
    assert forked_tr.final_output["result"] == "deny"
    assert forked_tr.final_output["decision"] == "deny"

    # Ledgers unchanged (issue_refund was skipped)
    assert len(get_refund_ledger()) == ledger_before_fork
    assert len(get_sandbox_refund_ledger()) == sandbox_before_fork

    # Event source auditing
    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == forked_tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) == 5

    # Steps 1 and 2: replayed from original
    assert steps[0].events[0].source == EventSource.REPLAYED
    assert steps[1].events[0].source == EventSource.REPLAYED

    # Step 3: mocked override
    assert steps[2].events[0].source == EventSource.MOCKED
    assert steps[2].events[0].output == {"policy": fixed_policy}

    # Steps 4 and 5: live forward execution
    assert steps[3].events[0].source == EventSource.LIVE
    assert steps[4].events[0].source == EventSource.LIVE
    assert steps[4].events[0].output["status"] == "skipped"


# ---------------------------------------------------------------------------
# 7. Redaction applies to handler data
# ---------------------------------------------------------------------------


def test_redaction_applies_to_handler_data(db_session, monkeypatch):
    """PII and secret keys passed into nodes and tools are redacted before persistence."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    run_lg_refund_agent(
        "C001",
        "O100",
        description="Redaction verification run",
    )

    tr = (
        db_session.query(Trace)
        .filter_by(description="Redaction verification run")
        .one()
    )

    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    cust_ev = steps[0].events[0]

    # Email address in customer output must be redacted
    assert "<REDACTED_EMAIL>" in cust_ev.raw_output
    assert "alice@example.com" not in cust_ev.raw_output


# ---------------------------------------------------------------------------
# 8. No API key or Authorization header is stored
# ---------------------------------------------------------------------------


def test_no_api_key_or_auth_header_stored(db_session, monkeypatch):
    """Authorization headers and API keys in metadata or inputs are completely stripped."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    secret_auth = "Bearer sk-proj-1234567890abcdef"
    secret_key = "key-secret987654321"

    graph = create_refund_graph()
    with record_trace(
        agent_name="lg_refund_agent",
        description="Auth header security test run",
        meta={"Authorization": secret_auth, "api_key": secret_key},
    ) as rec:
        handler = AgentReplayHandler(rec)
        inputs = {
            "customer_id": "C001",
            "order_id": "O100",
            "authorization": secret_auth,
            "api_key": secret_key,
        }
        res = graph.invoke(inputs, config={"callbacks": [handler]})
        rec.finalize(final_output=res)

    tr = (
        db_session.query(Trace)
        .filter_by(description="Auth header security test run")
        .one()
    )

    # Check Trace metadata
    assert "Authorization" not in tr.meta
    assert "api_key" not in tr.meta

    # Check Steps and Events
    steps = db_session.query(Step).filter(Step.trace_id == tr.id).all()
    for s in steps:
        assert "authorization" not in (s.state_before or {})
        assert "api_key" not in (s.state_before or {})
        for ev in s.events:
            assert secret_auth not in (ev.raw_input or "")
            assert secret_key not in (ev.raw_input or "")
            assert secret_auth not in (ev.raw_output or "")
            assert secret_key not in (ev.raw_output or "")
