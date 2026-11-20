# tests/test_generic_handler.py
"""Tests proving AgentReplayHandler is completely generic (Phase 6a).

Uses a small separate LangGraph StateGraph (not refund agent, not demo_agent.tools)
with a real langchain_core @tool function and a fake chat model from
langchain_core.language_models.fake_chat_models (100% offline, zero network).
"""
from __future__ import annotations

import uuid
from typing import Any, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from agentreplay.fork import fork_trace
from agentreplay.integrations.langgraph import AgentReplayHandler
from agentreplay.models import EventKind, EventSource, Step, Trace, TraceStatus
from agentreplay.registry import UnknownAgentError, get_agent
from agentreplay.replay import ReplayEngine
from agentreplay.sdk import record_trace


class GenericState(TypedDict, total=False):
    """Small state schema for generic test agent."""

    x: int
    y: int
    calc_result: int
    summary: str


@tool("generic_adder")
def generic_adder(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool("generic_failing_tool")
def generic_failing_tool(val: int) -> int:
    """Tool that unconditionally raises an operational error."""
    raise RuntimeError(f"Operational hardware fault at register {val}")


def build_generic_graph(fake_msg: AIMessage) -> Any:
    """Build a two-node StateGraph using real tool and fake chat model."""
    chat = GenericFakeChatModel(messages=iter([fake_msg]))

    def node_calculate(state: GenericState) -> dict[str, Any]:
        a = state.get("x", 0)
        b = state.get("y", 0)
        res = generic_adder.invoke({"a": a, "b": b})
        return {"calc_result": res}

    def node_explain(state: GenericState) -> dict[str, Any]:
        res = state.get("calc_result", 0)
        prompt = f"Explain the calculation result: {res}"
        ai_resp = chat.invoke(prompt)
        return {"summary": ai_resp.content}

    builder = StateGraph(GenericState)
    builder.add_node("calculate", node_calculate)
    builder.add_node("explain", node_explain)
    builder.add_edge(START, "calculate")
    builder.add_edge("calculate", "explain")
    builder.add_edge("explain", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# 1. Generic Handler Recording Parity (Nodes, LLM Call, Tool Call, Token Usage)
# ---------------------------------------------------------------------------


def test_generic_handler_records_nodes_tools_and_llm(db_session, monkeypatch):
    """AgentReplayHandler records nodes to Steps, @tool to TOOL_CALL, and
    FakeChatModel to LLM_CALL with token metrics and model metadata.
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    fake_msg = AIMessage(
        content="Calculation approved: sum is 42.",
        usage_metadata={"input_tokens": 14, "output_tokens": 9, "total_tokens": 23},
    )
    graph = build_generic_graph(fake_msg)

    with record_trace(
        agent_name="generic_math_agent",
        description="Generic handler test execution",
    ) as rec:
        handler = AgentReplayHandler(rec)
        final_res = graph.invoke(
            {"x": 18, "y": 24},
            config={"callbacks": [handler]},
        )
        rec.finalize(final_output=final_res)

    tr = (
        db_session.query(Trace)
        .filter_by(description="Generic handler test execution")
        .one()
    )
    assert tr.status == TraceStatus.COMPLETED
    assert tr.agent_name == "generic_math_agent"
    assert tr.final_output["calc_result"] == 42
    assert "sum is 42" in tr.final_output["summary"]

    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) == 2
    assert [s.node_name for s in steps] == ["calculate", "explain"]

    # Step 1: calculate node with generic_adder tool call
    step1 = steps[0]
    assert step1.state_before["x"] == 18
    assert step1.state_after["calc_result"] == 42
    assert len(step1.events) == 1
    ev_tool = step1.events[0]
    assert ev_tool.kind == EventKind.TOOL_CALL
    assert ev_tool.source == EventSource.RECORDED
    assert "generic_adder" in ev_tool.raw_input
    assert ev_tool.output == 42

    # Step 2: explain node with GenericFakeChatModel llm call
    step2 = steps[1]
    assert step2.state_before["calc_result"] == 42
    assert len(step2.events) == 1
    ev_llm = step2.events[0]
    assert ev_llm.kind == EventKind.LLM_CALL
    assert ev_llm.source == EventSource.RECORDED
    assert "Explain the calculation result: 42" in ev_llm.raw_input

    llm_out = ev_llm.output
    assert llm_out["response"] == "Calculation approved: sum is 42."
    assert llm_out["model"] == "GenericFakeChatModel"
    assert llm_out["input_tokens"] == 14
    assert llm_out["output_tokens"] == 9
    assert llm_out["total_tokens"] == 23

    # Step row aggregated metrics
    assert step2.model == "GenericFakeChatModel"
    assert step2.input_tokens == 14
    assert step2.output_tokens == 9


# ---------------------------------------------------------------------------
# 2. Tool Exception Leaves Failed Trace With Error
# ---------------------------------------------------------------------------


def test_generic_tool_exception_leaves_failed_trace_with_error(db_session, monkeypatch):
    """An unhandled tool exception marks the trace FAILED and records the error."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    def node_faulty(state: GenericState) -> dict[str, Any]:
        generic_failing_tool.invoke({"val": 99})
        return {}

    builder = StateGraph(GenericState)
    builder.add_node("faulty_node", node_faulty)
    builder.add_edge(START, "faulty_node")
    builder.add_edge("faulty_node", END)
    faulty_graph = builder.compile()

    with pytest.raises(RuntimeError, match="Operational hardware fault at register 99"):
        with record_trace(
            agent_name="generic_failing_agent",
            description="Generic failing tool test execution",
        ) as rec:
            handler = AgentReplayHandler(rec)
            faulty_graph.invoke({"x": 1}, config={"callbacks": [handler]})

    tr = (
        db_session.query(Trace)
        .filter_by(description="Generic failing tool test execution")
        .one()
    )
    assert tr.status == TraceStatus.FAILED

    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) == 1
    assert steps[0].node_name == "faulty_node"

    assert len(steps[0].events) == 1
    ev = steps[0].events[0]
    assert ev.kind == EventKind.TOOL_CALL
    assert "Operational hardware fault at register 99" in (ev.error or "")


# ---------------------------------------------------------------------------
# 3. Interleaved Callbacks Matched by run_id
# ---------------------------------------------------------------------------


def test_generic_interleaved_callbacks_matched_by_run_id(db_session, monkeypatch):
    """Concurrent/nested callbacks are isolated and matched strictly by run_id."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    with record_trace(
        agent_name="generic_interleave_agent",
        description="Generic interleaved callbacks test",
    ) as rec:
        handler = AgentReplayHandler(rec)

        # 1. Start Node 1
        node1_id = uuid.uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={"val": 1},
            run_id=node1_id,
            metadata={"langgraph_node": "node_alpha"},
            tags=["graph:step:1"],
        )

        # 2. Start Node 2 (interleaved before Node 1 ends)
        node2_id = uuid.uuid4()
        handler.on_chain_start(
            serialized=None,
            inputs={"val": 2},
            run_id=node2_id,
            metadata={"langgraph_node": "node_beta"},
            tags=["graph:step:2"],
        )

        # 3. Tool call inside Node 1
        tool1_id = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "alpha_tool"},
            input_str="{}",
            run_id=tool1_id,
            parent_run_id=node1_id,
            inputs={"num": 100},
        )

        # 4. Tool call inside Node 2
        tool2_id = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "beta_tool"},
            input_str="{}",
            run_id=tool2_id,
            parent_run_id=node2_id,
            inputs={"num": 200},
        )

        # 5. End tool 2, then tool 1
        handler.on_tool_end(output={"res": "beta_done"}, run_id=tool2_id)
        handler.on_tool_end(output={"res": "alpha_done"}, run_id=tool1_id)

        # 6. End Node 2, then Node 1
        handler.on_chain_end(outputs={"out": "beta"}, run_id=node2_id)
        handler.on_chain_end(outputs={"out": "alpha"}, run_id=node1_id)

    tr = (
        db_session.query(Trace)
        .filter_by(description="Generic interleaved callbacks test")
        .one()
    )
    steps = (
        db_session.query(Step)
        .filter(Step.trace_id == tr.id)
        .order_by(Step.index)
        .all()
    )
    assert len(steps) == 2

    step_alpha = next(s for s in steps if s.node_name == "node_alpha")
    step_beta = next(s for s in steps if s.node_name == "node_beta")

    assert len(step_alpha.events) == 1
    assert step_alpha.events[0].output == {"res": "alpha_done"}

    assert len(step_beta.events) == 1
    assert step_beta.events[0].output == {"res": "beta_done"}


# ---------------------------------------------------------------------------
# 4. Unknown Agent Raises Clear Registration Error
# ---------------------------------------------------------------------------


def test_unknown_agent_raises_clear_registration_error(db_session, monkeypatch):
    """Forking or replaying an unregistered agent name raises UnknownAgentError
    with clear guidance to call register_agent.
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Direct registry lookup
    with pytest.raises(UnknownAgentError, match="Please register it with `agentreplay.register_agent"):
        get_agent("unregistered_robot_agent")

    # 2. Fork an unregistered agent
    with record_trace(
        agent_name="unregistered_robot_agent",
        description="Unregistered agent trace to fork",
    ) as rec:
        rec.start_step("step1", state_before={})
        rec.record_event(kind=EventKind.TOOL_CALL, raw_input={"tool": "noop"}, output={"ok": True})
        rec.end_step(state_after={})
        rec.finalize(final_output={"done": True})

    tr = (
        db_session.query(Trace)
        .filter_by(description="Unregistered agent trace to fork")
        .one()
    )

    with pytest.raises(UnknownAgentError, match="Please register it with `agentreplay.register_agent"):
        fork_trace(trace_id=tr.id, step_index=1, db=db_session)

    # 3. Replay an unregistered agent
    engine = ReplayEngine(db=db_session)
    with pytest.raises(UnknownAgentError, match="Please register it with `agentreplay.register_agent"):
        engine.replay_agent(trace_id=tr.id)
