# demo_agent/lg_agent.py
"""LangGraph StateGraph implementation of the Refund Agent for AgentReplay.

Rebuilds the refund agent workflow as a pure LangGraph StateGraph with nodes:
1. get_customer
2. get_orders
3. read_refund_policy
4. llm_decide
5. issue_refund

The agent code contains ZERO manual recording calls (no rec.start_step or
rec.record_event). Recording, replay, and forking are powered entirely by
AgentReplayHandler attached via config={"callbacks": [AgentReplayHandler(rec)]}.
"""
from __future__ import annotations

import datetime
from typing import Any, TypedDict

from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from agentreplay.integrations.langgraph import AgentReplayHandler
from agentreplay.models import TraceStatus
from agentreplay.sdk import record_trace
from demo_agent import llm as llm_mod
from demo_agent import tools
from demo_agent.agent import (
    _extract_model_override,
    _extract_policy_override,
    _extract_prompt_override,
)

# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------


class RefundAgentState(TypedDict, total=False):
    """Execution state passed through the LangGraph refund agent nodes."""

    customer_id: str
    order_id: str
    stale_policy: bool
    policy_days_override: int | None
    overrides: Any
    start_step: int
    customer: dict[str, Any]
    orders: list[dict[str, Any]]
    target_order: dict[str, Any] | None
    policy: str
    days_old: int
    decision: str
    llm_decision: str
    refund_result: dict[str, Any]
    result: str
    detail: str
    refund: dict[str, Any]


# ---------------------------------------------------------------------------
# LangChain Tool Wrappers (Route to demo_agent.tools functions)
# ---------------------------------------------------------------------------


@tool("get_customer")
def tool_get_customer(customer_id: str) -> dict[str, Any]:
    """Retrieve customer record by customer ID."""
    return tools.get_customer(customer_id)


@tool("get_orders")
def tool_get_orders(customer_id: str) -> dict[str, Any]:
    """Retrieve all historical orders for a given customer ID."""
    orders = tools.get_orders(customer_id)
    return {"orders": orders}


@tool("read_refund_policy")
def tool_read_refund_policy(
    stale: bool = False,
    override_policy: str | None = None,
) -> dict[str, Any]:
    """Fetch store refund policy guidelines, optionally with stale flag or override."""
    if override_policy is not None:
        return {"policy": override_policy}
    policy = tools.read_refund_policy(stale=stale)
    return {"policy": policy}


@tool("issue_refund")
def tool_issue_refund(
    order_id: str | None = None,
    amount: float | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """Execute refund disbursement, or report skipped status if denied."""
    if not order_id or amount is None:
        return {"status": "skipped", "reason": "refund denied by policy"}
    return tools.issue_refund(order_id, amount, reason=reason)


# ---------------------------------------------------------------------------
# LangChain LLM Wrapper (Delegates to demo_agent.llm.get_llm)
# ---------------------------------------------------------------------------


class RefundLLM(BaseLLM):
    """LangChain BaseLLM implementation delegating to demo_agent.llm.get_llm()."""

    policy_days: int | None = None
    model_override: str | None = None

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt = prompts[0]
        llm_fn = llm_mod.get_llm()
        resp = llm_fn(
            prompt,
            policy_days=self.policy_days,
            model=self.model_override,
        )
        text = f"DECISION: {resp.decision}  REASON: {resp.reason}"
        gen = Generation(text=text)
        llm_output = {
            "model": resp.model,
            "decision": resp.decision,
            "reason": resp.reason,
            "input_tokens": resp.prompt_tokens,
            "output_tokens": resp.completion_tokens,
            "total_tokens": resp.total_tokens,
            "cost": resp.cost,
            "latency_ms": resp.latency_ms,
            "error": resp.error,
        }
        return LLMResult(generations=[[gen]], llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        return "refund_llm"


# ---------------------------------------------------------------------------
# StateGraph Node Functions
# ---------------------------------------------------------------------------


def node_get_customer(state: RefundAgentState) -> dict[str, Any]:
    """Step 1: Look up customer information."""
    cust_id = state.get("customer_id", "")
    cust = tool_get_customer.invoke({"customer_id": cust_id})
    if "error" in cust:
        return {"customer": cust, "result": "error", "detail": cust["error"]}
    return {"customer": cust}


def node_get_orders(state: RefundAgentState) -> dict[str, Any]:
    """Step 2: Retrieve orders for customer and select target order."""
    cust_id = state.get("customer_id", "")
    order_id = state.get("order_id", "")
    res = tool_get_orders.invoke({"customer_id": cust_id})
    orders = res.get("orders", []) if isinstance(res, dict) else res
    target_order = next((o for o in orders if o.get("order_id") == order_id), None)
    if target_order is None:
        return {
            "orders": orders,
            "target_order": None,
            "result": "error",
            "detail": f"Order {order_id} not found for customer {cust_id}",
        }
    return {"orders": orders, "target_order": target_order}


def node_read_refund_policy(state: RefundAgentState) -> dict[str, Any]:
    """Step 3: Read current store refund policy."""
    start_step = state.get("start_step", 1)
    overrides = state.get("overrides")
    policy_override = (
        _extract_policy_override(overrides)
        if (start_step == 3 and overrides)
        else None
    )

    stale = state.get("stale_policy", False)
    if policy_override is not None:
        res = tool_read_refund_policy.invoke(
            {"stale": stale, "override_policy": policy_override}
        )
        policy_text = policy_override
    else:
        res = tool_read_refund_policy.invoke({"stale": stale})
        policy_text = (
            res.get("policy", "") if isinstance(res, dict) else str(res)
        )
    return {"policy": policy_text}


def node_llm_decide(state: RefundAgentState) -> dict[str, Any]:
    """Step 4: LLM decision evaluating order against policy."""
    order = state.get("target_order") or {}
    order_id = state.get("order_id") or order.get("order_id", "")
    customer = state.get("customer") or {}
    cust_name = customer.get("name", "Unknown")
    ord_amt = order.get("amount", 0.0)

    days_old = 0
    if order and "order_date" in order:
        order_date = datetime.date.fromisoformat(order["order_date"])
        days_old = (datetime.date.today() - order_date).days

    start_step = state.get("start_step", 1)
    overrides = state.get("overrides")
    prompt_override = (
        _extract_prompt_override(overrides)
        if (start_step == 4 and overrides)
        else None
    )
    model_override = (
        _extract_model_override(overrides)
        if (start_step == 4 and overrides)
        else None
    )

    if prompt_override is not None:
        prompt = prompt_override
    else:
        policy = state.get("policy", "")
        prompt = (
            f"Customer: {cust_name}\n"
            f"Order: {order_id}  Amount: ${ord_amt}\n"
            f"DAYS_OLD: {days_old}\n"
            f"Policy: {policy}\n"
            f"Decide whether to approve or deny this refund request. "
            f"Return JSON with 'decision' ('approve' or 'deny') and 'reason'."
        )

    llm = RefundLLM(
        policy_days=state.get("policy_days_override"),
        model_override=model_override,
    )
    resp_text = llm.invoke(prompt)
    decision = "approve" if "DECISION: approve" in resp_text else "deny"

    return {
        "decision": decision,
        "llm_decision": resp_text,
        "days_old": days_old,
    }


def node_issue_refund(state: RefundAgentState) -> dict[str, Any]:
    """Step 5: Disburse refund if approved, or skip if denied."""
    decision = state.get("decision", "deny")
    order = state.get("target_order") or {}
    order_id = state.get("order_id") or order.get("order_id", "")

    if decision == "approve":
        ord_amt = float(order.get("amount", 0.0))
        res = tool_issue_refund.invoke(
            {
                "order_id": order_id,
                "amount": ord_amt,
                "reason": "approved by agent",
            }
        )
        return {"refund_result": res, "result": "approve", "refund": res}
    else:
        res = tool_issue_refund.invoke(
            {
                "order_id": None,
                "amount": None,
                "reason": "refund denied by policy",
            }
        )
        return {"refund_result": res, "result": "deny", "refund": res}


# ---------------------------------------------------------------------------
# StateGraph Definition and Compilation
# ---------------------------------------------------------------------------


def create_refund_graph() -> Any:
    """Build and compile the Refund Agent LangGraph StateGraph."""
    builder = StateGraph(RefundAgentState)

    builder.add_node("get_customer", node_get_customer)
    builder.add_node("get_orders", node_get_orders)
    builder.add_node("read_refund_policy", node_read_refund_policy)
    builder.add_node("llm_decide", node_llm_decide)
    builder.add_node("issue_refund", node_issue_refund)

    def route_start(state: RefundAgentState) -> str:
        step = state.get("start_step", 1)
        if step == 2:
            return "get_orders"
        elif step == 3:
            return "read_refund_policy"
        elif step == 4:
            return "llm_decide"
        elif step == 5:
            return "issue_refund"
        return "get_customer"

    builder.add_conditional_edges(
        START,
        route_start,
        {
            "get_customer": "get_customer",
            "get_orders": "get_orders",
            "read_refund_policy": "read_refund_policy",
            "llm_decide": "llm_decide",
            "issue_refund": "issue_refund",
        },
    )

    def route_after_customer(state: RefundAgentState) -> str:
        if state.get("result") == "error":
            return END
        return "get_orders"

    def route_after_orders(state: RefundAgentState) -> str:
        if state.get("result") == "error":
            return END
        return "read_refund_policy"

    builder.add_conditional_edges(
        "get_customer",
        route_after_customer,
        {"get_orders": "get_orders", END: END},
    )
    builder.add_conditional_edges(
        "get_orders",
        route_after_orders,
        {"read_refund_policy": "read_refund_policy", END: END},
    )
    builder.add_edge("read_refund_policy", "llm_decide")
    builder.add_edge("llm_decide", "issue_refund")
    builder.add_edge("issue_refund", END)

    return builder.compile()


# Pre-compiled global graph instance
_compiled_graph = create_refund_graph()


# ---------------------------------------------------------------------------
# Agent Runner Helpers
# ---------------------------------------------------------------------------


def run_lg_refund_agent(
    customer_id: str,
    order_id: str,
    *,
    stale_policy: bool = False,
    run_id: str | None = None,
    description: str | None = None,
    meta: dict[str, Any] | None = None,
    agent_version: str = "1.0.0-langgraph",
    policy_days_override: int | None = None,
    overrides: Any = None,
    start_step: int = 1,
    restored_state: dict[str, Any] | None = None,
    rec: Any = None,
    handler: AgentReplayHandler | None = None,
) -> dict[str, Any]:
    """Execute the LangGraph Refund Agent with automatic AgentReplay recording."""
    inputs: dict[str, Any] = {
        "customer_id": customer_id,
        "order_id": order_id,
        "stale_policy": stale_policy,
        "policy_days_override": policy_days_override,
        "overrides": overrides,
        "start_step": start_step,
    }
    if restored_state:
        inputs.update(restored_state)
    inputs["start_step"] = start_step
    inputs["overrides"] = overrides

    if rec is None:
        run_meta = {"stale_policy": stale_policy}
        if meta:
            run_meta.update(meta)

        with record_trace(
            agent_name="lg_refund_agent",
            run_id=run_id,
            description=description or "LangGraph Refund Agent execution",
            meta=run_meta,
            agent_version=agent_version,
        ) as recorder:
            active_handler = handler or AgentReplayHandler(recorder)
            recorder._handler = active_handler
            res = _compiled_graph.invoke(inputs, config={"callbacks": [active_handler]})
            if recorder.trace.status == TraceStatus.RECORDING:
                recorder.finalize(final_output=res)
            return res
    else:
        active_handler = handler or AgentReplayHandler(rec)
        rec._handler = active_handler
        res = _compiled_graph.invoke(inputs, config={"callbacks": [active_handler]})
        if rec.trace.status == TraceStatus.RECORDING:
            rec.finalize(final_output=res)
        return res


def run_lg_refund_agent_from_step(
    rec: Any,
    start_step: int,
    restored_state: dict[str, Any],
    overrides: Any = None,
    *,
    stale_policy: bool = False,
    policy_days_override: int | None = None,
) -> dict[str, Any]:
    """Execute LangGraph refund agent forward starting from `start_step` (for fork)."""
    customer_id = (
        restored_state.get("customer_id")
        or (restored_state.get("customer") or {}).get("customer_id")
        or ""
    )
    order_id = (
        restored_state.get("order_id")
        or (restored_state.get("target_order") or {}).get("order_id")
        or (restored_state.get("order") or {}).get("order_id")
        or ""
    )
    return run_lg_refund_agent(
        customer_id=customer_id,
        order_id=order_id,
        stale_policy=stale_policy,
        policy_days_override=policy_days_override,
        overrides=overrides,
        start_step=start_step,
        restored_state=restored_state,
        rec=rec,
    )


# Register lg_refund_agent with AgentReplay
from agentreplay.registry import register_agent  # noqa: E402

register_agent(
    name="lg_refund_agent",
    runner=run_lg_refund_agent,
    runner_from_step=run_lg_refund_agent_from_step,
    patch_targets=[
        "demo_agent.tools.get_customer",
        "demo_agent.lg_agent.get_customer",
        "demo_agent.tools.get_orders",
        "demo_agent.lg_agent.get_orders",
        "demo_agent.tools.read_refund_policy",
        "demo_agent.lg_agent.read_refund_policy",
        "demo_agent.tools.issue_refund",
        "demo_agent.lg_agent.issue_refund",
        "demo_agent.llm.get_llm",
        "demo_agent.lg_agent.get_llm",
    ],
)
