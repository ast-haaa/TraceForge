# demo_agent/agent.py
"""LangGraph Refund Agent with AgentReplay SDK instrumentation.

The agent is intentionally simple so we can inject realistic bugs:
  - stale policy text   → agent approves out-of-window refunds
  - wrong order data    → wrong amount or date
  - prompt bug          → agent misreads the policy day limit

The agent uses a pure-Python "LLM" stub so the demo runs without an API key.
For real LLM calls, swap `_stub_llm()` for a langchain chat model.
"""
from __future__ import annotations

import datetime
import time
from typing import Any

from agentreplay.models import EventKind
from agentreplay.sdk import record_trace

from .tools import get_customer, get_orders, issue_refund, read_refund_policy

# ---------------------------------------------------------------------------
# Pure-Python LLM stub  (no API key needed for the demo)
# ---------------------------------------------------------------------------


def _stub_llm(prompt: str, *, policy_days: int = 7) -> str:
    """Simulate an LLM that decides whether to issue a refund.

    Real implementations would call an OpenAI-compatible endpoint here.
    The stub reads ``ORDER_DAYS_OLD`` injected into the prompt by the agent
    so we can test boundary conditions deterministically.
    """
    # Parse the number of days from the prompt (injected by the agent)
    days_old: int | None = None
    for line in prompt.splitlines():
        if "DAYS_OLD:" in line:
            try:
                days_old = int(line.split("DAYS_OLD:")[-1].strip())
            except ValueError:
                pass

    if days_old is None:
        return "DECISION: approve  REASON: unable to determine order age"

    if days_old <= policy_days:
        return f"DECISION: approve  REASON: order is {days_old} day(s) old, within the {policy_days}-day window"
    else:
        return f"DECISION: deny  REASON: order is {days_old} day(s) old, outside the {policy_days}-day window"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def run_refund_agent(
    customer_id: str,
    order_id: str,
    *,
    run_id: str | None = None,
    description: str = "Refund Agent demo run",
    stale_policy: bool = False,
    policy_days_override: int | None = None,  # inject a wrong day limit
) -> dict[str, Any]:
    """Execute the Refund Agent and record the full trace.

    Returns the final output dict.
    """
    with record_trace(
        run_id=run_id,
        agent_name="refund_agent",
        description=description,
        meta={"stale_policy": stale_policy, "policy_days_override": policy_days_override},
    ) as rec:

        # ----------------------------------------------------------------
        # Step 1: get_customer
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        rec.start_step("get_customer", state_before={"customer_id": customer_id, "order_id": order_id})

        tool_input = {"name": "get_customer", "args": {"customer_id": customer_id}}
        customer = get_customer(customer_id)
        tool_output = customer
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output=tool_output,
        )
        rec.end_step(
            state_after={"customer": customer},
            latency_ms=latency,
        )

        if "error" in customer:
            final = {"result": "error", "detail": customer["error"]}
            rec.finalize(final_output=final)
            return final

        # ----------------------------------------------------------------
        # Step 2: get_orders
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        rec.start_step("get_orders", state_before={"customer": customer})

        tool_input = {"name": "get_orders", "args": {"customer_id": customer_id}}
        orders = get_orders(customer_id)
        # find the specific order
        order = next((o for o in orders if o["order_id"] == order_id), None)
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output={"orders": orders},
        )
        rec.end_step(
            state_after={"customer": customer, "orders": orders, "target_order": order},
            latency_ms=latency,
        )

        if order is None:
            final = {"result": "error", "detail": f"Order {order_id} not found for customer {customer_id}"}
            rec.finalize(final_output=final)
            return final

        # ----------------------------------------------------------------
        # Step 3: read_refund_policy
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        rec.start_step("read_refund_policy", state_before={"target_order": order})

        tool_input = {"name": "read_refund_policy", "args": {"stale": stale_policy}}
        policy_text = read_refund_policy(stale=stale_policy)
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output={"policy": policy_text},
        )
        rec.end_step(
            state_after={"target_order": order, "policy": policy_text},
            latency_ms=latency,
        )

        # ----------------------------------------------------------------
        # Step 4: LLM decides approve/deny
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        order_date = datetime.date.fromisoformat(order["order_date"])
        days_old = (datetime.date.today() - order_date).days

        # policy_days_override simulates an LLM that misread the day limit
        policy_days = policy_days_override if policy_days_override is not None else (
            30 if stale_policy else 7
        )

        prompt = (
            f"Customer: {customer['name']}\n"
            f"Order: {order_id}  Amount: ${order['amount']}\n"
            f"DAYS_OLD: {days_old}\n"
            f"Policy: {policy_text}\n"
            f"Decide whether to approve or deny this refund request."
        )

        rec.start_step("llm_decide", state_before={"target_order": order, "policy": policy_text, "days_old": days_old})
        llm_response = _stub_llm(prompt, policy_days=policy_days)
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.LLM_CALL,
            raw_input={"prompt": prompt},
            output={"response": llm_response, "model": "stub-llm-v1"},
        )
        rec.end_step(
            state_after={"llm_decision": llm_response, "days_old": days_old},
            model="stub-llm-v1",
            latency_ms=latency,
        )

        decision = "approve" if "DECISION: approve" in llm_response else "deny"

        # ----------------------------------------------------------------
        # Step 5: issue_refund (only if approved)
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        rec.start_step("issue_refund", state_before={"decision": decision, "order": order})

        if decision == "approve":
            tool_input = {
                "name": "issue_refund",
                "args": {"order_id": order_id, "amount": order["amount"], "reason": "approved by agent"},
            }
            refund_result = issue_refund(order_id, order["amount"], reason="approved by agent")
        else:
            tool_input = {"name": "issue_refund", "args": None}
            refund_result = {"status": "skipped", "reason": "refund denied by policy"}

        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output=refund_result,
        )
        rec.end_step(
            state_after={"refund_result": refund_result},
            latency_ms=latency,
        )

        final = {
            "result": decision,
            "customer": customer["name"],
            "order_id": order_id,
            "amount": order["amount"],
            "days_old": days_old,
            "refund": refund_result,
        }
        rec.finalize(final_output=final)
        return final
