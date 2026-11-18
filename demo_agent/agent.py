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

from agentreplay.models import EventKind, EventSource
from agentreplay.sdk import record_trace

from .tools import get_customer, get_orders, issue_refund, read_refund_policy

# ---------------------------------------------------------------------------
# Pure-Python LLM stub  (no API key needed for the demo)
# ---------------------------------------------------------------------------


def _stub_llm(prompt: str, *, policy_days: int | None = None) -> str:
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

    if policy_days is None:
        import re
        policy_match = re.search(r"within (\d+) days", prompt, re.IGNORECASE)
        if policy_match:
            policy_days = int(policy_match.group(1))
        else:
            policy_days = 7

    if days_old <= policy_days:
        return f"DECISION: approve  REASON: order is {days_old} day(s) old, within the {policy_days}-day window"
    else:
        return f"DECISION: deny  REASON: order is {days_old} day(s) old, outside the {policy_days}-day window"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers for Step Overrides
# ---------------------------------------------------------------------------


def _extract_policy_override(overrides: Any) -> str | None:
    """Extract policy string from various override formats."""
    if overrides is None:
        return None
    if isinstance(overrides, str):
        return overrides
    if isinstance(overrides, dict):
        if "policy" in overrides:
            return overrides["policy"]
        if "tool_output" in overrides:
            to = overrides["tool_output"]
            if isinstance(to, dict) and "policy" in to:
                return to["policy"]
            return str(to)
        if "read_refund_policy" in overrides:
            rrp = overrides["read_refund_policy"]
            if isinstance(rrp, dict) and "policy" in rrp:
                return rrp["policy"]
            return str(rrp)
    return None


def _extract_prompt_override(overrides: Any) -> str | None:
    """Extract prompt string from various override formats."""
    if overrides is None:
        return None
    if isinstance(overrides, dict):
        if "prompt" in overrides:
            return overrides["prompt"]
        if "llm_decide" in overrides:
            ld = overrides["llm_decide"]
            if isinstance(ld, dict) and "prompt" in ld:
                return ld["prompt"]
            return str(ld)
    return None


def _extract_model_override(overrides: Any) -> str | None:
    """Extract model string from various override formats."""
    if overrides is None:
        return None
    if isinstance(overrides, dict):
        if "model" in overrides:
            return overrides["model"]
    return None


# ---------------------------------------------------------------------------
# Agent Execution Functions
# ---------------------------------------------------------------------------


def run_refund_agent_from_step(
    rec: Any,
    start_step: int,
    restored_state: dict[str, Any],
    overrides: Any = None,
    *,
    stale_policy: bool = False,
    policy_days_override: int | None = None,
) -> dict[str, Any]:
    """Execute Refund Agent forward starting at `start_step` using `restored_state`.

    Steps before `start_step` are skipped (they were already replayed/copied).
    At `start_step`, overrides are applied and events marked MOCKED.
    Subsequent steps run live agent code with events marked LIVE.
    """
    customer: dict[str, Any] = dict(restored_state.get("customer") or {})
    orders: list[dict[str, Any]] = list(restored_state.get("orders") or [])
    order: dict[str, Any] | None = (
        restored_state.get("target_order") or restored_state.get("order")
    )
    customer_id: str = (
        restored_state.get("customer_id")
        or (customer.get("customer_id") if customer else "")
        or ""
    )
    order_id: str = (
        restored_state.get("order_id")
        or (order.get("order_id") if order else "")
        or ""
    )
    policy_text: str = str(restored_state.get("policy") or "")
    days_old: int = int(restored_state.get("days_old") or 0)
    decision: str = str(restored_state.get("decision") or "")

    # ----------------------------------------------------------------
    # Step 1: get_customer
    # ----------------------------------------------------------------
    if start_step <= 1:
        t0 = time.perf_counter()
        rec.start_step("get_customer", state_before={"customer_id": customer_id, "order_id": order_id})
        tool_input = {"name": "get_customer", "args": {"customer_id": customer_id}}

        override_cust = overrides.get("get_customer") if isinstance(overrides, dict) else None
        if start_step == 1 and override_cust is not None:
            customer = override_cust
            ev_source = EventSource.MOCKED
        else:
            customer = get_customer(customer_id)
            ev_source = EventSource.RECORDED if start_step == 1 else EventSource.LIVE

        latency = (time.perf_counter() - t0) * 1000
        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output=customer,
            source=ev_source,
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
    if start_step <= 2:
        t0 = time.perf_counter()
        rec.start_step("get_orders", state_before={"customer": customer})
        tool_input = {"name": "get_orders", "args": {"customer_id": customer_id}}

        override_orders = overrides.get("get_orders") if isinstance(overrides, dict) else None
        if start_step == 2 and override_orders is not None:
            orders = override_orders if isinstance(override_orders, list) else override_orders.get("orders", [])
            ev_source = EventSource.MOCKED
        else:
            orders = get_orders(customer_id)
            ev_source = EventSource.RECORDED if start_step == 1 else EventSource.LIVE

        order = next((o for o in orders if o["order_id"] == order_id), None)
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output={"orders": orders},
            source=ev_source,
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
    if start_step <= 3:
        t0 = time.perf_counter()
        rec.start_step("read_refund_policy", state_before={"target_order": order})
        tool_input = {"name": "read_refund_policy", "args": {"stale": stale_policy}}

        policy_override = _extract_policy_override(overrides) if start_step == 3 else None
        if policy_override is not None:
            policy_text = policy_override
            ev_source = EventSource.MOCKED
        else:
            policy_text = read_refund_policy(stale=stale_policy)
            ev_source = EventSource.RECORDED if start_step == 1 else EventSource.LIVE

        latency = (time.perf_counter() - t0) * 1000
        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output={"policy": policy_text},
            source=ev_source,
        )
        rec.end_step(
            state_after={"target_order": order, "policy": policy_text},
            latency_ms=latency,
        )

    # ----------------------------------------------------------------
    # Step 4: LLM decides approve/deny
    # ----------------------------------------------------------------
    if start_step <= 4:
        t0 = time.perf_counter()
        if order and "order_date" in order:
            order_date = datetime.date.fromisoformat(order["order_date"])
            days_old = (datetime.date.today() - order_date).days

        prompt_override = _extract_prompt_override(overrides) if start_step == 4 else None
        model_override = _extract_model_override(overrides) if start_step == 4 else None
        model_name = model_override or "stub-llm-v1"

        if prompt_override is not None:
            prompt = prompt_override
            ev_source = EventSource.MOCKED
        else:
            cust_name = customer.get("name", "Unknown") if customer else "Unknown"
            ord_amt = order.get("amount", 0.0) if order else 0.0
            prompt = (
                f"Customer: {cust_name}\n"
                f"Order: {order_id}  Amount: ${ord_amt}\n"
                f"DAYS_OLD: {days_old}\n"
                f"Policy: {policy_text}\n"
                f"Decide whether to approve or deny this refund request."
            )
            ev_source = EventSource.MOCKED if (start_step == 4 and overrides) else (
                EventSource.RECORDED if start_step == 1 else EventSource.LIVE
            )

        rec.start_step(
            "llm_decide",
            state_before={"target_order": order, "policy": policy_text, "days_old": days_old},
        )
        llm_response = _stub_llm(prompt, policy_days=policy_days_override)
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.LLM_CALL,
            raw_input={"prompt": prompt},
            output={"response": llm_response, "model": model_name},
            source=ev_source,
        )
        rec.end_step(
            state_after={"llm_decision": llm_response, "days_old": days_old},
            model=model_name,
            latency_ms=latency,
        )

        decision = "approve" if "DECISION: approve" in llm_response else "deny"

    # ----------------------------------------------------------------
    # Step 5: issue_refund (only if approved)
    # ----------------------------------------------------------------
    if start_step <= 5:
        t0 = time.perf_counter()
        rec.start_step("issue_refund", state_before={"decision": decision, "order": order})

        if decision == "approve":
            ord_amt = order.get("amount", 0.0) if order else 0.0
            tool_input = {
                "name": "issue_refund",
                "args": {"order_id": order_id, "amount": ord_amt, "reason": "approved by agent"},
            }
            refund_result = issue_refund(order_id, ord_amt, reason="approved by agent")
        else:
            tool_input = {"name": "issue_refund", "args": None}
            refund_result = {"status": "skipped", "reason": "refund denied by policy"}

        ev_source = EventSource.MOCKED if (start_step == 5 and overrides) else (
            EventSource.RECORDED if start_step == 1 else EventSource.LIVE
        )
        latency = (time.perf_counter() - t0) * 1000

        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input=tool_input,
            output=refund_result,
            source=ev_source,
        )
        rec.end_step(
            state_after={"refund_result": refund_result},
            latency_ms=latency,
        )

    cust_name = customer.get("name", "Unknown") if customer else "Unknown"
    ord_amt = order.get("amount", 0.0) if order else 0.0
    final = {
        "result": decision,
        "customer": cust_name,
        "order_id": order_id,
        "amount": ord_amt,
        "days_old": days_old,
        "refund": refund_result if "refund_result" in locals() else None,
    }
    rec.finalize(final_output=final)
    return final


def run_refund_agent(
    customer_id: str,
    order_id: str,
    *,
    run_id: str | None = None,
    description: str = "Refund Agent demo run",
    stale_policy: bool = False,
    policy_days_override: int | None = None,  # inject a wrong day limit
    start_step: int = 1,
    restored_state: dict[str, Any] | None = None,
    overrides: Any = None,
) -> dict[str, Any]:
    """Execute the Refund Agent and record the full trace.

    Returns the final output dict.
    """
    initial_state = dict(restored_state or {})
    initial_state["customer_id"] = customer_id
    initial_state["order_id"] = order_id

    with record_trace(
        run_id=run_id,
        agent_name="refund_agent",
        description=description,
        meta={"stale_policy": stale_policy, "policy_days_override": policy_days_override},
    ) as rec:
        return run_refund_agent_from_step(
            rec=rec,
            start_step=start_step,
            restored_state=initial_state,
            overrides=overrides,
            stale_policy=stale_policy,
            policy_days_override=policy_days_override,
        )
