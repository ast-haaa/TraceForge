# demo_agent/variants.py
"""Real agent variant callables for regression testing and validation.

Used by AGENTREPLAY_AGENT fixture in generated tests and validate_test():
- buggy_agent: Runs refund agent with the stale 30-day policy (approves out-of-window).
- fixed_agent: Runs refund agent with corrected 7-day policy (denies out-of-window).
- always_deny_agent: Over-blocking agent that goes through the real tools and ledger
  but unconditionally denies refunds (catches false-positive passes).
"""
from __future__ import annotations

from typing import Any

from demo_agent.agent import run_refund_agent


def buggy_agent(customer_id: str, order_id: str) -> dict[str, Any]:
    """Agent variant running with the stale 30-day refund policy."""
    return run_refund_agent(customer_id, order_id, stale_policy=True)


def fixed_agent(customer_id: str, order_id: str) -> dict[str, Any]:
    """Agent variant running with the corrected 7-day refund policy."""
    return run_refund_agent(customer_id, order_id, stale_policy=False)


def always_deny_agent(customer_id: str, order_id: str) -> dict[str, Any]:
    """Agent variant that goes through the tools and ledger but always returns deny."""
    return run_refund_agent(customer_id, order_id, policy_days_override=-1)
