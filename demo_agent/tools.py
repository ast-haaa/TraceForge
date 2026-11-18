# demo_agent/tools.py
"""Mocked tools for the Refund Agent demo.

All tools are pure Python functions – no real network/database calls.
issue_refund writes to a module-level fake ledger so tests can inspect it.
"""
from __future__ import annotations

import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Fake in-memory ledger for issued refunds (reset between runs in tests)
# ---------------------------------------------------------------------------
_refund_ledger: list[dict[str, Any]] = []


def reset_ledger() -> None:
    """Call this in test setUp to get a clean ledger."""
    _refund_ledger.clear()


def get_refund_ledger() -> list[dict[str, Any]]:
    return list(_refund_ledger)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

CUSTOMERS = {
    "C001": {"customer_id": "C001", "name": "Alice Smith", "email": "alice@example.com"},
    "C002": {"customer_id": "C002", "name": "Bob Jones", "email": "bob@example.com"},
    "C003": {"customer_id": "C003", "name": "Carol White", "email": "carol@example.com"},
}

# order_date intentionally close to the boundary to trigger policy edge cases
_today = datetime.date.today()

ORDERS = {
    "O100": {
        "order_id": "O100",
        "customer_id": "C001",
        "amount": 49.99,
        "order_date": (_today - datetime.timedelta(days=3)).isoformat(),
        "item": "Widget Pro",
    },
    "O101": {
        "order_id": "O101",
        "customer_id": "C001",
        "amount": 199.00,
        "order_date": (_today - datetime.timedelta(days=10)).isoformat(),  # outside 7-day window
        "item": "Gadget Max",
    },
    "O102": {
        "order_id": "O102",
        "customer_id": "C002",
        "amount": 25.00,
        "order_date": (_today - datetime.timedelta(days=5)).isoformat(),
        "item": "Thingamajig",
    },
    "O103": {
        "order_id": "O103",
        "customer_id": "C001",
        "amount": 79.50,
        "order_date": (_today - datetime.timedelta(days=12)).isoformat(),  # 12 days old (> 7 days)
        "item": "Super Gizmo",
    },
}

REFUND_POLICY = (
    "Refunds are accepted within 7 days of the order date. "
    "No refunds will be issued for orders older than 7 days. "
    "Refunds are credited to the original payment method within 3-5 business days."
)

# Stale / buggy policy – used by fault-injection benchmark
STALE_REFUND_POLICY = (
    "Refunds are accepted within 30 days of the order date. "
    "No refunds will be issued for orders older than 30 days."
)


def get_customer(customer_id: str) -> dict[str, Any]:
    """Return customer record or an error dict."""
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        return {"error": f"Customer {customer_id!r} not found"}
    return customer


def get_orders(customer_id: str) -> list[dict[str, Any]]:
    """Return all orders for the given customer."""
    return [o for o in ORDERS.values() if o["customer_id"] == customer_id]


def read_refund_policy(*, stale: bool = False) -> str:
    """Return the current refund policy text.

    Pass ``stale=True`` in fault-injection scenarios to simulate a cached
    outdated policy.
    """
    return STALE_REFUND_POLICY if stale else REFUND_POLICY


def issue_refund(order_id: str, amount: float, reason: str = "") -> dict[str, Any]:
    """Pretend to issue a refund and record it in the fake ledger."""
    if order_id not in ORDERS:
        return {"error": f"Order {order_id!r} not found"}
    entry = {
        "order_id": order_id,
        "amount": amount,
        "reason": reason,
        "issued_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    _refund_ledger.append(entry)
    return {"status": "refund_issued", "order_id": order_id, "amount": amount}
