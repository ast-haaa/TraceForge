# evals/fault_injection.py
"""Fault-injection benchmark dataset generator for AgentReplay (Phase 5).

Generates 40 failing runs of the refund agent across 5 fault types (8 runs each):
1. stale_policy        -> ground truth: read_refund_policy (step 3)
2. wrong_order_date    -> ground truth: get_orders (step 2)
3. prompt_bug          -> ground truth: llm_decide (step 4)
4. missing_tool_result -> ground truth: read_refund_policy (step 3)
5. wrong_customer_data -> ground truth: get_customer (step 1)

GROUND TRUTH ISOLATION:
- Ground truth step and fault type live ONLY in EvalLabel and evals/dataset.json.
- Trace.description, Trace.meta, Trace.agent_version, Step, and Event rows contain
  NO mention of the fault names or ground truth.
- Resumable: saves incrementally to SQLite & evals/dataset.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import dotenv

import demo_agent.tools as tools_mod
from agentreplay.db import SessionLocal, init_db
from agentreplay.models import EvalLabel, Event, Step, Trace
from demo_agent.agent import run_refund_agent
from demo_agent.llm import clean_env_var

# Load .env for LLM credentials if available
dotenv.load_dotenv()

logger = logging.getLogger("evals.fault_injection")
DATASET_PATH = Path("evals/dataset.json")
HELD_OUT_PATH = Path("evals/held_out_dataset.json")

FAULT_TYPES = [
    "stale_policy",
    "wrong_order_date",
    "prompt_bug",
    "missing_tool_result",
    "wrong_customer_data",
]

HELD_OUT_FAULT_TYPES = [
    "duplicate_refund",
    "wrong_amount",
]

FAULT_GROUND_TRUTH: dict[str, tuple[int, str]] = {
    "stale_policy": (3, "read_refund_policy"),
    "wrong_order_date": (2, "get_orders"),
    "prompt_bug": (4, "llm_decide"),
    "missing_tool_result": (3, "read_refund_policy"),
    "wrong_customer_data": (1, "get_customer"),
    "duplicate_refund": (5, "issue_refund"),
    "wrong_amount": (5, "issue_refund"),
}

# Fixed synthetic customers and orders pool for repeatable benchmark runs
BENCHMARK_CUSTOMERS = [
    {"customer_id": "C201", "name": "Sophia Bennett", "email": "sophia@example.com"},
    {"customer_id": "C202", "name": "Liam Murphy", "email": "liam@example.com"},
    {"customer_id": "C203", "name": "Olivia Zhang", "email": "olivia@example.com"},
    {"customer_id": "C204", "name": "Noah Patel", "email": "noah@example.com"},
    {"customer_id": "C205", "name": "Ava Rodriguez", "email": "ava@example.com"},
    {"customer_id": "C206", "name": "William Chen", "email": "william@example.com"},
    {"customer_id": "C207", "name": "Isabella Kim", "email": "isabella@example.com"},
    {"customer_id": "C208", "name": "James Wilson", "email": "james@example.com"},
]

AMOUNTS = [29.99, 49.50, 68.00, 89.95, 119.00, 145.50, 180.00, 225.00]


def register_benchmark_data() -> None:
    """Register benchmark customers in demo_agent.tools.CUSTOMERS."""
    for c in BENCHMARK_CUSTOMERS:
        tools_mod.CUSTOMERS[c["customer_id"]] = c


def deterministic_oracle(
    true_customer_name: str,
    true_days_old: int,
    agent_output: dict[str, Any],
    true_amount: float | None = None,
) -> tuple[bool, str]:
    """Verify whether the agent's outcome is WRONG vs true 7-day policy.

    True policy:
    - true_days_old <= 7 -> outcome MUST be approve, refund status refund_issued, credited to true_customer_name, exact amount
    - true_days_old > 7  -> outcome MUST be deny, refund status skipped (no refund)

    Returns:
    - (is_wrong: bool, reason: str)
    """
    decision = agent_output.get("result")
    customer = agent_output.get("customer")
    refund = agent_output.get("refund") or {}
    refund_status = refund.get("status")

    if true_days_old <= 7:
        if decision != "approve":
            return True, f"Expected approve for order {true_days_old}d old, got {decision}"
        if customer != true_customer_name:
            return True, f"Refund credited to wrong customer {customer!r} (expected {true_customer_name!r})"
        if refund_status != "refund_issued":
            return True, f"Expected refund_issued, got {refund_status}"
        if refund.get("duplicate_issued"):
            return True, "Duplicate refund was issued for the order"
        if true_amount is not None and refund.get("amount") is not None:
            if abs(float(refund["amount"]) - float(true_amount)) > 0.01:
                return True, f"Refund amount ${refund['amount']} differs from order amount ${true_amount}"
        return False, "Correctly approved"
    else:
        if decision != "deny":
            return True, f"Expected deny for order {true_days_old}d old (> 7d), got {decision}"
        if refund_status == "refund_issued":
            return True, f"Refund wrongly issued for out-of-policy order ({true_days_old}d old)"
        return False, "Correctly denied"


def load_dataset() -> list[dict[str, Any]]:
    """Load existing dataset records from dataset.json if it exists."""
    if DATASET_PATH.exists():
        try:
            with open(DATASET_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read %s: %s", DATASET_PATH, e)
    return []


def save_dataset(dataset: list[dict[str, Any]]) -> None:
    """Save dataset records to dataset.json safely."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            with open(DATASET_PATH, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
            break
        except (PermissionError, OSError):
            time.sleep(0.15)


def generate_benchmark(
    target_per_fault: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build or resume the 40-run benchmark dataset.

    Returns summary dictionary with counts and statistics.
    """
    init_db()
    register_benchmark_data()

    dataset = load_dataset()
    db = SessionLocal()

    # Track completed runs per fault type
    completed_counts: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}
    existing_trace_ids = set()

    for item in dataset:
        ft = item.get("fault_type")
        tid = item.get("trace_id")
        if ft in completed_counts and tid:
            # Verify trace exists in DB
            tr = db.query(Trace).filter(Trace.id == tid).first()
            if tr:
                completed_counts[ft] += 1
                existing_trace_ids.add(tid)

    non_failing_counts: dict[str, int] = {ft: 0 for ft in FAULT_TYPES}
    today = datetime.date.today()

    print("=" * 75)
    print("AgentReplay Phase 5: Fault-Injection Benchmark Dataset Generator")
    print("=" * 75)
    print(f"Target: {target_per_fault} failing runs per fault type ({len(FAULT_TYPES) * target_per_fault} total)")
    print(f"Current progress: {completed_counts}")
    print(f"Provider: {clean_env_var(os.environ.get('LLM_PROVIDER'), 'stub')}")
    print("-" * 75)

    for ft in FAULT_TYPES:
        needed = target_per_fault - completed_counts[ft]
        if needed <= 0:
            print(f"[{ft}] Already completed {completed_counts[ft]}/{target_per_fault} runs. Skipping.")
            continue

        gt_step, gt_node = FAULT_GROUND_TRUTH[ft]
        print(f"\n[{ft}] Generating {needed} more failing run(s) (ground truth: step {gt_step} {gt_node})...")

        iteration = 0
        while completed_counts[ft] < target_per_fault and iteration < 40:
            iteration += 1
            idx = completed_counts[ft]
            cust = BENCHMARK_CUSTOMERS[(idx + iteration - 1) % len(BENCHMARK_CUSTOMERS)]
            amt = AMOUNTS[(idx + iteration - 1) % len(AMOUNTS)]
            order_id = f"ORD_{ft[:3].upper()}_{idx + 1}_{iteration}"

            # Prepare fault injection parameters
            fault_args: dict[str, Any] = {"type": ft}

            if ft == "stale_policy":
                # Order > 7 days old, stale policy says 30 days
                days_old_opts = [9, 11, 13, 15, 17, 19, 22, 25]
                true_days_old = days_old_opts[(idx + iteration - 1) % len(days_old_opts)]
                ord_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
                tools_mod.ORDERS[order_id] = {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "amount": amt,
                    "order_date": ord_date,
                    "item": f"Item {order_id}",
                }

            elif ft == "wrong_order_date":
                # True order > 7 days old; tool returns date making it appear 2-4 days old
                days_old_opts = [10, 12, 14, 16, 18, 20, 23, 27]
                true_days_old = days_old_opts[(idx + iteration - 1) % len(days_old_opts)]
                real_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
                tools_mod.ORDERS[order_id] = {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "amount": amt,
                    "order_date": real_date,
                    "item": f"Item {order_id}",
                }
                fake_days = ((idx + iteration) % 3) + 2  # 2, 3, or 4 days
                fake_date = (today - datetime.timedelta(days=fake_days)).isoformat()
                fault_args["fake_orders"] = [
                    {
                        "order_id": order_id,
                        "customer_id": cust["customer_id"],
                        "amount": amt,
                        "order_date": fake_date,
                        "item": f"Item {order_id}",
                    }
                ]

            elif ft == "prompt_bug":
                # True order > 7 days old; Step 3 reads 7-day policy, but Step 4 prompt instructs 30 days
                days_old_opts = [8, 10, 12, 14, 16, 18, 21, 24]
                true_days_old = days_old_opts[(idx + iteration - 1) % len(days_old_opts)]
                ord_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
                tools_mod.ORDERS[order_id] = {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "amount": amt,
                    "order_date": ord_date,
                    "item": f"Item {order_id}",
                }
                fault_args["buggy_prompt_policy"] = (
                    "Refunds are accepted within 30 days of the order date. "
                    "No refunds will be issued for orders older than 30 days."
                )

            elif ft == "missing_tool_result":
                # read_refund_policy returns empty string
                # If true_days_old > 7 and LLM approves -> failing run
                # If true_days_old <= 7 and LLM denies -> failing run
                if iteration % 2 == 1:
                    out_days = [9, 11, 13, 15, 17, 19, 22, 25]
                    true_days_old = out_days[(idx + iteration) % len(out_days)]
                else:
                    in_days = [2, 3, 4, 5, 2, 3, 4, 5]
                    true_days_old = in_days[(idx + iteration) % len(in_days)]
                ord_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
                tools_mod.ORDERS[order_id] = {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "amount": amt,
                    "order_date": ord_date,
                    "item": f"Item {order_id}",
                }

            elif ft == "wrong_customer_data":
                # Order <= 7 days old, but get_customer returns another customer
                days_old_opts = [2, 3, 4, 5, 2, 3, 4, 5]
                true_days_old = days_old_opts[(idx + iteration - 1) % len(days_old_opts)]
                ord_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
                tools_mod.ORDERS[order_id] = {
                    "order_id": order_id,
                    "customer_id": cust["customer_id"],
                    "amount": amt,
                    "order_date": ord_date,
                    "item": f"Item {order_id}",
                }
                # Other customer
                other_cust = BENCHMARK_CUSTOMERS[(idx + iteration) % len(BENCHMARK_CUSTOMERS)]
                fault_args["fake_customer"] = other_cust

            # Execute agent run with STRICT GROUND TRUTH ISOLATION:
            # - description: generic benchmark execution
            # - meta: empty dict {} (no stale_policy or fault names!)
            run_id = f"bench-{ft}-{idx + 1}-{int(time.time() * 1000)}"
            try:
                agent_res = run_refund_agent(
                    customer_id=cust["customer_id"],
                    order_id=order_id,
                    run_id=run_id,
                    description=f"Benchmark execution #{idx + 1}",
                    meta={},  # STRICT ISOLATION: Empty metadata!
                    fault=fault_args,
                )
            except Exception as e:
                logger.error("Run %s raised exception: %s", run_id, e)
                time.sleep(2.0)
                continue

            # Query the newly recorded trace from DB
            tr = db.query(Trace).filter(Trace.run_id == run_id).first()
            if not tr:
                logger.error("Trace %s was not found in DB after execution.", run_id)
                continue

            # Verify outcome with deterministic oracle
            is_wrong, reason = deterministic_oracle(
                true_customer_name=cust["name"],
                true_days_old=true_days_old,
                agent_output=agent_res,
            )

            if not is_wrong:
                # The injected bug did NOT produce a wrong outcome
                non_failing_counts[ft] += 1
                print(f"  [Attempt {iteration}] Injected {ft} did NOT fail oracle: {reason}. Retrying with another configuration...")
                continue

            # It IS a confirmed failing run! Store EvalLabel & update dataset
            eval_lbl = EvalLabel(
                trace_id=tr.id,
                injected_fault=ft,
                ground_truth_step=gt_step,
                ground_truth_node=gt_node,
            )
            db.add(eval_lbl)
            db.commit()

            record = {
                "trace_id": tr.id,
                "run_id": run_id,
                "fault_type": ft,
                "ground_truth_step": gt_step,
                "ground_truth_node": gt_node,
                "customer_id": cust["customer_id"],
                "customer_name": cust["name"],
                "order_id": order_id,
                "true_days_old": true_days_old,
                "amount": amt,
                "actual_decision": agent_res.get("result"),
                "outcome_is_wrong": True,
                "oracle_failure_reason": reason,
            }
            dataset.append(record)
            save_dataset(dataset)

            completed_counts[ft] += 1
            print(
                f"  [{ft} {completed_counts[ft]}/{target_per_fault}] Trace ID {tr.id} saved: "
                f"days_old={true_days_old}, decision={agent_res.get('result')}, reason={reason}"
            )

            # Avoid rate limit spikes
            time.sleep(0.5)

    db.close()

    print("\n" + "=" * 75)
    print("Benchmark Dataset Generation Complete")
    print("=" * 75)
    print("Completed failing runs per fault type:")
    for ft, cnt in completed_counts.items():
        print(f"  - {ft:<22}: {cnt} runs")
    print("\nInjected runs that did NOT fail (per fault type):")
    for ft, cnt in non_failing_counts.items():
        print(f"  - {ft:<22}: {cnt} non-failing runs")
    print(f"Total dataset entries: {len(dataset)} saved to {DATASET_PATH}")
    print("=" * 75)

    return {
        "completed_counts": completed_counts,
        "non_failing_counts": non_failing_counts,
        "total_dataset_entries": len(dataset),
    }


def generate_held_out(
    target_per_fault: int = 8,
    seed: int = 1337,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build or resume the 16-run held-out dataset for duplicate_refund and wrong_amount."""
    init_db()
    register_benchmark_data()

    held_out_dataset = []
    if HELD_OUT_PATH.exists():
        try:
            with open(HELD_OUT_PATH, encoding="utf-8") as f:
                held_out_dataset = json.load(f)
        except Exception:
            held_out_dataset = []

    db = SessionLocal()
    completed_counts: dict[str, int] = {ft: 0 for ft in HELD_OUT_FAULT_TYPES}
    for item in held_out_dataset:
        ft = item.get("fault_type")
        tid = item.get("trace_id")
        if ft in completed_counts and tid:
            tr = db.query(Trace).filter(Trace.id == tid).first()
            if tr:
                completed_counts[ft] += 1

    non_failing_counts: dict[str, int] = {ft: 0 for ft in HELD_OUT_FAULT_TYPES}
    today = datetime.date.today()

    print("=" * 75)
    print("AgentReplay Phase 5: Held-Out Benchmark Dataset Generator")
    print("=" * 75)
    print(f"Target: {target_per_fault} runs per fault type ({len(HELD_OUT_FAULT_TYPES) * target_per_fault} total)")
    print(f"Current progress: {completed_counts}")
    print("-" * 75)

    for ft in HELD_OUT_FAULT_TYPES:
        needed = target_per_fault - completed_counts[ft]
        if needed <= 0:
            print(f"[{ft}] Already completed {completed_counts[ft]}/{target_per_fault} runs. Skipping.")
            continue

        gt_step, gt_node = FAULT_GROUND_TRUTH[ft]
        print(f"\n[{ft}] Generating {needed} more failing run(s) (ground truth: step {gt_step} {gt_node})...")

        iteration = 0
        while completed_counts[ft] < target_per_fault and iteration < 40:
            iteration += 1
            idx = completed_counts[ft]
            cust = BENCHMARK_CUSTOMERS[(idx + iteration - 1) % len(BENCHMARK_CUSTOMERS)]
            amt = AMOUNTS[(idx + iteration - 1) % len(AMOUNTS)]
            order_id = f"ORD_HELD_{ft[:3].upper()}_{idx + 1}_{iteration}"

            # Valid recent order (<= 7 days old) so refund is normally approved
            days_old_opts = [2, 3, 4, 5, 2, 3, 4, 5]
            true_days_old = days_old_opts[(idx + iteration - 1) % len(days_old_opts)]
            ord_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
            tools_mod.ORDERS[order_id] = {
                "order_id": order_id,
                "customer_id": cust["customer_id"],
                "amount": amt,
                "order_date": ord_date,
                "item": f"Item {order_id}",
            }

            fault_args: dict[str, Any] = {"type": ft}
            if ft == "wrong_amount":
                fault_args["fake_amount"] = amt + 50.0

            run_id = f"bench-held-{ft}-{idx + 1}-{int(time.time() * 1000)}"
            try:
                agent_res = run_refund_agent(
                    customer_id=cust["customer_id"],
                    order_id=order_id,
                    run_id=run_id,
                    description=f"Held-out benchmark execution #{idx + 1}",
                    meta={},
                    fault=fault_args,
                )
            except Exception as e:
                logger.error("Held-out run %s failed: %s", run_id, e)
                time.sleep(1.0)
                continue

            tr = db.query(Trace).filter(Trace.run_id == run_id).first()
            if not tr:
                continue

            is_wrong, reason = deterministic_oracle(
                true_customer_name=cust["name"],
                true_days_old=true_days_old,
                true_amount=amt,
                agent_output=agent_res,
            )

            if not is_wrong:
                non_failing_counts[ft] += 1
                continue

            eval_lbl = EvalLabel(
                trace_id=tr.id,
                injected_fault=ft,
                ground_truth_step=gt_step,
                ground_truth_node=gt_node,
            )
            db.add(eval_lbl)
            db.commit()

            record = {
                "trace_id": tr.id,
                "run_id": run_id,
                "fault_type": ft,
                "ground_truth_step": gt_step,
                "ground_truth_node": gt_node,
                "customer_id": cust["customer_id"],
                "customer_name": cust["name"],
                "order_id": order_id,
                "true_days_old": true_days_old,
                "amount": amt,
                "actual_decision": agent_res.get("result"),
                "outcome_is_wrong": True,
                "oracle_failure_reason": reason,
            }
            held_out_dataset.append(record)
            save_dataset_to(held_out_dataset, HELD_OUT_PATH)

            completed_counts[ft] += 1
            print(
                f"  [{ft} {completed_counts[ft]}/{target_per_fault}] Trace ID {tr.id} saved: "
                f"days_old={true_days_old}, amount={amt}, reason={reason}"
            )
            time.sleep(0.5)

    db.close()
    return {
        "completed_counts": completed_counts,
        "non_failing_counts": non_failing_counts,
        "total_dataset_entries": len(held_out_dataset),
    }


def save_dataset_to(dataset: list[dict[str, Any]], path: Path) -> None:
    """Save dataset records to a specific path safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
            break
        except (PermissionError, OSError):
            time.sleep(0.15)


def assert_ground_truth_isolation(db_session: Any) -> None:
    """Assert that NO benchmark trace or event or step contains fault names."""
    dataset = load_dataset()
    if HELD_OUT_PATH.exists():
        try:
            with open(HELD_OUT_PATH, encoding="utf-8") as f:
                dataset.extend(json.load(f))
        except Exception:
            pass

    if not dataset:
        return

    benchmark_trace_ids = [d["trace_id"] for d in dataset]
    traces = db_session.query(Trace).filter(Trace.id.in_(benchmark_trace_ids)).all()
    steps = db_session.query(Step).filter(Step.trace_id.in_(benchmark_trace_ids)).all()
    step_ids = [s.id for s in steps]
    events = db_session.query(Event).filter(Event.step_id.in_(step_ids)).all()

    forbidden_terms = [
        "stale_policy",
        "wrong_order_date",
        "prompt_bug",
        "missing_tool_result",
        "wrong_customer_data",
        "duplicate_refund",
        "wrong_amount",
    ]

    for tr in traces:
        for term in forbidden_terms:
            assert term not in (tr.description or "").lower(), f"Leak in Trace.description: {term}"
            meta_str = json.dumps(tr.meta or {})
            assert term not in meta_str.lower(), f"Leak in Trace.meta: {term}"
            assert term not in (tr.agent_version or "").lower(), f"Leak in Trace.agent_version: {term}"

    for st in steps:
        for term in forbidden_terms:
            assert term not in st.node_name.lower(), f"Leak in Step.node_name: {term}"

    for ev in events:
        for term in forbidden_terms:
            raw_in = (ev.raw_input or "").lower()
            raw_out = (ev.raw_output or "").lower()
            out_str = json.dumps(ev.output or {}).lower()
            assert term not in raw_in, f"Leak in Event.raw_input: {term}"
            assert term not in raw_out, f"Leak in Event.raw_output: {term}"
            assert term not in out_str, f"Leak in Event.output: {term}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fault-injection benchmark dataset.")
    parser.add_argument(
        "--runs-per-fault",
        type=int,
        default=8,
        help="Number of failing runs per fault type (default: 8)",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Generate held-out dataset instead of primary benchmark",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run with dry run / stub mode"
    )
    args = parser.parse_args()

    if args.dry_run:
        os.environ["LLM_PROVIDER"] = "stub"

    if args.held_out:
        res = generate_held_out(target_per_fault=args.runs_per_fault)
    else:
        res = generate_benchmark(target_per_fault=args.runs_per_fault)

    with SessionLocal() as s:
        assert_ground_truth_isolation(s)
        print("Ground truth isolation assertion PASSED on all benchmark records!")
