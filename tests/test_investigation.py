# tests/test_investigation.py
"""Offline unit tests for fault injection, oracle, LLM investigator, and evaluation metrics.

CI MUST NOT run the live LLM benchmark. All tests in this file run offline
with LLM_PROVIDER=stub or in-memory SQLite fixtures.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event

from agentreplay.db import SessionLocal
from agentreplay.investigate import (
    build_investigator_prompt,
    investigate,
)
from agentreplay.models import EvalLabel, Event, EventKind, Step, Trace
from demo_agent.agent import run_refund_agent
from evals.fault_injection import (
    assert_ground_truth_isolation,
    deterministic_oracle,
    generate_benchmark,
)
from evals.run_eval import run_evaluation, wilson_score_interval

# ---------------------------------------------------------------------------
# 1. Oracle Correctness Tests
# ---------------------------------------------------------------------------


def test_oracle_out_of_window_approved_is_failing():
    """An order older than 7 days that gets approved must be flagged as failing."""
    agent_output = {
        "result": "approve",
        "customer": "Alice Smith",
        "order_id": "O100",
        "refund": {"status": "refund_issued"},
    }
    is_wrong, reason = deterministic_oracle(
        true_customer_name="Alice Smith",
        true_days_old=12,  # > 7 days
        agent_output=agent_output,
    )
    assert is_wrong is True
    assert "Expected deny" in reason


def test_oracle_out_of_window_denied_is_correct():
    """An order older than 7 days that gets denied must be flagged as correct."""
    agent_output = {
        "result": "deny",
        "customer": "Alice Smith",
        "order_id": "O100",
        "refund": {"status": "skipped"},
    }
    is_wrong, reason = deterministic_oracle(
        true_customer_name="Alice Smith",
        true_days_old=14,  # > 7 days
        agent_output=agent_output,
    )
    assert is_wrong is False
    assert "Correctly denied" in reason


def test_oracle_in_window_approved_right_customer_is_correct():
    """An order <= 7 days old approved to the right customer must be correct."""
    agent_output = {
        "result": "approve",
        "customer": "Sophia Bennett",
        "order_id": "O100",
        "refund": {"status": "refund_issued"},
    }
    is_wrong, reason = deterministic_oracle(
        true_customer_name="Sophia Bennett",
        true_days_old=3,  # <= 7 days
        agent_output=agent_output,
    )
    assert is_wrong is False
    assert "Correctly approved" in reason


def test_oracle_in_window_denied_is_failing():
    """An order <= 7 days old that gets denied must be flagged as failing."""
    agent_output = {
        "result": "deny",
        "customer": "Sophia Bennett",
        "order_id": "O100",
        "refund": {"status": "skipped"},
    }
    is_wrong, reason = deterministic_oracle(
        true_customer_name="Sophia Bennett",
        true_days_old=3,
        agent_output=agent_output,
    )
    assert is_wrong is True
    assert "Expected approve" in reason


def test_oracle_wrong_customer_is_failing():
    """An approved order credited to the wrong customer must be flagged as failing."""
    agent_output = {
        "result": "approve",
        "customer": "Liam Murphy",  # Wrong customer!
        "order_id": "O100",
        "refund": {"status": "refund_issued"},
    }
    is_wrong, reason = deterministic_oracle(
        true_customer_name="Sophia Bennett",
        true_days_old=3,
        agent_output=agent_output,
    )
    assert is_wrong is True
    assert "wrong customer" in reason.lower()


# ---------------------------------------------------------------------------
# 2. Wilson Score Confidence Interval Math Tests
# ---------------------------------------------------------------------------


def test_wilson_score_interval_properties():
    """Verify mathematical properties of the Wilson 95% score interval."""
    # Zero successes
    low, high = wilson_score_interval(0, 40)
    assert low == 0.0
    assert 0.0 < high < 0.15

    # 100% successes
    low, high = wilson_score_interval(40, 40)
    assert 0.85 < low < 1.0
    assert high == 1.0

    # 50% successes
    low, high = wilson_score_interval(20, 40)
    assert 0.30 < low < 0.50
    assert 0.50 < high < 0.70
    assert low < 0.5 < high

    # Empty total
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 3. Leak Prevention Tests
# ---------------------------------------------------------------------------


def test_canary_leak_prevention():
    """Canary strings in Trace.description and Trace.meta must NOT appear in investigator prompt."""
    canary_desc = "CANARY_IN_DESCRIPTION_SECRET_XYZ99"
    canary_meta = "CANARY_IN_META_SECRET_ABC11"

    st = Step(
        index=1,
        node_name="get_customer",
        state_before={"customer_id": "C001"},
        state_after={"customer": {"name": "Alice"}},
    )
    st.id = 101

    ev = Event(
        step_id=101,
        kind=EventKind.TOOL_CALL,
        seq=1,
        input_hash="hash123",
        raw_input="{'name': 'get_customer'}",
        output={"customer": "Alice"},
    )
    ev.id = 201

    prompt = build_investigator_prompt([st], {st.id: [ev]})

    assert (
        canary_desc not in prompt
    ), "Leak detected: canary description was found in investigator prompt!"
    assert (
        canary_meta not in prompt
    ), "Leak detected: canary meta was found in investigator prompt!"


def test_investigate_ast_has_no_eval_label():
    """Static AST check: investigate.py must NEVER import or reference EvalLabel."""
    investigate_path = Path("agentreplay/investigate.py")
    assert investigate_path.exists()

    tree = ast.parse(investigate_path.read_text(encoding="utf-8"))

    eval_label_found = False
    found_node_details: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "EvalLabel":
            eval_label_found = True
            found_node_details.append(f"Name 'EvalLabel' at line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr == "EvalLabel":
            eval_label_found = True
            found_node_details.append(f"Attribute 'EvalLabel' at line {node.lineno}")
        elif isinstance(node, ast.alias) and (
            node.name == "EvalLabel" or node.asname == "EvalLabel"
        ):
            eval_label_found = True
            found_node_details.append(f"Import alias 'EvalLabel' at line {node.lineno}")

    assert (
        not eval_label_found
    ), f"Static check failed! EvalLabel referenced in investigate.py: {found_node_details}"


def test_runtime_db_guard_eval_label_not_queried():
    """Runtime DB check: queries during investigate() must NEVER touch eval_labels."""
    from agentreplay.db import engine

    session = SessionLocal()
    try:
        run_refund_agent("C001", "O100", run_id="runtime-guard-test", meta={})
        tr = session.query(Trace).filter(Trace.run_id == "runtime-guard-test").first()
        assert tr is not None

        table_accessed: list[str] = []

        def before_cursor_execute(
            conn, cursor, statement, parameters, context, executemany
        ):
            if "eval_labels" in statement.lower():
                table_accessed.append(statement)
                raise AssertionError(
                    f"ILLEGAL ACCESS: Query touched eval_labels during investigate(): {statement}"
                )

        event.listen(engine, "before_cursor_execute", before_cursor_execute)

        try:
            result = investigate(tr.id, session=session)
            assert len(result.ranked_steps) > 0
            assert (
                len(table_accessed) == 0
            ), f"eval_labels was accessed during investigate: {table_accessed}"

            # Verify our listener DOES catch actual queries to eval_labels
            with pytest.raises(AssertionError, match="ILLEGAL ACCESS"):
                session.query(EvalLabel).all()

        finally:
            event.remove(engine, "before_cursor_execute", before_cursor_execute)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 4. Offline Dataset Builder & Resumability Tests
# ---------------------------------------------------------------------------


def test_dataset_builder_offline_resumability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that dataset generator creates records and skips already-completed runs."""
    import evals.fault_injection as fi_mod

    test_dataset_path = tmp_path / "dataset.json"
    monkeypatch.setattr(fi_mod, "DATASET_PATH", test_dataset_path)

    # Run with target_per_fault=1 in dry_run/stub mode
    summary1 = generate_benchmark(target_per_fault=1, dry_run=True)
    assert summary1["total_dataset_entries"] == 5  # 5 fault types * 1 run each

    # Verify dataset.json was created and has 5 entries
    assert test_dataset_path.exists()
    with open(test_dataset_path, encoding="utf-8") as f:
        data1 = json.load(f)
    assert len(data1) == 5

    # Run again with target_per_fault=1 -> should skip all 5 runs
    summary2 = generate_benchmark(target_per_fault=1, dry_run=True)
    assert summary2["total_dataset_entries"] == 5

    # Run isolation assertion on generated records
    with SessionLocal() as db:
        assert_ground_truth_isolation(db)


# ---------------------------------------------------------------------------
# 5. Offline Evaluation & Results Markdown Tests
# ---------------------------------------------------------------------------


def test_evaluation_runner_offline(tmp_path: Path):
    """Test evals/run_eval.py generates results.md and parses metrics offline."""
    # Create a small 5-run dataset
    dataset_file = tmp_path / "test_dataset.json"
    results_file = tmp_path / "test_results.md"

    # Generate 1 run per fault using stub
    with SessionLocal() as db:
        # Create 5 test traces
        items: list[dict[str, Any]] = []
        for idx, (ft, (gt_step, gt_node)) in enumerate(
            [
                ("stale_policy", (3, "read_refund_policy")),
                ("wrong_order_date", (2, "get_orders")),
                ("prompt_bug", (4, "llm_decide")),
                ("missing_tool_result", (3, "read_refund_policy")),
                ("wrong_customer_data", (1, "get_customer")),
            ]
        ):
            run_id = f"eval-test-{ft}-{idx}"
            run_refund_agent("C001", "O100", run_id=run_id, meta={}, description="Test eval run")
            tr = db.query(Trace).filter(Trace.run_id == run_id).first()
            assert tr is not None

            # Add EvalLabel
            el = EvalLabel(
                trace_id=tr.id,
                injected_fault=ft,
                ground_truth_step=gt_step,
                ground_truth_node=gt_node,
            )
            db.add(el)
            db.commit()

            items.append(
                {
                    "trace_id": tr.id,
                    "run_id": run_id,
                    "fault_type": ft,
                    "ground_truth_step": gt_step,
                    "ground_truth_node": gt_node,
                    "customer_id": "C001",
                    "order_id": "O100",
                }
            )

    with open(dataset_file, "w", encoding="utf-8") as f:
        json.dump(items, f)

    # Run evaluation
    eval_res = run_evaluation(dataset_path=dataset_file, results_path=results_file)

    assert eval_res["total_runs"] == 5
    assert results_file.exists()
    content = results_file.read_text(encoding="utf-8")
    assert "Summary Comparison" in content
    assert "Per-Fault Breakdown" in content
    assert "Where It Fails" in content
