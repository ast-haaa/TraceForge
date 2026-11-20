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


# ---------------------------------------------------------------------------
# 6. Phase 5 Closeout Invariant & Deterministic Diff Baseline Tests
# ---------------------------------------------------------------------------


def test_stub_investigate_unreachable_when_provider_not_stub(
    monkeypatch: pytest.MonkeyPatch,
):
    """Confirm _stub_investigate is unreachable when LLM_PROVIDER is not 'stub'."""
    from agentreplay.db import SessionLocal
    from agentreplay.investigate import investigate

    session = SessionLocal()
    try:
        run_refund_agent("C001", "O100", run_id="reachability-test", meta={})
        tr = session.query(Trace).filter(Trace.run_id == "reachability-test").first()
        assert tr is not None

        # When set to unsupported provider, must raise ValueError, NEVER fall back to stub
        monkeypatch.setenv("LLM_PROVIDER", "unsupported_provider")
        with pytest.raises(
            ValueError, match="Unsupported or unconfigured LLM_PROVIDER"
        ):
            investigate(tr.id, session=session)

        # When set to openai_compatible without API key, must raise ValueError from _live_investigate, NOT call stub
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("LLM_API_KEY", "")
        with pytest.raises(ValueError, match="LLM_API_KEY is not configured"):
            investigate(tr.id, session=session)
    finally:
        session.close()


def test_investigator_prompt_contains_no_leak_words():
    """Assert the investigator prompt contains none of: 30 days, stale, empty, null, outdated, corrupt."""
    st = Step(index=1, node_name="get_customer")
    st.id = 1
    ev = Event(
        step_id=1,
        seq=1,
        kind=EventKind.TOOL_CALL,
        input_hash="h1",
        raw_input="{}",
        output={"customer": "Alice"},
    )
    ev.id = 1
    prompt = build_investigator_prompt([st], {1: [ev]})

    forbidden = ["30 days", "stale", "empty", "null", "outdated", "corrupt"]
    for word in forbidden:
        assert (
            word not in prompt.lower()
        ), f"Leak detected: '{word}' found in investigator prompt!"


def test_deterministic_diff_baseline_hand_built_traces():
    """Verify deterministic diff accurately locates the first divergent step on hand-built traces."""
    from agentreplay.db import SessionLocal
    from agentreplay.diff import diff_traces

    session = SessionLocal()
    try:
        # Create clean baseline trace
        clean_tr = Trace(run_id="handbuilt-clean", agent_name="refund_agent")
        session.add(clean_tr)
        session.flush()

        # Step 1: get_customer
        s1 = Step(trace_id=clean_tr.id, index=1, node_name="get_customer")
        session.add(s1)
        session.flush()
        e1 = Event(
            step_id=s1.id,
            seq=1,
            kind=EventKind.TOOL_CALL,
            input_hash="h_c1",
            output={"customer_id": "C001", "name": "Alice"},
        )
        session.add(e1)

        # Step 2: get_orders
        s2 = Step(trace_id=clean_tr.id, index=2, node_name="get_orders")
        session.add(s2)
        session.flush()
        e2 = Event(
            step_id=s2.id,
            seq=1,
            kind=EventKind.TOOL_CALL,
            input_hash="h_o1",
            output={
                "orders": [{"order_id": "O1", "order_date": "2026-11-01"}]
            },
        )
        session.add(e2)

        # Step 3: read_refund_policy
        s3 = Step(trace_id=clean_tr.id, index=3, node_name="read_refund_policy")
        session.add(s3)
        session.flush()
        e3 = Event(
            step_id=s3.id,
            seq=1,
            kind=EventKind.TOOL_CALL,
            input_hash="h_p1",
            output={"policy": "within 7 days"},
        )
        session.add(e3)

        # Step 4: llm_decide
        s4 = Step(trace_id=clean_tr.id, index=4, node_name="llm_decide")
        session.add(s4)
        session.flush()
        e4 = Event(
            step_id=s4.id,
            seq=1,
            kind=EventKind.LLM_CALL,
            input_hash="h_prompt_7d",
            output={"decision": "deny"},
        )
        session.add(e4)

        # Step 5: issue_refund
        s5 = Step(trace_id=clean_tr.id, index=5, node_name="issue_refund")
        session.add(s5)
        session.flush()
        e5 = Event(
            step_id=s5.id,
            seq=1,
            kind=EventKind.TOOL_CALL,
            input_hash="h_ref1",
            output={"status": "skipped"},
        )
        session.add(e5)

        session.commit()

        # Helper to build a comparison trace
        def make_buggy_trace(
            run_id, diff_step, diff_output=None, diff_hash=None, extra_event=False
        ):
            tr = Trace(run_id=run_id, agent_name="refund_agent")
            session.add(tr)
            session.flush()
            for idx, node, out, ih in [
                (1, "get_customer", {"customer_id": "C001", "name": "Alice"}, "h_c1"),
                (
                    2,
                    "get_orders",
                    {"orders": [{"order_id": "O1", "order_date": "2026-11-01"}]},
                    "h_o1",
                ),
                (3, "read_refund_policy", {"policy": "within 7 days"}, "h_p1"),
                (4, "llm_decide", {"decision": "deny"}, "h_prompt_7d"),
                (5, "issue_refund", {"status": "skipped"}, "h_ref1"),
            ]:
                st = Step(trace_id=tr.id, index=idx, node_name=node)
                session.add(st)
                session.flush()
                kind = EventKind.LLM_CALL if idx == 4 else EventKind.TOOL_CALL
                eff_out = (
                    diff_output
                    if idx == diff_step and diff_output is not None
                    else out
                )
                eff_hash = (
                    diff_hash if idx == diff_step and diff_hash is not None else ih
                )
                ev = Event(
                    step_id=st.id,
                    seq=1,
                    kind=kind,
                    input_hash=eff_hash,
                    output=eff_out,
                )
                session.add(ev)
                if extra_event and idx == diff_step:
                    ev2 = Event(
                        step_id=st.id,
                        seq=2,
                        kind=kind,
                        input_hash=eff_hash,
                        output=eff_out,
                    )
                    session.add(ev2)
            session.commit()
            return tr.id

        # 1. wrong_customer_data -> step 1
        t_wcd = make_buggy_trace(
            "t_wcd", 1, diff_output={"customer_id": "C002", "name": "Bob"}
        )
        assert diff_traces(clean_tr.id, t_wcd, db=session).first_divergent_step == 1

        # 2. wrong_order_date -> step 2
        t_wod = make_buggy_trace(
            "t_wod",
            2,
            diff_output={
                "orders": [{"order_id": "O1", "order_date": "2026-11-17"}]
            },
        )
        assert diff_traces(clean_tr.id, t_wod, db=session).first_divergent_step == 2

        # 3. stale_policy -> step 3
        t_sp = make_buggy_trace("t_sp", 3, diff_output={"policy": "within 30 days"})
        assert diff_traces(clean_tr.id, t_sp, db=session).first_divergent_step == 3

        # 4. missing_tool_result -> step 3
        t_mtr = make_buggy_trace("t_mtr", 3, diff_output={"policy": ""})
        assert diff_traces(clean_tr.id, t_mtr, db=session).first_divergent_step == 3

        # 5. prompt_bug -> step 4
        t_pb = make_buggy_trace("t_pb", 4, diff_hash="h_prompt_30d_buggy")
        assert diff_traces(clean_tr.id, t_pb, db=session).first_divergent_step == 4

        # 6. duplicate_refund -> step 5
        t_dr = make_buggy_trace("t_dr", 5, extra_event=True)
        assert diff_traces(clean_tr.id, t_dr, db=session).first_divergent_step == 5

        # 7. wrong_amount -> step 5
        t_wa = make_buggy_trace(
            "t_wa", 5, diff_output={"status": "refund_issued", "amount": 999.0}
        )
        assert diff_traces(clean_tr.id, t_wa, db=session).first_divergent_step == 5
    finally:
        session.close()
