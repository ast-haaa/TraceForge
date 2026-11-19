# tests/test_llm_adapter.py
"""Tests for Phase 4.5: Real LLM Adapter, Determinism, Replay Safety, and Secret Isolation."""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
from unittest.mock import MagicMock, patch

import pytest

from agentreplay.fork import fork_trace
from agentreplay.models import Event, EventKind, EventSource, Step, Trace, TraceRunType, TraceStatus
from agentreplay.replay import ReplayEngine
from demo_agent.agent import run_refund_agent
from demo_agent.llm import openai_compatible_llm
from demo_agent.tools import (
    REFUND_POLICY,
    get_refund_ledger,
    get_sandbox_refund_ledger,
    reset_ledger,
)


@pytest.fixture(autouse=True)
def clean_test_ledger():
    reset_ledger()
    yield
    reset_ledger()


# ---------------------------------------------------------------------------
# Test 1: Prompt Volatility & Determinism (Requirement 3)
# ---------------------------------------------------------------------------


def test_prompt_contains_no_volatile_values(db_session, monkeypatch):
    """The LLM prompt must contain policy text and order fields (including days_old),
    but must NOT contain today's date or timestamps, ensuring replayability on later days.
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    res = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Prompt volatility test run",
    )
    assert res["result"] in ("approve", "deny")

    tr = (
        db_session.query(Trace)
        .filter_by(description="Prompt volatility test run")
        .one()
    )
    llm_step = next(s for s in tr.steps if s.node_name == "llm_decide")
    llm_ev = next(e for e in llm_step.events if e.kind == EventKind.LLM_CALL)

    raw = llm_ev.raw_input
    if isinstance(raw, str):
        raw = json.loads(raw)
    prompt = raw.get("prompt", "")
    assert prompt, "Prompt should be non-empty"

    # Must contain computed DAYS_OLD and Policy
    assert "DAYS_OLD: 12" in prompt
    assert "Policy:" in prompt
    assert "Customer: Alice Smith" in prompt
    assert "Order: O103" in prompt

    # Must NOT contain volatile today's date
    today_iso = datetime.date.today().isoformat()
    assert today_iso not in prompt, f"Today's date ({today_iso}) leaked into prompt!"

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    assert today_str not in prompt, f"Today's date ({today_str}) leaked into prompt!"

    # Must NOT contain ISO timestamps (YYYY-MM-DDTHH:MM:SS)
    iso_timestamp_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
    assert not iso_timestamp_pattern.search(prompt), "Found ISO timestamp inside LLM prompt!"

    # Must NOT contain UUIDs
    uuid_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
    assert not uuid_pattern.search(prompt), "Found UUID inside LLM prompt!"


# ---------------------------------------------------------------------------
# Test 2: Zero Leaks of API Key & Auth Header in DB (Requirement 4)
# ---------------------------------------------------------------------------


def test_no_api_key_or_auth_header_in_database(db_session, monkeypatch):
    """The API key and Authorization header must NEVER be stored in any database column."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    test_api_key = "gsk_SUPER_SECRET_KEY_1234567890_XYZ"
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", test_api_key)
    monkeypatch.setenv("LLM_MODEL", "mock-model-v1")

    # Mock httpx.post to return a valid OpenAI-compatible response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps({
        "id": "chatcmpl-mock",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"decision": "approve", "reason": "Within 30-day window"}),
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
    })
    mock_resp.json.return_value = json.loads(mock_resp.text)
    mock_resp.headers = {}

    with patch("httpx.post", return_value=mock_resp):
        res = run_refund_agent(
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
            description="Secret scan run",
        )
        assert res["result"] == "approve"

    # Scan EVERY table and EVERY column in the database
    for model_cls in (Trace, Step, Event):
        rows = db_session.query(model_cls).all()
        for r in rows:
            for col in r.__table__.columns:
                val = getattr(r, col.name)
                val_str = json.dumps(val, default=str) if isinstance(val, (dict, list)) else str(val)

                # Assert API key is NEVER present
                assert test_api_key not in val_str, (
                    f"CRITICAL: API key leaked into {model_cls.__name__}.{col.name}: {val_str}"
                )
                # Assert Authorization header is NEVER stored
                assert "Bearer " not in val_str, (
                    f"Bearer token pattern leaked into {model_cls.__name__}.{col.name}"
                )
                assert "Authorization" not in val_str, (
                    f"Authorization header leaked into {model_cls.__name__}.{col.name}"
                )


# ---------------------------------------------------------------------------
# Test 3: Pydantic Validation & Retry on Invalid Output (Requirement 2)
# ---------------------------------------------------------------------------


def test_pydantic_validation_and_retry_on_invalid_output(monkeypatch):
    """When the LLM returns invalid JSON or schema, it retries once.
    If retry succeeds, returns valid DecisionOutput; if retry fails, records event error.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "mock-key")

    # Scenario A: First call invalid, second call valid -> Succeeds
    resp_invalid = MagicMock()
    resp_invalid.status_code = 200
    resp_invalid.text = "NOT JSON OUTPUT"
    resp_invalid.json.return_value = {"choices": [{"message": {"content": "NOT JSON OUTPUT"}}]}
    resp_invalid.headers = {}

    resp_valid = MagicMock()
    resp_valid.status_code = 200
    resp_valid.text = json.dumps({"decision": "deny", "reason": "Order exceeds 7-day limit"})
    resp_valid.json.return_value = {
        "choices": [{"message": {"content": resp_valid.text}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 10},
    }
    resp_valid.headers = {}

    with patch("httpx.post", side_effect=[resp_invalid, resp_valid]) as mock_http:
        res = openai_compatible_llm("DAYS_OLD: 12\nPolicy: 7 days")
        assert res.decision == "deny"
        assert "Order exceeds 7-day limit" in res.reason
        assert res.error is None
        assert mock_http.call_count == 2

    # Scenario B: Both calls invalid -> returns error
    with patch("httpx.post", side_effect=[resp_invalid, resp_invalid]) as mock_http2:
        res2 = openai_compatible_llm("DAYS_OLD: 12\nPolicy: 7 days")
        assert res2.decision == "deny"
        assert res2.error is not None
        assert "Invalid output after retry" in res2.error
        assert mock_http2.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: Replay with Real LLM Makes ZERO Network Calls (Requirement 5)
# ---------------------------------------------------------------------------


def test_replay_makes_zero_network_calls(db_session, monkeypatch):
    """Deterministic replay of a real-LLM recorded trace makes zero HTTP network calls."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Record an execution with mock HTTP
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = json.dumps({"decision": "approve", "reason": "Approved by policy"})
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": mock_resp.text}}],
        "usage": {"prompt_tokens": 45, "completion_tokens": 12},
    }
    mock_resp.headers = {}

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    with patch("httpx.post", return_value=mock_resp):
        res = run_refund_agent(
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
            description="Pre-replay recording",
        )
        assert res["result"] == "approve"

    tr = db_session.query(Trace).filter_by(description="Pre-replay recording").one()

    # 2. Replay with httpx patched to EXPLODE if called
    def boom(*args, **kwargs):
        raise RuntimeError("FATAL: Network call attempted during deterministic replay!")

    engine = ReplayEngine(db=db_session)
    with patch("httpx.post", side_effect=boom), patch("httpx.Client.post", side_effect=boom):
        replay_res = engine.replay_agent(
            trace_id=tr.id,
            agent_fn=run_refund_agent,
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
        )

    assert replay_res.fidelity.fidelity_percent == 100.0
    assert replay_res.final_output.get("result") == "approve"


# ---------------------------------------------------------------------------
# Test 5: Fork Calls Live LLM while Tools Stay Sandboxed (Requirement 5)
# ---------------------------------------------------------------------------


def test_fork_calls_live_llm_with_sandboxed_tools(db_session, monkeypatch):
    """Steps after the fork step call the live LLM while tools write to the sandbox ledger."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Record original failing run (approved)
    orig_output = run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Failing run for live fork test",
    )
    assert orig_output["result"] == "approve"
    failed_trace = (
        db_session.query(Trace)
        .filter_by(description="Failing run for live fork test")
        .one()
    )

    # 2. Fork at step 3 with fixed policy
    # The live LLM should decide 'deny', skipping issue_refund
    fork_res = fork_trace(
        trace_id=failed_trace.id,
        step_index=3,
        overrides={"tool_output": REFUND_POLICY},
        db=db_session,
    )

    assert fork_res.final_output.get("result") == "deny"
    assert len(get_refund_ledger()) == 1  # only from original run
    assert len(get_sandbox_refund_ledger()) == 0  # denied, so nothing written


# ---------------------------------------------------------------------------
# Test 6: Offline Golden LLM Trace Replay (Requirement 8)
# ---------------------------------------------------------------------------


def test_golden_llm_trace_replay(db_session):
    """Golden real-LLM trace fixture replays offline with 100% fidelity and zero network calls."""
    fixtures_path = (
        pathlib.Path(__file__).resolve().parent / "fixtures" / "golden_llm_trace.json"
    )
    if not fixtures_path.exists():
        pytest.skip("golden_llm_trace.json fixture not yet exported")

    with open(fixtures_path, encoding="utf-8") as f:
        data = json.load(f)

    # Verify no secrets in fixture file
    raw_str = json.dumps(data)
    assert "gsk_" not in raw_str
    assert "Bearer " not in raw_str
    assert "Authorization:" not in raw_str

    # Load trace into DB
    tr_obj = Trace(
        run_id=data["run_id"] + "-fixture-test",
        agent_name=data["agent_name"],
        description=data["description"],
        meta=data.get("meta", {}),
        status=TraceStatus(data["status"]),
        run_type=TraceRunType(data["run_type"]),
        final_output=data.get("final_output"),
        agent_version=data.get("agent_version"),
        created_at=datetime.datetime.now(datetime.UTC),
    )
    db_session.add(tr_obj)
    db_session.flush()

    for s_data in data.get("steps", []):
        st = Step(
            trace_id=tr_obj.id,
            index=s_data["index"],
            node_name=s_data["node_name"],
            state_before=s_data.get("state_before"),
            state_after=s_data.get("state_after"),
            model=s_data.get("model"),
            input_tokens=s_data.get("input_tokens"),
            output_tokens=s_data.get("output_tokens"),
            cost=s_data.get("cost"),
            latency_ms=s_data.get("latency_ms", 0.0),
        )
        db_session.add(st)
        db_session.flush()

        for e_data in s_data.get("events", []):
            ev = Event(
                step_id=st.id,
                kind=EventKind(e_data["kind"]),
                seq=e_data["seq"],
                input_hash=e_data["input_hash"],
                output=e_data.get("output"),
                raw_input=e_data.get("raw_input"),
                raw_output=e_data.get("raw_output"),
                error=e_data.get("error"),
                source=EventSource(e_data["source"]),
            )
            db_session.add(ev)
        db_session.flush()

    db_session.commit()

    # Replay trace offline with network blocked
    def boom(*args, **kwargs):
        raise RuntimeError("Network call during golden replay!")

    engine = ReplayEngine(db=db_session)
    with patch("httpx.post", side_effect=boom), patch("httpx.Client.post", side_effect=boom):
        replay_result = engine.replay_trace(tr_obj.id)

    assert replay_result.fidelity.fidelity_percent == 100.0, (
        f"Replay fidelity was {replay_result.fidelity.fidelity_percent:.1f}%, expected 100%"
    )


# ---------------------------------------------------------------------------
# Test 7: Live LLM Call (Requirement 6 - @pytest.mark.live)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_openai_compatible_llm():
    """Live integration test against the real LLM endpoint (skipped by default)."""
    import dotenv
    dotenv.load_dotenv()
    if os.environ.get("RUN_LIVE_LLM") != "1" or not os.environ.get("LLM_API_KEY"):
        pytest.skip("RUN_LIVE_LLM=1 and LLM_API_KEY required for live test")

    res = openai_compatible_llm(
        "Customer: Alice Smith\nOrder: O103  Amount: $79.5\nDAYS_OLD: 12\n"
        "Policy: Refunds are accepted within 30 days of the order date.\n"
        "Decide whether to approve or deny this refund request."
    )
    assert res.decision in ("approve", "deny")
    assert len(res.reason) > 0
    assert res.total_tokens > 0
    assert res.error is None


# ---------------------------------------------------------------------------
# Test 8: Script Formatting with Non-Breaking Hyphen \u2011 (Requirement 3)
# ---------------------------------------------------------------------------


def test_live_check_formatting_with_non_breaking_hyphen():
    """Verify that formatting functions in live_check handle strings with \u2011 without raising."""
    from scripts.live_check import format_iteration_log, format_reason, format_run_row

    reason_with_hyphen = "Order is 12 days old, within the 30\u2011day refund window."

    # Must not raise on reason truncation/cleaning
    formatted = format_reason(reason_with_hyphen)
    assert "\u2011" in formatted

    # Must not raise on summary row formatting
    row = format_run_row(
        iteration=1,
        trace_id=42,
        decision="approve",
        in_toks=238,
        out_toks=73,
        latency_ms=1024.5,
        reason=reason_with_hyphen,
    )
    assert "#   42" in row
    assert "APPROVE" in row
    assert "\u2011" in row

    # Must not raise on iteration log line formatting
    log_line = format_iteration_log(
        iteration=1,
        trace_id=42,
        decision="approve",
        total_toks=311,
        latency_ms=1024.5,
        reason=reason_with_hyphen,
    )
    assert "Iteration  1/10" in log_line
    assert "\u2011" in log_line
