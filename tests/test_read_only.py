"""Tests for Read-Only Mode (AGENTREPLAY_READ_ONLY=1) and Subprocess Lockdown."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agentreplay import api
from agentreplay.api import app
from agentreplay.models import Step, Trace, TraceRunType, TraceStatus


@pytest.fixture
def test_client(db_session):
    engine = db_session.get_bind()
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[api.get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_trace(db_session):
    trace = Trace(
        run_id="run-ro-test-1",
        agent_name="refund_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
        description="Sample trace for read-only testing",
    )
    db_session.add(trace)
    db_session.commit()
    db_session.refresh(trace)

    step = Step(
        trace_id=trace.id,
        index=1,
        node_name="get_customer",
        state_before={},
        state_after={"customer_id": "C001"},
    )
    db_session.add(step)
    db_session.commit()
    return trace


# ---------------------------------------------------------------------------
# 1. Writable Mode (Default)
# ---------------------------------------------------------------------------


def test_writable_mode_allows_mutations(test_client, monkeypatch):
    """By default, AGENTREPLAY_READ_ONLY is off, so POST routes function normally."""
    monkeypatch.delenv("AGENTREPLAY_READ_ONLY", raising=False)
    assert not api.is_read_only()

    # POST /traces/ works
    res1 = test_client.post("/traces/", json={"run_id": "writable-run-1", "agent_name": "test"})
    assert res1.status_code == 200
    assert res1.json().get("created") is True

    # POST /traces without trailing slash works
    res2 = test_client.post("/traces", json={"run_id": "writable-run-2", "agent_name": "test"})
    assert res2.status_code == 200
    assert res2.json().get("created") is True


# ---------------------------------------------------------------------------
# 2. Read-Only Mode (AGENTREPLAY_READ_ONLY=1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_val", ["1", "true", "True", "yes", "YES", "on"])
def test_read_only_mode_blocks_mutating_routes(test_client, sample_trace, monkeypatch, env_val):
    """When AGENTREPLAY_READ_ONLY is enabled, all mutating routes return 403."""
    monkeypatch.setenv("AGENTREPLAY_READ_ONLY", env_val)
    assert api.is_read_only()

    # 1. POST /traces/
    res_ingest = test_client.post(
        "/traces/",
        json={"run_id": "ro-attempt-1", "agent_name": "test"},
    )
    assert res_ingest.status_code == 403
    assert "read-only mode" in res_ingest.json()["detail"].lower()

    # 2. POST /traces (no trailing slash)
    res_ingest_no_slash = test_client.post(
        "/traces",
        json={"run_id": "ro-attempt-2", "agent_name": "test"},
    )
    assert res_ingest_no_slash.status_code == 403
    assert "read-only mode" in res_ingest_no_slash.json()["detail"].lower()

    # 3. POST /traces/{id}/fork
    res_fork = test_client.post(
        f"/traces/{sample_trace.id}/fork",
        json={"step_index": 1, "overrides": {}},
    )
    assert res_fork.status_code == 403
    assert "read-only mode" in res_fork.json()["detail"].lower()

    # 4. POST /tests/generate
    res_testgen = test_client.post(
        "/tests/generate",
        json={"failed_trace_id": sample_trace.id, "fixed_trace_id": sample_trace.id},
    )
    assert res_testgen.status_code == 403
    assert "read-only mode" in res_testgen.json()["detail"].lower()


def test_read_only_mode_allows_get_routes(test_client, sample_trace, monkeypatch):
    """GET endpoints remain accessible in read-only mode."""
    monkeypatch.setenv("AGENTREPLAY_READ_ONLY", "1")

    # GET / (dashboard)
    res_index = test_client.get("/")
    assert res_index.status_code == 200

    # GET /trace/{id} (viewer)
    res_trace = test_client.get(f"/trace/{sample_trace.id}")
    assert res_trace.status_code == 200
    assert "refund_agent" in res_trace.text


# ---------------------------------------------------------------------------
# 3. Subprocess Lockdown Verification
# ---------------------------------------------------------------------------


def test_read_only_mode_prevents_subprocess_execution(test_client, sample_trace, monkeypatch):
    """Ensure no endpoint can trigger subprocess execution in read-only mode."""
    monkeypatch.setenv("AGENTREPLAY_READ_ONLY", "1")

    mock_run = MagicMock(side_effect=AssertionError("Subprocess execution is strictly forbidden!"))
    mock_popen = MagicMock(side_effect=AssertionError("Popen execution is strictly forbidden!"))

    with patch("subprocess.run", mock_run), patch("subprocess.Popen", mock_popen):
        # Even if a malicious or accidental request hits test generation,
        # it is intercepted at the HTTP boundary before any subprocess call.
        res = test_client.post(
            "/tests/generate",
            json={"failed_trace_id": sample_trace.id, "fixed_trace_id": sample_trace.id},
        )
        assert res.status_code == 403

        # Confirm zero subprocess invocations
        assert mock_run.call_count == 0
        assert mock_popen.call_count == 0


# ---------------------------------------------------------------------------
# 4. Template UI Read-Only Mode & Writable Mode Verification
# ---------------------------------------------------------------------------


def test_ui_read_only_mode_disables_actions_and_shows_banner(test_client, sample_trace, monkeypatch, db_session):
    """In read-only mode, banner is displayed and mutating buttons/forms are disabled."""
    monkeypatch.setenv("AGENTREPLAY_READ_ONLY", "1")

    # Create a fork trace so fork-diff can be rendered
    fork = Trace(
        run_id="run-ro-fork-1",
        parent_trace_id=sample_trace.id,
        fork_step=1,
        agent_name="refund_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.FORK,
        description="Fork for UI test",
    )
    db_session.add(fork)
    db_session.commit()

    # 1. Dashboard: shows read-only banner
    res_index = test_client.get("/")
    assert res_index.status_code == 200
    assert "read-only demo mode" in res_index.text.lower()

    # 2. Trace viewer: shows banner, fork form is disabled
    res_trace = test_client.get(f"/trace/{sample_trace.id}")
    assert res_trace.status_code == 200
    assert "read-only demo mode" in res_trace.text.lower()
    assert "Fork Disabled" in res_trace.text
    assert "disabled" in res_trace.text

    # 3. Fork diff: shows banner, create test button is disabled
    res_diff = test_client.get(f"/fork-diff/{sample_trace.id}/{fork.id}")
    assert res_diff.status_code == 200
    assert "read-only demo mode" in res_diff.text.lower()
    assert "Create Test (Disabled)" in res_diff.text


def test_ui_writable_mode_enables_actions_without_banner(test_client, sample_trace, monkeypatch, db_session):
    """In writable mode, banner is absent and action forms are submittable."""
    monkeypatch.delenv("AGENTREPLAY_READ_ONLY", raising=False)

    fork = Trace(
        run_id="run-ro-fork-2",
        parent_trace_id=sample_trace.id,
        fork_step=1,
        agent_name="refund_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.FORK,
    )
    db_session.add(fork)
    db_session.commit()

    # 1. Dashboard
    res_index = test_client.get("/")
    assert res_index.status_code == 200
    assert "read-only demo mode" not in res_index.text.lower()

    # 2. Trace viewer: submittable fork form
    res_trace = test_client.get(f"/trace/{sample_trace.id}")
    assert res_trace.status_code == 200
    assert "read-only demo mode" not in res_trace.text.lower()
    assert f'action="/traces/{sample_trace.id}/fork"' in res_trace.text
    assert "Fork Trace →" in res_trace.text

    # 3. Fork diff: submittable create test form
    res_diff = test_client.get(f"/fork-diff/{sample_trace.id}/{fork.id}")
    assert res_diff.status_code == 200
    assert "read-only demo mode" not in res_diff.text.lower()
    assert 'action="/tests/generate"' in res_diff.text
    assert "Create Test →" in res_diff.text


# ---------------------------------------------------------------------------
# 5. Ground Truth Leakage Prevention & Healthcheck Verification
# ---------------------------------------------------------------------------


def test_viewer_does_not_display_eval_label_ground_truth(test_client, sample_trace, db_session):
    """Assert trace viewer never leaks ground-truth EvalLabel metadata."""
    from agentreplay.models import EvalLabel

    eval_label = EvalLabel(
        trace_id=sample_trace.id,
        injected_fault="secret_stale_policy_fault",
        ground_truth_step=99,
        ground_truth_node="secret_truth_node",
    )
    db_session.add(eval_label)
    db_session.commit()

    res = test_client.get(f"/trace/{sample_trace.id}")
    assert res.status_code == 200

    # Ensure ground truth fields are never rendered in the HTML
    assert "secret_stale_policy_fault" not in res.text
    assert "secret_truth_node" not in res.text
    assert "ground_truth" not in res.text.lower()
    assert "eval_label" not in res.text.lower()


def test_healthz_endpoint(test_client):
    """GET /healthz returns 200 and JSON {'status': 'ok'}."""
    res = test_client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

