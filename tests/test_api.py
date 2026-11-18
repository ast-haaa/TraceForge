"""Tests for FastAPI endpoints and trace viewer rendering."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agentreplay.api import app
from agentreplay.models import Event, EventKind, EventSource, Step, Trace, TraceRunType, TraceStatus


@pytest.fixture
def client(db_session):
    engine = db_session.get_bind()
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    from agentreplay import api
    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[api.get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_dashboard_empty(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "No traces recorded yet" in response.text


def test_dashboard_with_traces(client, db_session):
    trace = Trace(
        run_id="run-api-1",
        agent_name="refund_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.commit()

    response = client.get("/")
    assert response.status_code == 200
    assert "refund_agent" in response.text
    assert "completed" in response.text


def test_view_trace_not_found(client):
    response = client.get("/trace/99999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Trace not found"


def test_view_trace_found(client, db_session):
    trace = Trace(
        run_id="run-api-view",
        agent_name="refund_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.flush()

    step = Step(
        trace_id=trace.id,
        index=1,
        node_name="get_customer",
        state_before={"customer_id": "C001"},
        state_after={"customer": {"name": "Alice"}},
    )
    db_session.add(step)
    db_session.flush()

    event = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=1,
        input_hash="a" * 64,
        output={"name": "Alice", "email": "<REDACTED_EMAIL>"},
        raw_input='{"customer_id": "C001"}',
        raw_output='{"name": "Alice"}',
        source=EventSource.RECORDED,
    )
    db_session.add(event)
    db_session.commit()

    response = client.get(f"/trace/{trace.id}")
    assert response.status_code == 200
    assert "get_customer" in response.text
    assert "tool_call" in response.text
    assert "recorded" in response.text
    assert "alice@example.com" not in response.text


def test_ingest_trace_endpoint(client):
    payload = {
        "run_id": "run-post-1",
        "agent_name": "refund_agent",
        "description": "via post",
        "meta": {"api_key": "secret_key_123", "env": "prod"},
    }
    response = client.post("/traces/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["created"] is True
    assert "trace_id" in data

    # Idempotent retry
    response_retry = client.post("/traces/", json=payload)
    assert response_retry.status_code == 200
    assert response_retry.json()["created"] is False
    assert response_retry.json()["trace_id"] == data["trace_id"]


def test_xss_protection_in_trace_viewer(client, db_session):
    """HTML tags like <script> in raw_input / raw_output must be auto-escaped by Jinja2."""
    trace = Trace(
        run_id="run-xss-test",
        agent_name="xss_test_agent",
        status=TraceStatus.COMPLETED,
        run_type=TraceRunType.ORIGINAL,
    )
    db_session.add(trace)
    db_session.flush()

    step = Step(
        trace_id=trace.id,
        index=1,
        node_name="malicious_node",
        state_before={},
        state_after={},
    )
    db_session.add(step)
    db_session.flush()

    malicious_payload = "<script>alert(1)</script>"
    event = Event(
        step_id=step.id,
        kind=EventKind.TOOL_CALL,
        seq=1,
        input_hash="x" * 64,
        output={"msg": malicious_payload},
        raw_input=malicious_payload,
        raw_output=malicious_payload,
        source=EventSource.RECORDED,
    )
    db_session.add(event)
    db_session.commit()

    response = client.get(f"/trace/{trace.id}")
    assert response.status_code == 200
    assert "<script>alert(1)" not in response.text
    assert "&lt;script&gt;" in response.text


def test_no_safe_filter_in_templates():
    """Confirm no Jinja2 template uses |safe on rendered fields."""
    import pathlib

    template_dir = pathlib.Path(__file__).resolve().parent.parent / "templates"
    for html_file in template_dir.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        assert "|safe" not in content, f"Found unsafe '|safe' filter in {html_file.name}"

