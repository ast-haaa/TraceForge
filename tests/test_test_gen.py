"""Phase 4 Tests: Regression Test Generation, Subprocess Validation, and Safety."""

import ast
import datetime
import os
import pathlib
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from agentreplay import api
from agentreplay.api import app
from agentreplay.fork import fork_trace
from agentreplay.models import GeneratedTest, Trace
from agentreplay.test_gen import generate_test, sanitize_trace_id, validate_test
from demo_agent.agent import run_refund_agent
from demo_agent.tools import reset_ledger


@pytest.fixture
def workspace_tmp_dir():
    """Ensure generated test files reside in the workspace repository drive (E:),
    avoiding Windows cross-drive permission traversal in pytest subprocesses.
    """
    target = pathlib.Path(__file__).resolve().parent.parent / "generated_tests" / "_pytest_tmp"
    target.mkdir(parents=True, exist_ok=True)
    yield target
    for f in target.glob("*.py"):
        try:
            f.unlink()
        except Exception:
            pass


@pytest.fixture
def client(db_session):
    engine = db_session.get_bind()
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

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


@pytest.fixture
def failed_and_fixed_traces(db_session, monkeypatch):
    """Fixture that produces a failed trace (stale policy) and its fixed fork trace (7-day policy)."""
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    reset_ledger()
    # 1. Failing run: order O103 (12 days old) wrongly approved under 30-day stale policy
    run_refund_agent(
        customer_id="C001",
        order_id="O103",
        stale_policy=True,
        description="Failing trace for test generation",
    )
    failed_trace = (
        db_session.query(Trace)
        .filter_by(description="Failing trace for test generation")
        .one()
    )

    # 2. Fixed fork: 7-day policy overrides Step 3 -> order O103 denied
    fixed_policy = "Refunds are accepted within 7 days of the order date. No refunds after 7 days."
    fixed_trace = fork_trace(
        trace_id=failed_trace.id,
        step_index=3,
        overrides={"tool_output": fixed_policy},
        db=db_session,
    )
    return failed_trace, fixed_trace


def test_generated_file_compiles(db_session, failed_and_fixed_traces, workspace_tmp_dir):
    """The generated test file must exist and compile cleanly without syntax errors."""
    failed, fixed = failed_and_fixed_traces

    gen_test = generate_test(
        failed.id, fixed.id, db=db_session, output_dir=workspace_tmp_dir
    )

    assert gen_test.trace_id == failed.id
    assert os.path.exists(gen_test.file_path)

    # Verify syntax compilation
    with open(gen_test.file_path, encoding="utf-8") as f:
        code = f.read()

    compiled = compile(code, gen_test.file_path, "exec")
    assert compiled is not None


def test_generated_test_fails_on_buggy_and_passes_on_fixed(
    db_session, failed_and_fixed_traces, workspace_tmp_dir
):
    """Subprocess validation:
    - Must FAIL on buggy agent variant (stale policy approved out-of-window refund)
    - Must PASS on fixed agent variant (7-day policy denies out-of-window refund)
    - Validates test status becomes 'valid'
    """
    failed, fixed = failed_and_fixed_traces

    gen_test = generate_test(
        failed.id, fixed.id, db=db_session, output_dir=workspace_tmp_dir
    )
    validated = validate_test(gen_test.id, db=db_session)

    assert validated.fails_on_buggy is True, (
        "Generated test should fail on buggy agent"
    )
    assert validated.passes_on_fixed is True, (
        "Generated test should pass on fixed agent"
    )
    assert validated.status == "valid"
    assert validated.rejection_reason is None
    assert "RUN 1: BUGGY VARIANT" in validated.pytest_output
    assert "RUN 2: FIXED VARIANT" in validated.pytest_output


def test_positive_control_present_in_generated_test(
    db_session, failed_and_fixed_traces, workspace_tmp_dir
):
    """AST check: parse generated test with AST and assert:
    - Negative controls parametrize failing trace's own days_old and 8 (boundary)
    - Positive control uses 7
    """
    failed, fixed = failed_and_fixed_traces

    gen_test = generate_test(
        failed.id, fixed.id, db=db_session, output_dir=workspace_tmp_dir
    )

    code = gen_test.test_code
    tree = ast.parse(code)

    func_neg = None
    func_pos = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.name == "test_out_of_window_refund_denied_and_ledger_untouched":
                func_neg = node
            elif node.name == "test_positive_control_within_window_refund_approved":
                func_pos = node

    assert func_neg is not None, "Negative control function not found"
    assert func_pos is not None, "Positive control function not found"

    # 1. Assert parametrize list contains failing trace's days_old (12) and 8 (boundary)
    parametrize_days = None
    for dec in func_neg.decorator_list:
        if isinstance(dec, ast.Call) and len(dec.args) >= 2:
            if isinstance(dec.args[0], ast.Constant) and dec.args[0].value == "days_old":
                if isinstance(dec.args[1], ast.List):
                    parametrize_days = [
                        elt.value for elt in dec.args[1].elts if isinstance(elt, ast.Constant)
                    ]

    assert parametrize_days is not None, "Could not find parametrize list for days_old"
    expected_days_old = failed.final_output.get("days_old", 12)
    assert expected_days_old in parametrize_days, (
        f"Failing trace's days_old ({expected_days_old}) not found in parametrize list: {parametrize_days}"
    )
    assert 8 in parametrize_days, f"Boundary 8 not found in parametrize list: {parametrize_days}"

    # 2. Assert positive control uses 7
    positive_days = None
    for node in ast.walk(func_pos):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "days" and isinstance(kw.value, ast.Constant):
                    positive_days = kw.value.value

    assert positive_days == 7, f"Positive control should use 7, got: {positive_days}"


def test_generate_test_derives_days_old_from_trace(
    db_session, workspace_tmp_dir, monkeypatch
):
    """Generate from a failing trace whose order is 15 days old and assert 15 is in
    the parametrize list (proves the value is derived from the trace, not a constant).
    """
    import agentreplay.db as db_mod
    import agentreplay.sdk as sdk_mod
    from demo_agent.tools import ORDERS, REFUND_POLICY

    monkeypatch.setattr(sdk_mod, "SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: db_session)

    # 1. Create a 15-day-old order
    order_id = "O_TEST_15_DAYS"
    order_date = (datetime.date.today() - datetime.timedelta(days=15)).isoformat()
    mock_order = {
        "order_id": order_id,
        "customer_id": "C001",
        "amount": 85.00,
        "order_date": order_date,
        "item": "15-Day Item",
    }
    monkeypatch.setitem(ORDERS, order_id, mock_order)

    # 2. Record failing run under stale 30-day policy (15-day-old order wrongly approved)
    run_refund_agent(
        customer_id="C001",
        order_id=order_id,
        stale_policy=True,
        description="Failing trace 15 days old",
    )
    failed_trace = (
        db_session.query(Trace)
        .filter_by(description="Failing trace 15 days old")
        .one()
    )
    assert failed_trace.final_output.get("days_old") == 15

    # 3. Fork with 7-day policy (denies refund)
    fixed_trace = fork_trace(
        trace_id=failed_trace.id,
        step_index=3,
        overrides={"tool_output": REFUND_POLICY},
        db=db_session,
    )

    # 4. Generate regression test
    gen_test = generate_test(
        failed_trace.id, fixed_trace.id, db=db_session, output_dir=workspace_tmp_dir
    )

    # 5. Parse generated code with ast and verify 15 is in the parametrize list
    tree = ast.parse(gen_test.test_code)
    func_neg = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "test_out_of_window_refund_denied_and_ledger_untouched":
            func_neg = node
            break

    assert func_neg is not None, "Negative control function not found in AST"

    parametrize_days = None
    for dec in func_neg.decorator_list:
        if isinstance(dec, ast.Call) and len(dec.args) >= 2:
            if isinstance(dec.args[0], ast.Constant) and dec.args[0].value == "days_old":
                if isinstance(dec.args[1], ast.List):
                    parametrize_days = [
                        elt.value for elt in dec.args[1].elts if isinstance(elt, ast.Constant)
                    ]

    assert parametrize_days is not None, "Could not find parametrize list for days_old"
    assert 15 in parametrize_days, (
        f"Expected 15 to be in parametrize list, got: {parametrize_days}"
    )
    assert 8 in parametrize_days, (
        f"Expected 8 (boundary) to be in parametrize list, got: {parametrize_days}"
    )


def test_always_deny_agent_fails_generated_test(
    db_session, failed_and_fixed_traces, workspace_tmp_dir
):
    """An agent that always denies every refund must FAIL the generated test.

    Proves that the positive control prevents over-blocking agents from passing.
    """
    failed, fixed = failed_and_fixed_traces

    gen_test = generate_test(
        failed.id, fixed.id, db=db_session, output_dir=workspace_tmp_dir
    )

    # Run generated test under AGENTREPLAY_AGENT="demo_agent.variants:always_deny_agent" in subprocess
    env = os.environ.copy()
    env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parent.parent)
    env["AGENTREPLAY_AGENT"] = "demo_agent.variants:always_deny_agent"

    project_root = pathlib.Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "testpaths=",
        "-c",
        str(project_root / "pyproject.toml"),
        str(gen_test.file_path),
        "-v",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Subprocess must fail because positive control fails on always_deny
    assert proc.returncode != 0, (
        "An always-deny agent should fail the positive control"
    )
    output = proc.stdout + proc.stderr
    assert "test_positive_control_within_window_refund_approved" in output
    assert "FAILED" in output


def test_invalid_test_is_rejected(db_session, workspace_tmp_dir):
    """validate_test must reject a test that does not fail on buggy or pass on fixed."""
    # Create a dummy test file that unconditionally passes (does NOT fail on buggy)
    dummy_file = workspace_tmp_dir / "test_trace_9999.py"
    dummy_file.write_text("def test_dummy(): pass\n", encoding="utf-8")

    fake_test = GeneratedTest(
        trace_id=9999,
        description="Dummy test that always passes",
        test_code="def test_dummy(): pass\n",
        file_path=str(dummy_file),
        status="pending",
    )
    db_session.add(fake_test)
    db_session.commit()

    validated = validate_test(fake_test.id, db=db_session)

    # Must be marked rejected because it passed on buggy
    assert validated.status == "rejected"
    assert validated.fails_on_buggy is False
    assert validated.passes_on_fixed is True
    assert "did not fail on buggy" in validated.rejection_reason


def test_xss_escaping_on_test_page(client, db_session, workspace_tmp_dir):
    """HTML escaping holds on /tests/{id} viewer and no |safe is used."""
    xss_payload = "<script>alert('test_page_xss')</script>"
    dummy_file = workspace_tmp_dir / "test_trace_8888.py"
    dummy_file.write_text("# safe\n", encoding="utf-8")

    gen_test = GeneratedTest(
        trace_id=8888,
        description=f"Test with {xss_payload}",
        test_code=f"# Code {xss_payload}\n",
        file_path=str(dummy_file),
        fails_on_buggy=True,
        passes_on_fixed=True,
        status="valid",
        pytest_output=f"Output {xss_payload}",
    )
    db_session.add(gen_test)
    db_session.commit()

    res = client.get(f"/tests/{gen_test.id}")
    assert res.status_code == 200
    body = res.text

    # Raw unescaped payload must NEVER appear
    assert "<script>alert('test_page_xss')</script>" not in body
    # Escaped entity MUST appear
    assert "&lt;script&gt;" in body

    # Confirm no |safe filter exists in test_view.html
    template_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "agentreplay"
        / "templates"
        / "test_view.html"
    )
    assert template_path.exists()
    content = template_path.read_text(encoding="utf-8")
    assert "|safe" not in content, (
        "Unsafe '|safe' filter found in agentreplay/templates/test_view.html"
    )


def test_path_safety():
    """Ensure non-integer trace IDs or path traversals are rejected."""
    # 1. Traversal strings must raise ValueError
    with pytest.raises(ValueError):
        sanitize_trace_id("../../etc/passwd")

    with pytest.raises(ValueError):
        sanitize_trace_id("trace_12; rm -rf")

    # 2. Boolean or invalid types must raise TypeError
    with pytest.raises(TypeError):
        sanitize_trace_id(True)

    with pytest.raises(TypeError):
        sanitize_trace_id([1, 2])

    # 3. Non-positive numbers must raise ValueError
    with pytest.raises(ValueError):
        sanitize_trace_id(0)

    with pytest.raises(ValueError):
        sanitize_trace_id(-10)

    # 4. Valid integer coerced cleanly
    assert sanitize_trace_id(42) == 42
    assert sanitize_trace_id("42") == 42


def test_api_generate_and_download_endpoints(
    client, db_session, failed_and_fixed_traces
):
    """Test POST /tests/generate (form & JSON) and GET /tests/{id}/download."""
    failed, fixed = failed_and_fixed_traces

    # 1. JSON POST
    res_json = client.post(
        "/tests/generate",
        json={"failed_trace_id": failed.id, "fixed_trace_id": fixed.id},
    )
    assert res_json.status_code == 200
    data = res_json.json()
    assert "test_id" in data
    assert data["status"] == "valid"
    assert data["fails_on_buggy"] is True
    assert data["passes_on_fixed"] is True
    test_id = data["test_id"]

    # 2. GET /tests/{id}
    res_view = client.get(f"/tests/{test_id}")
    assert res_view.status_code == 200
    assert "Generated Regression Test" in res_view.text
    assert "FAILS ON BUGGY" in res_view.text
    assert "PASSES ON FIXED" in res_view.text

    # 3. GET /tests/{id}/download
    res_dl = client.get(f"/tests/{test_id}/download")
    assert res_dl.status_code == 200
    assert "test_out_of_window_refund_denied" in res_dl.text
    assert "attachment;" in res_dl.headers.get("content-disposition", "")
