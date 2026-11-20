# agentreplay/api.py
"""FastAPI application.

Routes
------
GET  /                                  – list all traces (dashboard)
GET  /trace/{trace_id}                  – trace detail / viewer
POST /traces/                           – lightweight ingest endpoint
POST /traces/{trace_id}/fork            – fork a trace at a step and redirect to diff
GET  /fork-diff/{original_id}/{fork_id} – side-by-side visual diff
"""
from __future__ import annotations

import json
import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import SessionLocal, init_db
from .diff import diff_traces
from .fork import fork_trace
from .models import Event, GeneratedTest, Step, Trace, TraceRunType, TraceStatus
from .redact import redact_sensitive
from .test_gen import generate_test, sanitize_trace_id, validate_test

init_db()

app = FastAPI(title="AgentReplay")

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


def is_read_only() -> bool:
    """Check if the server is running in read-only demo mode."""
    return os.getenv("AGENTREPLAY_READ_ONLY", "").strip().lower() in ("1", "true", "yes", "on")


@app.middleware("http")
async def check_read_only_mode(request: Request, call_next):
    if is_read_only() and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        return JSONResponse(
            status_code=403,
            content={
                "detail": (
                    "AgentReplay is running in read-only mode (AGENTREPLAY_READ_ONLY=1). "
                    "Mutating operations are disabled."
                ),
            },
        )
    return await call_next(request)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
def healthz():
    """Health check endpoint for container orchestrators and monitoring."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def list_traces(request: Request, db: Session = Depends(get_db)):
    traces = db.query(Trace).order_by(Trace.created_at.desc()).all()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"traces": traces, "read_only": is_read_only()},
    )


@app.get("/trace/{trace_id}", response_class=HTMLResponse)
def view_trace(request: Request, trace_id: int, db: Session = Depends(get_db)):
    trace = db.query(Trace).filter(Trace.id == trace_id).first()
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    steps = (
        db.query(Step)
        .filter(Step.trace_id == trace.id)
        .order_by(Step.index)
        .all()
    )
    for step in steps:
        step.sorted_events = (
            db.query(Event)
            .filter(Event.step_id == step.id)
            .order_by(Event.seq)
            .all()
        )

    return templates.TemplateResponse(
        request=request,
        name="trace.html",
        context={"trace": trace, "steps": steps, "read_only": is_read_only()},
    )


@app.post("/traces/")
@app.post("/traces")
def ingest_trace(payload: dict, db: Session = Depends(get_db)):
    """Idempotent ingest – returns existing trace if run_id already exists."""
    run_id = payload.get("run_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id is required")
    existing = db.query(Trace).filter(Trace.run_id == run_id).first()
    if existing:
        return {"trace_id": existing.id, "created": False}
    trace = Trace(
        run_id=run_id,
        agent_name=payload.get("agent_name", "unknown"),
        description=payload.get("description"),
        meta=redact_sensitive(payload.get("meta", {})),
        status=TraceStatus.RECORDING,
        run_type=TraceRunType.ORIGINAL,
    )
    db.add(trace)
    db.commit()
    db.refresh(trace)
    return {"trace_id": trace.id, "created": True}


@app.post("/traces/{trace_id}/fork")
async def fork_trace_endpoint(
    request: Request,
    trace_id: int,
    db: Session = Depends(get_db),
):
    """Fork a trace at step_index, run forward in sandbox, and redirect to diff."""
    orig_trace = db.query(Trace).filter(Trace.id == trace_id).first()
    if not orig_trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    content_type = request.headers.get("content-type", "")
    is_json = "application/json" in content_type

    if is_json:
        payload = await request.json()
        step_index = int(payload.get("step_index", 1))
        overrides = payload.get("overrides")
    else:
        form = await request.form()
        step_index = int(form.get("step_index", 1))
        override_type = form.get("override_type", "tool_output")
        raw_val = form.get("override_value", "")
        try:
            overrides_val = json.loads(raw_val)
        except Exception:
            overrides_val = raw_val

        if override_type == "tool_output":
            overrides = {"tool_output": overrides_val}
        elif override_type == "prompt":
            overrides = {"prompt": overrides_val}
        elif override_type == "model":
            overrides = {"model": str(overrides_val)}
        else:
            overrides = overrides_val

    fork = fork_trace(
        trace_id=trace_id,
        step_index=step_index,
        overrides=overrides,
        db=db,
        raise_on_error=False,
    )

    if is_json:
        return {
            "fork_trace_id": fork.id,
            "status": fork.status.value,
            "diff_url": f"/fork-diff/{trace_id}/{fork.id}",
        }
    return RedirectResponse(url=f"/fork-diff/{trace_id}/{fork.id}", status_code=303)


@app.get("/fork-diff/{original_id}/{fork_id}", response_class=HTMLResponse)
def view_fork_diff(
    request: Request,
    original_id: int,
    fork_id: int,
    db: Session = Depends(get_db),
):
    """Render side-by-side visual diff between original trace and fork trace."""
    orig_trace = db.query(Trace).filter(Trace.id == original_id).first()
    if not orig_trace:
        raise HTTPException(status_code=404, detail="Original trace not found")

    fork_trace_obj = db.query(Trace).filter(Trace.id == fork_id).first()
    if not fork_trace_obj:
        raise HTTPException(status_code=404, detail="Fork trace not found")

    diff_result = diff_traces(original_id, fork_id, db=db)

    return templates.TemplateResponse(
        request=request,
        name="fork_diff.html",
        context={
            "diff": diff_result,
            "orig_trace": orig_trace,
            "fork_trace": fork_trace_obj,
            "read_only": is_read_only(),
        },
    )


@app.post("/tests/generate")
async def generate_and_validate_test_endpoint(
    request: Request,
    db: Session = Depends(get_db),
):
    """Generate and validate a pytest regression test from failed and fixed traces."""
    content_type = request.headers.get("content-type", "")
    is_json = "application/json" in content_type

    if is_json:
        payload = await request.json()
        failed_trace_id = payload.get("failed_trace_id")
        fixed_trace_id = payload.get("fixed_trace_id")
    else:
        form = await request.form()
        failed_trace_id = form.get("failed_trace_id")
        fixed_trace_id = form.get("fixed_trace_id")

    try:
        clean_failed_id = sanitize_trace_id(failed_trace_id)
        clean_fixed_id = sanitize_trace_id(fixed_trace_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    gen_test = generate_test(clean_failed_id, clean_fixed_id, db=db)
    validated = validate_test(gen_test.id, db=db)

    if is_json:
        return {
            "test_id": validated.id,
            "status": validated.status,
            "fails_on_buggy": validated.fails_on_buggy,
            "passes_on_fixed": validated.passes_on_fixed,
            "url": f"/tests/{validated.id}",
        }
    return RedirectResponse(url=f"/tests/{validated.id}", status_code=303)


@app.get("/tests/{test_id}", response_class=HTMLResponse)
def view_generated_test(
    request: Request,
    test_id: int,
    db: Session = Depends(get_db),
):
    """Render the generated regression test viewer."""
    gen_test = db.query(GeneratedTest).filter(GeneratedTest.id == test_id).first()
    if not gen_test:
        raise HTTPException(status_code=404, detail="Generated test not found")

    return templates.TemplateResponse(
        request=request,
        name="test_view.html",
        context={"test": gen_test, "read_only": is_read_only()},
    )


@app.get("/tests/{test_id}/download")
def download_generated_test(
    test_id: int,
    db: Session = Depends(get_db),
):
    """Download the generated test python file."""
    gen_test = db.query(GeneratedTest).filter(GeneratedTest.id == test_id).first()
    if not gen_test:
        raise HTTPException(status_code=404, detail="Generated test not found")

    from fastapi.responses import Response

    return Response(
        content=gen_test.test_code,
        media_type="text/x-python",
        headers={"Content-Disposition": f"attachment; filename=test_trace_{gen_test.trace_id}.py"},
    )
