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
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import SessionLocal, init_db
from .diff import diff_traces
from .fork import fork_trace
from .models import Event, Step, Trace, TraceRunType, TraceStatus
from .redact import redact_sensitive

init_db()

app = FastAPI(title="AgentReplay")

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates")
templates = Jinja2Templates(directory=_TEMPLATE_DIR)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def list_traces(request: Request, db: Session = Depends(get_db)):
    traces = db.query(Trace).order_by(Trace.created_at.desc()).all()
    return templates.TemplateResponse(request=request, name="index.html", context={"traces": traces})


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
        context={"trace": trace, "steps": steps},
    )


@app.post("/traces/")
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
        },
    )
