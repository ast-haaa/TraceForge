# agentreplay/api.py
"""FastAPI application.

Routes
------
GET  /                   – list all traces (dashboard)
GET  /trace/{trace_id}   – trace detail / viewer
POST /traces/            – lightweight ingest endpoint (SDK writes directly;
                           this is for external callers or future streaming)
"""
from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .db import SessionLocal, init_db
from .models import Event, Step, Trace, TraceStatus
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
    from .models import TraceRunType
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
