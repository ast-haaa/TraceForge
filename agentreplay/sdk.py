# agentreplay/sdk.py
"""SDK for recording LangGraph agent executions incrementally.

Usage::

    from agentreplay.sdk import record_trace
    from agentreplay.models import EventKind

    with record_trace(agent_name="refund_agent", description="demo run") as rec:
        step = rec.start_step("route", state_before={"messages": [...]})
        rec.record_event(
            kind=EventKind.TOOL_CALL,
            raw_input={"name": "get_customer", "args": {"customer_id": "c1"}},
            output={"name": "Alice", "email": "alice@example.com"},
        )
        rec.end_step(state_after={"messages": [...], "customer": {...}})
        rec.finalize(final_output={"result": "refund_issued"})

The context manager catches exceptions and sets status=FAILED so that
crashed runs are still recorded.  Every flush() call persists state
immediately so partial traces survive process crashes.
"""
from __future__ import annotations

import contextvars
import datetime
import json
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy.orm import Session

from .db import SessionLocal, init_db
from .hash import compute_input_hash
from .models import Event, EventKind, EventSource, Step, Trace, TraceRunType, TraceStatus
from .redact import redact_sensitive

# Context variable used during deterministic replay sessions
_active_replay_session: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "_active_replay_session", default=None
)

# Ensure tables exist on import
init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jsonify(value: Any) -> str | None:
    """Serialize *value* to a compact JSON string, or None for None."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class Recorder:
    """Stateful helper yielded by :func:`record_trace`.

    All writes are flushed immediately so a crash leaves a consistent
    partial trace in the database.
    """

    def __init__(
        self,
        db: Session,
        run_id: str,
        agent_name: str,
        description: str | None = None,
        meta: dict | None = None,
        agent_version: str | None = None,
        run_type: TraceRunType = TraceRunType.ORIGINAL,
        parent_trace_id: int | None = None,
        fork_step: int | None = None,
    ) -> None:
        self.db = db
        # Idempotent: if run_id already exists, reuse that trace row
        existing = db.query(Trace).filter_by(run_id=run_id).first()
        if existing:
            self.trace = existing
        else:
            self.trace = Trace(
                run_id=run_id,
                agent_name=agent_name,
                description=description,
                meta=redact_sensitive(meta or {}),
                status=TraceStatus.RECORDING,
                run_type=run_type,
                agent_version=agent_version,
                parent_trace_id=parent_trace_id,
                fork_step=fork_step,
                created_at=datetime.datetime.now(datetime.UTC),
            )
            db.add(self.trace)
            db.flush()

        # If the trace already has steps (idempotent re-use), start counter after them
        existing_max = (
            db.query(Step)
            .filter_by(trace_id=self.trace.id)
            .order_by(Step.index.desc())
            .first()
        )
        self._step_counter: int = existing_max.index if existing_max else 0
        self._current_step: Step | None = None
        self._event_seq: int = 0

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def start_step(
        self,
        node_name: str,
        state_before: dict | None = None,
    ) -> Step:
        """Create and flush a new Step row; return it."""
        self._step_counter += 1
        step = Step(
            trace_id=self.trace.id,
            index=self._step_counter,
            node_name=node_name,
            state_before=redact_sensitive(state_before or {}),
        )
        self.db.add(step)
        self.db.flush()
        self._current_step = step
        self._event_seq = 0
        return step

    def end_step(
        self,
        state_after: dict | None = None,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float | None = None,
        step: Step | None = None,
    ) -> None:
        """Finalise the current step with its after-state and metrics."""
        s = step or self._current_step
        if s is None:
            raise RuntimeError("end_step() called without an active step")
        s.state_after = redact_sensitive(state_after or {})
        s.model = model
        s.input_tokens = input_tokens
        s.output_tokens = output_tokens
        s.cost = cost
        s.latency_ms = latency_ms
        self.db.flush()
        if self._current_step is s:
            self._current_step = None

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        kind: EventKind,
        raw_input: Any,
        output: Any,
        *,
        error: str | None = None,
        source: EventSource = EventSource.RECORDED,
        step: Step | None = None,
    ) -> Event:
        """Record one event inside the active step.

        ``raw_input`` and ``output`` are redacted before hashing and storage.
        """
        target_step = step or self._current_step
        if target_step is None:
            raise RuntimeError("record_event() called without an active step")
        self._event_seq += 1

        if source == EventSource.RECORDED and _active_replay_session.get() is not None:
            source = EventSource.REPLAYED

        safe_input = redact_sensitive(raw_input)
        safe_output = redact_sensitive(output)
        input_hash = compute_input_hash(safe_input)

        event = Event(
            step_id=target_step.id,
            kind=kind,
            seq=self._event_seq,
            input_hash=input_hash,
            raw_input=_jsonify(safe_input),
            raw_output=_jsonify(safe_output),
            output=safe_output,
            error=error,
            source=source,
        )
        self.db.add(event)
        self.db.flush()
        return event

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize(
        self,
        final_output: Any = None,
        status: TraceStatus = TraceStatus.COMPLETED,
    ) -> Trace:
        """Mark the trace completed and commit."""
        self.trace.final_output = redact_sensitive(final_output)
        self.trace.status = status
        self.db.flush()
        self.db.commit()
        return self.trace

    @property
    def trace_id(self) -> int:
        return self.trace.id


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def record_trace(
    *,
    run_id: str | None = None,
    agent_name: str,
    description: str | None = None,
    meta: dict | None = None,
    agent_version: str | None = None,
    run_type: TraceRunType = TraceRunType.ORIGINAL,
    parent_trace_id: int | None = None,
    fork_step: int | None = None,
) -> Generator[Recorder, None, None]:
    """Context manager that yields a :class:`Recorder`.

    On clean exit the trace is finalised as COMPLETED (unless the caller
    already called ``recorder.finalize()``).  On exception the trace is
    marked FAILED and the exception is re-raised.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    replay_info = _active_replay_session.get()
    if replay_info is not None:
        if run_type == TraceRunType.ORIGINAL:
            run_type = TraceRunType.REPLAY
        if parent_trace_id is None:
            parent_trace_id = replay_info.get("parent_trace_id")
        if description is None and parent_trace_id is not None:
            description = f"Replay of Trace #{parent_trace_id}"

    db: Session = SessionLocal()
    recorder = Recorder(
        db=db,
        run_id=run_id,
        agent_name=agent_name,
        description=description,
        meta=meta,
        agent_version=agent_version,
        run_type=run_type,
        parent_trace_id=parent_trace_id,
        fork_step=fork_step,
    )
    try:
        yield recorder
        if recorder.trace.status == TraceStatus.RECORDING:
            recorder.finalize(status=TraceStatus.COMPLETED)
    except Exception:
        recorder.trace.status = TraceStatus.FAILED
        db.flush()
        db.commit()
        raise
    finally:
        db.close()
