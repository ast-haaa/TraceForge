"""AgentReplay Trace Forking Engine.

Core loop: FAILURE -> REPLAY -> FORK -> REGRESSION TEST.

Forks an existing trace at a specific step:
1. Steps before step_index are replayed from the recording (copied, source=replayed).
2. At step_index, applies the overrides: replace a tool output, replace the LLM prompt,
   or switch the model. Mark changed events source=mocked or live accordingly.
3. Steps after step_index run the real agent code forward from the restored Step.state_before.
4. Creates a new Trace: run_type=fork, parent_trace_id, fork_step set.
5. Safety: during a fork every side-effecting tool (issue_refund) writes to a sandbox ledger,
   never the real one.
6. UNRECORDED_CALL policy: if a forked run makes a call that has no recording and no user-supplied
   mock, stop the fork immediately, mark the trace status=failed with a clear UNRECORDED_CALL error,
   and list the call. Never guess.
"""
from __future__ import annotations

import contextlib
import datetime
import uuid
from collections.abc import Callable, Generator
from typing import Any

from sqlalchemy.orm import Session

from demo_agent.tools import reset_sandbox_mode, set_sandbox_mode

from .db import SessionLocal
from .models import (
    Event,
    EventKind,
    EventSource,
    Step,
    Trace,
    TraceRunType,
    TraceStatus,
)
from .redact import redact_sensitive
from .registry import get_agent
from .replay import UnrecordedCall
from .sdk import Recorder

# ---------------------------------------------------------------------------
# Sandbox context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def sandbox_mode() -> Generator[None, None, None]:
    """Context manager ensuring all side-effecting tools write strictly to the sandbox ledger."""
    token = set_sandbox_mode(True)
    try:
        yield
    finally:
        reset_sandbox_mode(token)


# ---------------------------------------------------------------------------
# Fork Implementation
# ---------------------------------------------------------------------------


def fork_trace(
    trace_id: int,
    step_index: int,
    overrides: Any = None,
    *,
    agent_fn: Callable[..., Any] | None = None,
    db: Session | None = None,
    raise_on_error: bool = True,
) -> Trace:
    """Fork a recorded trace at `step_index`, apply overrides, and run forward.

    Parameters
    ----------
    trace_id : int
        ID of the original trace to fork from.
    step_index : int
        1-based step index at which to branch.
    overrides : Any, optional
        Replacement data at the fork step: e.g. a replacement policy string,
        {"tool_output": ...}, {"prompt": ...}, or {"model": ...}.
    agent_fn : Callable, optional
        Optional custom agent runner function. If None, defaults to the built-in
        agent for the recorded agent_name (e.g. refund_agent).
    db : Session, optional
        SQLAlchemy database session.
    raise_on_error : bool, default True
        If True, re-raises UnrecordedCall or execution exceptions after marking
        the trace as FAILED and committing to DB.
    """
    session = db or SessionLocal()
    owns_session = db is None

    try:
        orig_trace = session.query(Trace).filter(Trace.id == trace_id).first()
        if not orig_trace:
            raise ValueError(f"Trace #{trace_id} not found")

        orig_steps = (
            session.query(Step)
            .filter(Step.trace_id == orig_trace.id)
            .order_by(Step.index)
            .all()
        )
        if not orig_steps:
            raise ValueError(f"Trace #{trace_id} has no steps to fork")

        if step_index < 1 or step_index > len(orig_steps):
            raise ValueError(
                f"Invalid step_index {step_index}. Trace has {len(orig_steps)} step(s)."
            )

        target_orig_step = next(s for s in orig_steps if s.index == step_index)

        # 1. Create the new fork Trace
        fork_trace_obj = Trace(
            run_id=f"fork-{orig_trace.run_id}-{uuid.uuid4().hex[:8]}",
            agent_name=orig_trace.agent_name,
            description=f"Fork of Trace #{orig_trace.id} at step {step_index}",
            meta={
                "forked_from_trace_id": orig_trace.id,
                "fork_step": step_index,
                "overrides": redact_sensitive(overrides),
                **(orig_trace.meta or {}),
            },
            status=TraceStatus.RECORDING,
            run_type=TraceRunType.FORK,
            agent_version=orig_trace.agent_version,
            parent_trace_id=orig_trace.id,
            fork_step=step_index,
            created_at=datetime.datetime.now(datetime.UTC),
        )
        session.add(fork_trace_obj)
        session.flush()

        # 2. Steps before step_index are replayed from recording (copied, source=replayed)
        for s in orig_steps:
            if s.index < step_index:
                copied_step = Step(
                    trace_id=fork_trace_obj.id,
                    index=s.index,
                    node_name=s.node_name,
                    state_before=dict(s.state_before or {}),
                    state_after=dict(s.state_after or {}),
                    model=s.model,
                    input_tokens=s.input_tokens,
                    output_tokens=s.output_tokens,
                    cost=s.cost,
                    latency_ms=s.latency_ms,
                )
                session.add(copied_step)
                session.flush()

                for ev in sorted(s.events, key=lambda e: e.seq):
                    copied_ev = Event(
                        step_id=copied_step.id,
                        kind=ev.kind,
                        seq=ev.seq,
                        input_hash=ev.input_hash,
                        output=ev.output,
                        raw_input=ev.raw_input,
                        raw_output=ev.raw_output,
                        error=ev.error,
                        source=EventSource.REPLAYED,
                    )
                    session.add(copied_ev)
                session.flush()

        # 3. Restore graph state before step_index
        restored_state: dict[str, Any] = {}
        for s in orig_steps:
            if s.index < step_index:
                restored_state.update(s.state_before or {})
                restored_state.update(s.state_after or {})
        restored_state.update(target_orig_step.state_before or {})

        # 4. Initialize Recorder attached to this fork trace
        recorder = Recorder(
            db=session,
            run_id=fork_trace_obj.run_id,
            agent_name=fork_trace_obj.agent_name,
            description=fork_trace_obj.description,
            meta=fork_trace_obj.meta,
            agent_version=fork_trace_obj.agent_version,
            run_type=TraceRunType.FORK,
            parent_trace_id=orig_trace.id,
            fork_step=step_index,
        )

        # 5. Run forward from restored state under sandbox safety mode
        try:
            with sandbox_mode():
                if agent_fn is not None:
                    agent_fn(
                        rec=recorder,
                        start_step=step_index,
                        restored_state=restored_state,
                        overrides=overrides,
                    )
                else:
                    reg = get_agent(orig_trace.agent_name)
                    if reg.runner_from_step is None:
                        raise ValueError(
                            f"Agent '{orig_trace.agent_name}' has no runner_from_step registered. "
                            f"Please register it with agentreplay.register_agent."
                        )
                    stale_policy = (
                        orig_trace.meta.get("stale_policy", False)
                        if orig_trace.meta
                        else False
                    )
                    policy_override = (
                        orig_trace.meta.get("policy_days_override")
                        if orig_trace.meta
                        else None
                    )
                    reg.runner_from_step(
                        rec=recorder,
                        start_step=step_index,
                        restored_state=restored_state,
                        overrides=overrides,
                        stale_policy=stale_policy,
                        policy_days_override=policy_override,
                    )

            session.commit()
            return fork_trace_obj

        except UnrecordedCall as exc:
            # 3. UNRECORDED_CALL policy: if a forked run makes a call that has no recording
            # and no user-supplied mock, stop the fork, mark trace status=failed with
            # a clear UNRECORDED_CALL error, and list the call. Never guess.
            err_msg = (
                f"UNRECORDED_CALL: No recorded event or user mock for node='{exc.node}', "
                f"kind='{exc.kind}', input_hash='{exc.input_hash}'"
            )
            fork_trace_obj.status = TraceStatus.FAILED
            fork_trace_obj.description = err_msg

            if recorder._current_step is not None:
                recorder.record_event(
                    kind=(
                        EventKind.TOOL_CALL
                        if exc.kind == EventKind.TOOL_CALL.value
                        else EventKind.LLM_CALL
                    ),
                    raw_input={"node": exc.node, "kind": exc.kind, "input_hash": exc.input_hash},
                    output={"error": err_msg},
                    error=err_msg,
                    source=EventSource.LIVE,
                )
                recorder.end_step(state_after=recorder._current_step.state_before or {})
            else:
                # Add a step for the failed unrecorded call
                step = recorder.start_step(exc.node, state_before=restored_state)
                recorder.record_event(
                    kind=(
                        EventKind.TOOL_CALL
                        if exc.kind == EventKind.TOOL_CALL.value
                        else EventKind.LLM_CALL
                    ),
                    raw_input={"node": exc.node, "kind": exc.kind, "input_hash": exc.input_hash},
                    output={"error": err_msg},
                    error=err_msg,
                    source=EventSource.LIVE,
                )
                recorder.end_step(state_after=step.state_before or {})

            session.flush()
            session.commit()
            if raise_on_error:
                raise
            return fork_trace_obj

        except Exception as exc:
            fork_trace_obj.status = TraceStatus.FAILED
            fork_trace_obj.description = f"FAILED: {exc}"
            session.flush()
            session.commit()
            if raise_on_error:
                raise
            return fork_trace_obj

    finally:
        if owns_session:
            session.close()
