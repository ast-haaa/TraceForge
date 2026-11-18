"""Event-by-event diff engine for comparing traces.

Compares two traces (typically an original trace and a forked or replayed trace)
event by event and returns:
- The first step where outputs diverge (`first_divergent_step`, `first_divergent_node`).
- A complete list of changed events (`changed_events`).
- Aligned step-by-step diff records for UI rendering (`steps`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Event, Step, Trace

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EventDiff:
    """Comparison record for a single event occurrence."""

    step_index: int
    node_name: str
    seq: int
    kind: str
    original_event_id: int | None
    fork_event_id: int | None
    original_input_hash: str | None
    fork_input_hash: str | None
    original_raw_input: str | None
    fork_raw_input: str | None
    original_output: Any
    fork_output: Any
    original_raw_output: str | None
    fork_raw_output: str | None
    original_source: str | None
    fork_source: str | None
    diverged: bool
    reason: str | None = None


@dataclass
class StepDiff:
    """Aligned step comparison record for UI rendering."""

    step_index: int
    node_name: str
    original_step: Step | None
    fork_step: Step | None
    diverged: bool
    events: list[EventDiff] = field(default_factory=list)


@dataclass
class DiffResult:
    """Complete diff result between two traces."""

    original_trace: Trace
    fork_trace: Trace
    first_divergent_step: int | None
    first_divergent_node: str | None
    changed_events: list[EventDiff]
    steps: list[StepDiff]
    diverged: bool

    def summary(self) -> str:
        if not self.diverged:
            return "Traces are identical (no divergence detected)."
        return (
            f"Divergence detected at Step {self.first_divergent_step} "
            f"('{self.first_divergent_node}'). {len(self.changed_events)} event(s) changed."
        )


# ---------------------------------------------------------------------------
# Diff function
# ---------------------------------------------------------------------------


def diff_traces(
    original_trace_id: int,
    fork_trace_id: int,
    db: Session | None = None,
) -> DiffResult:
    """Compare two traces event by event and return first divergent step + changed events.

    Parameters
    ----------
    original_trace_id : int
        ID of the baseline or original trace.
    fork_trace_id : int
        ID of the forked or comparison trace.
    db : Session, optional
        SQLAlchemy session.

    Returns
    -------
    DiffResult
        Detailed diff report.
    """
    session = db or SessionLocal()
    owns_session = db is None

    try:
        orig_trace = session.query(Trace).filter(Trace.id == original_trace_id).first()
        if not orig_trace:
            raise ValueError(f"Original trace #{original_trace_id} not found")

        fork_trace = session.query(Trace).filter(Trace.id == fork_trace_id).first()
        if not fork_trace:
            raise ValueError(f"Fork trace #{fork_trace_id} not found")

        orig_steps = (
            session.query(Step)
            .filter(Step.trace_id == orig_trace.id)
            .order_by(Step.index)
            .all()
        )
        fork_steps = (
            session.query(Step)
            .filter(Step.trace_id == fork_trace.id)
            .order_by(Step.index)
            .all()
        )

        orig_steps_by_index: dict[int, Step] = {s.index: s for s in orig_steps}
        fork_steps_by_index: dict[int, Step] = {s.index: s for s in fork_steps}

        all_indices = sorted(set(orig_steps_by_index.keys()) | set(fork_steps_by_index.keys()))

        first_divergent_step: int | None = None
        first_divergent_node: str | None = None
        changed_events: list[EventDiff] = []
        step_diffs: list[StepDiff] = []

        for step_idx in all_indices:
            os = orig_steps_by_index.get(step_idx)
            fs = fork_steps_by_index.get(step_idx)

            node_name = os.node_name if os else (fs.node_name if fs else f"step_{step_idx}")

            os_events = (
                session.query(Event).filter(Event.step_id == os.id).order_by(Event.seq).all()
                if os
                else []
            )
            fs_events = (
                session.query(Event).filter(Event.step_id == fs.id).order_by(Event.seq).all()
                if fs
                else []
            )

            max_events = max(len(os_events), len(fs_events))
            step_diverged = False
            step_event_diffs: list[EventDiff] = []

            for i in range(max_events):
                oe = os_events[i] if i < len(os_events) else None
                fe = fs_events[i] if i < len(fs_events) else None

                seq = oe.seq if oe else (fe.seq if fe else i + 1)
                kind = oe.kind.value if oe else (fe.kind.value if fe else "unknown")

                diverged = False
                reason = None

                if oe is None and fe is not None:
                    diverged = True
                    reason = "Event added in fork trace"
                elif oe is not None and fe is None:
                    diverged = True
                    reason = "Event missing in fork trace"
                elif oe is not None and fe is not None:
                    if oe.output != fe.output:
                        diverged = True
                        reason = "Output values differ"
                    elif oe.input_hash != fe.input_hash:
                        diverged = True
                        reason = "Input hash differs"
                    elif oe.error != fe.error:
                        diverged = True
                        reason = "Error values differ"

                ev_diff = EventDiff(
                    step_index=step_idx,
                    node_name=node_name,
                    seq=seq,
                    kind=kind,
                    original_event_id=oe.id if oe else None,
                    fork_event_id=fe.id if fe else None,
                    original_input_hash=oe.input_hash if oe else None,
                    fork_input_hash=fe.input_hash if fe else None,
                    original_raw_input=oe.raw_input if oe else None,
                    fork_raw_input=fe.raw_input if fe else None,
                    original_output=oe.output if oe else None,
                    fork_output=fe.output if fe else None,
                    original_raw_output=oe.raw_output if oe else None,
                    fork_raw_output=fe.raw_output if fe else None,
                    original_source=oe.source.value if oe else None,
                    fork_source=fe.source.value if fe else None,
                    diverged=diverged,
                    reason=reason,
                )
                step_event_diffs.append(ev_diff)

                if diverged:
                    step_diverged = True
                    changed_events.append(ev_diff)

            # Check if steps themselves were mismatched (e.g. step missing in one trace)
            if (os is None and fs is not None) or (os is not None and fs is None):
                step_diverged = True

            if step_diverged and first_divergent_step is None:
                first_divergent_step = step_idx
                first_divergent_node = node_name

            step_diffs.append(
                StepDiff(
                    step_index=step_idx,
                    node_name=node_name,
                    original_step=os,
                    fork_step=fs,
                    diverged=step_diverged,
                    events=step_event_diffs,
                )
            )

        return DiffResult(
            original_trace=orig_trace,
            fork_trace=fork_trace,
            first_divergent_step=first_divergent_step,
            first_divergent_node=first_divergent_node,
            changed_events=changed_events,
            steps=step_diffs,
            diverged=len(changed_events) > 0 or first_divergent_step is not None,
        )

    finally:
        if owns_session:
            session.close()
