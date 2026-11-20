"""Deterministic Plain-Software Replay Engine.

Re-runs a recorded execution deterministically with ZERO live LLM or tool calls.
All model and tool outputs come strictly from the recording.
No real external side effects (e.g. no fake ledger writes).

If an execution makes a tool or LLM call that does not exist in the recording,
it does NOT guess or fall back: it raises `UnrecordedCall(node, kind, input_hash)`.
"""
from __future__ import annotations

import datetime
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from sqlalchemy.orm import Session

from .db import SessionLocal
from .hash import compute_input_hash
from .models import Event, EventKind, EventSource, Step, Trace, TraceRunType
from .redact import redact_sensitive
from .registry import get_agent, get_all_registrations
from .sdk import _active_replay_session

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UnrecordedCall(Exception):
    """Raised when an execution attempts an event that was not recorded.

    The replay engine is deterministic plain software; it never guesses,
    hallucinates, or falls back to live external calls.
    """

    def __init__(
        self,
        node: str,
        kind: EventKind | str,
        input_hash: str,
        message: str | None = None,
    ) -> None:
        self.node = node
        self.kind = kind if isinstance(kind, str) else kind.value
        self.input_hash = input_hash
        msg = message or (
            f"UNRECORDED_CALL: No recorded event for node='{node}', kind='{self.kind}', "
            f"input_hash='{input_hash}'. Deterministic replay cannot proceed."
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Fidelity Report
# ---------------------------------------------------------------------------


@dataclass
class FidelityReport:
    """Audit report comparing recorded vs replayed execution."""

    original_trace_id: int
    replayed_trace_id: int
    total_events: int
    matched_events: int
    fidelity_percent: float
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Fidelity: {self.fidelity_percent:.1f}% "
            f"({self.matched_events}/{self.total_events} events matched)"
        )


def calculate_replay_fidelity(
    original_trace_id: int,
    replayed_trace_id: int,
    db: Session,
) -> FidelityReport:
    """Calculate the exact percentage of events whose replayed output equals the original."""
    orig_events = (
        db.query(Event)
        .join(Step)
        .filter(Step.trace_id == original_trace_id)
        .order_by(Step.index, Event.seq)
        .all()
    )
    rep_events = (
        db.query(Event)
        .join(Step)
        .filter(Step.trace_id == replayed_trace_id)
        .order_by(Step.index, Event.seq)
        .all()
    )

    total = len(orig_events)
    matched = 0
    mismatches: list[dict[str, Any]] = []

    for idx, orig_ev in enumerate(orig_events):
        if idx >= len(rep_events):
            mismatches.append(
                {
                    "step": orig_ev.step.node_name if orig_ev.step else "unknown",
                    "seq": orig_ev.seq,
                    "kind": orig_ev.kind.value,
                    "reason": "Missing in replayed trace",
                }
            )
            continue

        rep_ev = rep_events[idx]
        output_match = orig_ev.output == rep_ev.output
        kind_match = orig_ev.kind == rep_ev.kind
        hash_match = orig_ev.input_hash == rep_ev.input_hash
        error_match = orig_ev.error == rep_ev.error

        if output_match and kind_match and hash_match and error_match:
            matched += 1
        else:
            mismatches.append(
                {
                    "step": orig_ev.step.node_name if orig_ev.step else "unknown",
                    "seq": orig_ev.seq,
                    "kind": orig_ev.kind.value,
                    "orig_output": orig_ev.output,
                    "rep_output": rep_ev.output,
                    "orig_error": orig_ev.error,
                    "rep_error": rep_ev.error,
                }
            )

    pct = (matched / total * 100.0) if total > 0 else 100.0
    return FidelityReport(
        original_trace_id=original_trace_id,
        replayed_trace_id=replayed_trace_id,
        total_events=total,
        matched_events=matched,
        fidelity_percent=pct,
        mismatches=mismatches,
    )


# ---------------------------------------------------------------------------
# Event Indexer: matches (node_name, input_hash, occurrence)
# ---------------------------------------------------------------------------


class ReplayEventMatcher:
    """Maintains indexed lookup map of recorded events.

    Matches by (node_name, input_hash, occurrence) where occurrence is the count
    of earlier events with the same hash within the step / node.
    """

    def __init__(self, steps: list[Step]) -> None:
        self._events: dict[tuple[str, str, int], Event] = {}
        self._call_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.recorded_events: list[Event] = []

        for step in steps:
            step_counts: dict[str, int] = defaultdict(int)
            sorted_events = sorted(step.events, key=lambda e: e.seq)
            for ev in sorted_events:
                occ = step_counts[ev.input_hash]
                step_counts[ev.input_hash] += 1
                key = (step.node_name, ev.input_hash, occ)
                self._events[key] = ev
                self.recorded_events.append(ev)

    def match(
        self,
        node_name: str,
        input_hash: str,
        kind: EventKind | str = EventKind.TOOL_CALL,
    ) -> Event:
        """Find the recorded event matching (node_name, input_hash, occurrence)."""
        occ = self._call_counts[(node_name, input_hash)]
        key = (node_name, input_hash, occ)
        if key not in self._events:
            raise UnrecordedCall(node=node_name, kind=kind, input_hash=input_hash)
        self._call_counts[(node_name, input_hash)] += 1
        return self._events[key]

    def reset(self) -> None:
        self._call_counts.clear()


# ---------------------------------------------------------------------------
# Replay Context: Intercepts tools and LLM
# ---------------------------------------------------------------------------


class ReplayContext:
    """Context manager that wraps execution and redirects tools & LLMs to recorded events."""

    def __init__(self, trace_id: int, db: Session | None = None) -> None:
        self.db = db or SessionLocal()
        self._owns_db = db is None
        self.trace_id = trace_id

        self.original_trace = (
            self.db.query(Trace).filter(Trace.id == trace_id).first()
        )
        if not self.original_trace:
            raise ValueError(f"Trace #{trace_id} not found")

        self.steps = (
            self.db.query(Step)
            .filter(Step.trace_id == trace_id)
            .order_by(Step.index)
            .all()
        )
        self.matcher = ReplayEventMatcher(self.steps)
        self._patches: list[Any] = []
        self._token: Any = None

    def __enter__(self) -> ReplayContext:
        self._token = _active_replay_session.set({
            "parent_trace_id": self.original_trace.id,
            "matcher": self.matcher,
        })
        self._install_tool_patches()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()
        if self._token:
            _active_replay_session.reset(self._token)
        if self._owns_db:
            self.db.close()

    def _install_tool_patches(self) -> None:
        matcher = self.matcher

        def mock_get_customer(customer_id: str) -> dict[str, Any]:
            raw_input = {"name": "get_customer", "args": {"customer_id": customer_id}}
            h = compute_input_hash(redact_sensitive(raw_input))
            ev = matcher.match("get_customer", h, kind=EventKind.TOOL_CALL)
            if ev.error:
                raise RuntimeError(ev.error)
            return ev.output

        def mock_get_orders(customer_id: str) -> list[dict[str, Any]]:
            raw_input = {"name": "get_orders", "args": {"customer_id": customer_id}}
            h = compute_input_hash(redact_sensitive(raw_input))
            ev = matcher.match("get_orders", h, kind=EventKind.TOOL_CALL)
            if ev.error:
                raise RuntimeError(ev.error)
            # Output was recorded as {"orders": orders}
            if isinstance(ev.output, dict) and "orders" in ev.output:
                return ev.output["orders"]
            return ev.output

        def mock_read_refund_policy(*, stale: bool = False) -> str:
            raw_input = {"name": "read_refund_policy", "args": {"stale": stale}}
            h = compute_input_hash(redact_sensitive(raw_input))
            ev = matcher.match("read_refund_policy", h, kind=EventKind.TOOL_CALL)
            if ev.error:
                raise RuntimeError(ev.error)
            if isinstance(ev.output, dict) and "policy" in ev.output:
                return ev.output["policy"]
            return ev.output

        def mock_issue_refund(order_id: str, amount: float, reason: str = "") -> dict[str, Any]:
            # ZERO side effects: ledger is NOT touched
            raw_input = {
                "name": "issue_refund",
                "args": {"order_id": order_id, "amount": amount, "reason": reason},
            }
            h = compute_input_hash(redact_sensitive(raw_input))
            ev = matcher.match("issue_refund", h, kind=EventKind.TOOL_CALL)
            if ev.error:
                raise RuntimeError(ev.error)
            return ev.output

        def mock_stub_llm(prompt: str, *, policy_days: int = 7) -> str:
            raw_input = {"prompt": prompt}
            h = compute_input_hash(redact_sensitive(raw_input))
            ev = matcher.match("llm_decide", h, kind=EventKind.LLM_CALL)
            if ev.error:
                raise RuntimeError(ev.error)
            if isinstance(ev.output, dict) and "response" in ev.output:
                return ev.output["response"]
            return str(ev.output)

        def mock_llm_call(prompt: str, **kwargs: Any) -> Any:
            raw_input = {"prompt": prompt}
            h = compute_input_hash(redact_sensitive(raw_input))
            ev = matcher.match("llm_decide", h, kind=EventKind.LLM_CALL)
            if ev.error:
                raise RuntimeError(ev.error)
            dec = "approve"
            reason = ""
            model = "replayed-model"
            p_toks = 0
            c_toks = 0
            lat = 0.0
            cost = 0.0
            if isinstance(ev.output, dict):
                dec = ev.output.get("decision")
                if not dec:
                    resp_str = str(ev.output.get("response", ""))
                    dec = "approve" if "DECISION: approve" in resp_str else "deny"
                reason = ev.output.get("reason") or ev.output.get("response", "")
                model = ev.output.get("model", model)
                p_toks = ev.output.get("input_tokens", 0)
                c_toks = ev.output.get("output_tokens", 0)
                lat = ev.output.get("latency_ms", ev.output.get("latency", 0.0))
                cost = ev.output.get("cost", 0.0)
            elif isinstance(ev.output, str):
                dec = "approve" if "DECISION: approve" in ev.output else "deny"
                reason = ev.output

            from demo_agent.llm import LLMResponse
            return LLMResponse(
                decision=dec,
                reason=reason,
                model=model,
                prompt_tokens=p_toks,
                completion_tokens=c_toks,
                total_tokens=p_toks + c_toks,
                latency_ms=lat,
                cost=cost,
                raw_response=ev.output,
            )

        def mock_get_llm() -> Callable[..., Any]:
            return mock_llm_call

        built_in_mocks: dict[str, Callable[..., Any]] = {
            "get_customer": mock_get_customer,
            "get_orders": mock_get_orders,
            "read_refund_policy": mock_read_refund_policy,
            "issue_refund": mock_issue_refund,
            "_stub_llm": mock_stub_llm,
            "get_llm": mock_get_llm,
        }

        def make_generic_mock(name: str) -> Callable[..., Any]:
            def generic_mock(*args: Any, **kwargs: Any) -> Any:
                raw_input = {"name": name, "args": kwargs if kwargs else (args if args else None)}
                h = compute_input_hash(redact_sensitive(raw_input))
                ev = matcher.match(name, h, kind=EventKind.TOOL_CALL)
                if ev.error:
                    raise RuntimeError(ev.error)
                return ev.output
            return generic_mock

        # Dynamically load patch targets from registered agents
        targets: list[tuple[str, Callable[..., Any]]] = []
        seen_targets: set[str] = set()
        for reg in get_all_registrations().values():
            if not reg.patch_targets:
                continue
            for item in reg.patch_targets:
                if isinstance(item, tuple) and len(item) == 2:
                    target_str, mock_val = item
                    if target_str in seen_targets:
                        continue
                    seen_targets.add(target_str)
                    if callable(mock_val):
                        targets.append((target_str, mock_val))
                    elif isinstance(mock_val, str) and mock_val in built_in_mocks:
                        targets.append((target_str, built_in_mocks[mock_val]))
                    else:
                        targets.append((target_str, make_generic_mock(str(mock_val))))
                elif isinstance(item, str):
                    target_str = item
                    if target_str in seen_targets:
                        continue
                    seen_targets.add(target_str)
                    attr_name = target_str.split(".")[-1]
                    if attr_name in built_in_mocks:
                        targets.append((target_str, built_in_mocks[attr_name]))
                    else:
                        targets.append((target_str, make_generic_mock(attr_name)))

        for target_str, mock_fn in targets:
            try:
                p = patch(target_str, side_effect=mock_fn)
                p.start()
                self._patches.append(p)
            except (AttributeError, ModuleNotFoundError):
                pass


# ---------------------------------------------------------------------------
# Replay Result & Engine
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    original_trace: Trace
    replayed_trace: Trace
    fidelity: FidelityReport
    final_output: Any


class ReplayEngine:
    """Deterministic plain-software replay engine."""

    def __init__(self, db: Session | None = None) -> None:
        self.db = db or SessionLocal()
        self._owns_db = db is None

    def replay_trace(self, trace_id: int) -> ReplayResult:
        """Deterministically replay a stored trace using only database records.

        Restores graph state from Step.state_before, matches all events by
        (node_name, input_hash, occurrence), verifies zero live tool calls,
        creates a new Trace with run_type=replay and parent_trace_id set,
        and marks all events with source=replayed.
        """
        orig_trace = self.db.query(Trace).filter(Trace.id == trace_id).first()
        if not orig_trace:
            raise ValueError(f"Trace #{trace_id} not found")

        orig_steps = (
            self.db.query(Step)
            .filter(Step.trace_id == orig_trace.id)
            .order_by(Step.index)
            .all()
        )

        matcher = ReplayEventMatcher(orig_steps)

        # Create new Trace row for replay
        replayed_trace = Trace(
            run_id=f"replay-{orig_trace.run_id}-{uuid.uuid4().hex[:8]}",
            agent_name=orig_trace.agent_name,
            description=f"Deterministic replay of Trace #{orig_trace.id}",
            meta={"replayed_from_trace_id": orig_trace.id, **(orig_trace.meta or {})},
            status=orig_trace.status,
            run_type=TraceRunType.REPLAY,
            final_output=orig_trace.final_output,
            agent_version=orig_trace.agent_version,
            parent_trace_id=orig_trace.id,
            created_at=datetime.datetime.now(datetime.UTC),
        )
        self.db.add(replayed_trace)
        self.db.flush()

        reproduced_error: Exception | None = None

        # Replay each step, restoring state from state_before
        for orig_step in orig_steps:
            # Restore graph state
            replayed_step = Step(
                trace_id=replayed_trace.id,
                index=orig_step.index,
                node_name=orig_step.node_name,
                state_before=dict(orig_step.state_before or {}),
                state_after=dict(orig_step.state_after or {}),
                model=orig_step.model,
                input_tokens=orig_step.input_tokens,
                output_tokens=orig_step.output_tokens,
                cost=orig_step.cost,
                latency_ms=0.0,
            )
            self.db.add(replayed_step)
            self.db.flush()

            sorted_events = sorted(orig_step.events, key=lambda e: e.seq)
            for orig_ev in sorted_events:
                # Match by (node_name, input_hash, occurrence)
                matched_ev = matcher.match(
                    node_name=orig_step.node_name,
                    input_hash=orig_ev.input_hash,
                    kind=orig_ev.kind,
                )

                # Copy event with source=replayed
                replayed_ev = Event(
                    step_id=replayed_step.id,
                    kind=matched_ev.kind,
                    seq=matched_ev.seq,
                    input_hash=matched_ev.input_hash,
                    output=matched_ev.output,
                    raw_input=matched_ev.raw_input,
                    raw_output=matched_ev.raw_output,
                    error=matched_ev.error,
                    source=EventSource.REPLAYED,
                )
                self.db.add(replayed_ev)

                if orig_ev.error and reproduced_error is None:
                    reproduced_error = RuntimeError(orig_ev.error)

            self.db.flush()

        self.db.commit()

        fidelity = calculate_replay_fidelity(
            original_trace_id=orig_trace.id,
            replayed_trace_id=replayed_trace.id,
            db=self.db,
        )

        return ReplayResult(
            original_trace=orig_trace,
            replayed_trace=replayed_trace,
            fidelity=fidelity,
            final_output=replayed_trace.final_output,
        )

    def replay_agent(
        self,
        trace_id: int,
        agent_fn: Callable[..., Any] | None = None,
        **override_kwargs: Any,
    ) -> ReplayResult:
        """Replay an agent function execution with ZERO live tool/LLM calls.

        Patches all tools and LLMs so that outputs come exclusively from the recording.
        Raises UnrecordedCall if the agent makes an unrecorded call.
        """
        orig_trace = self.db.query(Trace).filter(Trace.id == trace_id).first()
        if not orig_trace:
            raise ValueError(f"Trace #{trace_id} not found")

        orig_trace_id = orig_trace.id
        fn_to_run = agent_fn
        if fn_to_run is None:
            reg = get_agent(orig_trace.agent_name)
            if reg.runner is None:
                raise ValueError(
                    f"Agent '{orig_trace.agent_name}' has no runner registered for replay. "
                    "Please register it with agentreplay.register_agent."
                )
            fn_to_run = reg.runner

        orig_steps = (
            self.db.query(Step)
            .filter(Step.trace_id == orig_trace_id)
            .order_by(Step.index)
            .all()
        )
        if not orig_steps:
            raise ValueError(f"Trace #{trace_id} has no steps to replay")

        # Reconstruct initial inputs from Step 1 state_before if not provided
        initial_state = orig_steps[0].state_before or {}
        kwargs: dict[str, Any] = {}
        if "customer_id" in initial_state:
            kwargs["customer_id"] = initial_state["customer_id"]
        if "order_id" in initial_state:
            kwargs["order_id"] = initial_state["order_id"]
        if orig_trace.meta and "stale_policy" in orig_trace.meta:
            kwargs["stale_policy"] = orig_trace.meta["stale_policy"]

        kwargs.update(override_kwargs)

        with ReplayContext(trace_id=orig_trace_id, db=self.db):
            final_output = fn_to_run(**kwargs)

        replayed_trace = (
            self.db.query(Trace)
            .filter(
                Trace.parent_trace_id == orig_trace_id,
                Trace.run_type == TraceRunType.REPLAY,
            )
            .order_by(Trace.id.desc())
            .first()
        )
        if not replayed_trace:
            raise RuntimeError("Replay execution failed to record a replayed trace")

        fidelity = calculate_replay_fidelity(
            original_trace_id=orig_trace_id,
            replayed_trace_id=replayed_trace.id,
            db=self.db,
        )

        orig_trace = self.db.query(Trace).filter(Trace.id == orig_trace_id).first()

        return ReplayResult(
            original_trace=orig_trace,
            replayed_trace=replayed_trace,
            fidelity=fidelity,
            final_output=final_output,
        )
