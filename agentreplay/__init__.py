"""AgentReplay: Debugger and regression-testing platform for LLM agents."""

from .diff import DiffResult, EventDiff, StepDiff, diff_traces
from .fork import fork_trace, sandbox_mode
from .hash import compute_input_hash, strip_volatile_fields
from .models import Event, EventKind, EventSource, Step, Trace, TraceRunType, TraceStatus
from .redact import redact_sensitive
from .replay import (
    FidelityReport,
    ReplayContext,
    ReplayEngine,
    ReplayResult,
    UnrecordedCall,
    calculate_replay_fidelity,
)
from .sdk import Recorder, record_trace

__all__ = [
    "DiffResult",
    "Event",
    "EventDiff",
    "EventKind",
    "EventSource",
    "FidelityReport",
    "Recorder",
    "ReplayContext",
    "ReplayEngine",
    "ReplayResult",
    "Step",
    "StepDiff",
    "Trace",
    "TraceRunType",
    "TraceStatus",
    "UnrecordedCall",
    "calculate_replay_fidelity",
    "compute_input_hash",
    "diff_traces",
    "fork_trace",
    "record_trace",
    "redact_sensitive",
    "sandbox_mode",
    "strip_volatile_fields",
]
