"""AgentReplay: Debugger and regression-testing platform for LLM agents."""

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
    "Event",
    "EventKind",
    "EventSource",
    "FidelityReport",
    "Recorder",
    "ReplayContext",
    "ReplayEngine",
    "ReplayResult",
    "Step",
    "Trace",
    "TraceRunType",
    "TraceStatus",
    "UnrecordedCall",
    "calculate_replay_fidelity",
    "compute_input_hash",
    "record_trace",
    "redact_sensitive",
    "strip_volatile_fields",
]
