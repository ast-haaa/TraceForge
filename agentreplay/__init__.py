"""AgentReplay: Debugger and regression-testing platform for LLM agents."""

from .diff import DiffResult, EventDiff, StepDiff, diff_traces
from .fork import fork_trace, sandbox_mode
from .hash import compute_input_hash, strip_volatile_fields
from .models import (
    Event,
    EventKind,
    EventSource,
    GeneratedTest,
    Step,
    Trace,
    TraceRunType,
    TraceStatus,
)
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
from .test_gen import generate_test, sanitize_trace_id, validate_test

__all__ = [
    "DiffResult",
    "Event",
    "EventDiff",
    "EventKind",
    "EventSource",
    "FidelityReport",
    "GeneratedTest",
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
    "generate_test",
    "record_trace",
    "redact_sensitive",
    "sandbox_mode",
    "sanitize_trace_id",
    "strip_volatile_fields",
    "validate_test",
]
