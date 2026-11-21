"""AgentReplay: Debugger and regression-testing platform for LLM agents."""

from .diff import DiffResult, EventDiff, StepDiff, diff_traces
from .fork import fork_trace, sandbox_mode
from .hash import compute_input_hash, strip_volatile_fields
from .integrations.langgraph import AgentReplayHandler
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
from .registry import (
    AgentRegistration,
    UnknownAgentError,
    get_agent,
    register_agent,
)
from .replay import (
    FidelityReport,
    ReplayContext,
    ReplayEngine,
    ReplayResult,
    UnrecordedCall,
    calculate_replay_fidelity,
)
from .sdk import Recorder, record_trace
from .test_gen import generate_test, sanitize_paths, sanitize_trace_id, validate_test

__all__ = [
    "AgentRegistration",
    "AgentReplayHandler",
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
    "UnknownAgentError",
    "UnrecordedCall",
    "calculate_replay_fidelity",
    "compute_input_hash",
    "diff_traces",
    "fork_trace",
    "generate_test",
    "get_agent",
    "record_trace",
    "redact_sensitive",
    "register_agent",
    "sandbox_mode",
    "sanitize_paths",
    "sanitize_trace_id",
    "strip_volatile_fields",
    "validate_test",
]
