# agentreplay/models.py
"""SQLAlchemy ORM models for AgentReplay.
All timestamps are stored as UTC-aware datetimes.
"""
import datetime
import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TraceRunType(enum.StrEnum):
    ORIGINAL = "original"
    REPLAY = "replay"
    FORK = "fork"


class TraceStatus(enum.StrEnum):
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"


class EventKind(enum.StrEnum):
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    STATE = "state"


class EventSource(enum.StrEnum):
    RECORDED = "recorded"
    REPLAYED = "replayed"
    LIVE = "live"
    MOCKED = "mocked"


# ---------------------------------------------------------------------------
# Core tables
# ---------------------------------------------------------------------------


class Trace(Base):
    """One execution of a LangGraph agent (original, replay, or fork)."""

    __tablename__ = "traces"

    id = Column(Integer, primary_key=True)
    # idempotency key supplied by SDK – retried ingest calls are no-ops
    run_id = Column(String, unique=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.UTC),
        index=True,
    )

    agent_name = Column(String, nullable=False)
    description = Column(Text)
    meta = Column(JSON, default=dict)  # renamed from `metadata` (reserved by SQLAlchemy)

    status = Column(Enum(TraceStatus), nullable=False, default=TraceStatus.RECORDING)
    run_type = Column(Enum(TraceRunType), nullable=False, default=TraceRunType.ORIGINAL)
    final_output = Column(JSON)
    agent_version = Column(String)

    # Fork lineage – NULL for original runs
    parent_trace_id = Column(Integer, ForeignKey("traces.id", ondelete="SET NULL"))
    fork_step = Column(Integer)

    parent = relationship("Trace", remote_side=[id], backref="forks")
    steps = relationship(
        "Step",
        back_populates="trace",
        cascade="all, delete-orphan",
        order_by="Step.index",
    )


class Step(Base):
    """Logical step (node execution) within a trace."""

    __tablename__ = "steps"

    id = Column(Integer, primary_key=True)
    trace_id = Column(Integer, ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    index = Column(Integer, nullable=False)  # UI order; NOT used for deterministic replay
    node_name = Column(String, nullable=False)

    # Graph state snapshots – required for restoring state at fork point
    state_before = Column(JSON, default=dict)
    state_after = Column(JSON, default=dict)

    # Aggregated metrics derived from child Events
    model = Column(String)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    latency_ms = Column(Float)

    trace = relationship("Trace", back_populates="steps")

    __table_args__ = (UniqueConstraint("trace_id", "index", name="uq_trace_step_index"),)


class Event(Base):
    """A single recorded event inside a Step.

    Deterministic replay matches on (node_name, input_hash, occurrence) where
    *occurrence* is the count of earlier Events with the same (step_id, input_hash).
    seq is an explicit ordering column so the DB index is cheap.
    """

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    step_id = Column(Integer, ForeignKey("steps.id", ondelete="CASCADE"), index=True)

    kind = Column(Enum(EventKind), nullable=False)
    seq = Column(Integer, nullable=False)  # order within the step
    input_hash = Column(String, nullable=False, index=True)
    output = Column(JSON, nullable=False)
    raw_input = Column(Text)   # human-readable / debug only
    raw_output = Column(Text)
    error = Column(Text, nullable=True)  # captured exception text, if any
    source = Column(Enum(EventSource), nullable=False, default=EventSource.RECORDED)

    step = relationship("Step", backref="events")

    __table_args__ = (UniqueConstraint("step_id", "seq", name="uq_event_step_seq"),)


# ---------------------------------------------------------------------------
# Evaluation labels  (benchmark metadata – kept separate from normal traces)
# investigate.py must NEVER import or query this table (enforced by test)
# ---------------------------------------------------------------------------


class EvalLabel(Base):
    """Metadata attached to fault-injection benchmark runs."""

    __tablename__ = "eval_labels"

    id = Column(Integer, primary_key=True)
    trace_id = Column(Integer, ForeignKey("traces.id", ondelete="CASCADE"), unique=True)
    injected_fault = Column(String)
    ground_truth_step = Column(Integer)

    trace = relationship("Trace", backref="eval_label")


# ---------------------------------------------------------------------------
# Generated regression tests
# ---------------------------------------------------------------------------


class GeneratedTest(Base):
    __tablename__ = "generated_tests"

    id = Column(Integer, primary_key=True)
    trace_id = Column(Integer, ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    description = Column(Text, nullable=False)
    test_code = Column(Text, nullable=False)
    file_path = Column(String)
    # Boolean pass/fail flags
    passes_on_fixed = Column(Boolean, nullable=True)
    fails_on_buggy = Column(Boolean, nullable=True)
    fixed_agent_version = Column(String)
    pytest_output = Column(Text)
    status = Column(String, default="pending")  # valid, rejected, pending
    rejection_reason = Column(Text)

    trace = relationship("Trace", backref="generated_tests")


# ---------------------------------------------------------------------------
# LLM investigation results  (must NEVER read EvalLabel)
# ---------------------------------------------------------------------------


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(Integer, primary_key=True)
    trace_id = Column(Integer, ForeignKey("traces.id", ondelete="CASCADE"), index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    # list of {"step": int, "hypothesis": str, "confidence": float}
    hypotheses = Column(JSON, nullable=False)
    ranked_steps = Column(JSON, nullable=False)
    raw_llm_output = Column(Text)

    trace = relationship("Trace", backref="investigations")
