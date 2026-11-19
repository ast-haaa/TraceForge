# scripts/export_trace.py
"""Export a real LLM trace to tests/fixtures/golden_llm_trace.json for offline replay tests.

Applies sensitive data redaction and strictly asserts that no API keys or credentials
appear in the exported fixture file.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import dotenv

from agentreplay.db import SessionLocal, init_db
from agentreplay.models import Step, Trace
from agentreplay.redact import redact_sensitive
from demo_agent.agent import run_refund_agent
from demo_agent.llm import clean_env_var
from demo_agent.tools import reset_ledger

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
_GOLDEN_PATH = _FIXTURES_DIR / "golden_llm_trace.json"


def scan_for_secrets(data_str: str, api_key: str = "") -> None:
    """Strictly assert no API keys, credentials, or auth headers exist in the text."""
    if api_key and api_key in data_str:
        raise ValueError("CRITICAL: LLM_API_KEY detected in trace data before export!")

    forbidden_patterns = ["Bearer ", "gsk_", "sk-", "Authorization:"]
    for pat in forbidden_patterns:
        if pat.lower() in data_str.lower():
            raise ValueError(f"CRITICAL: Potential credential pattern '{pat}' detected in trace export!")


def export_trace_to_golden(trace_id: int | None = None) -> pathlib.Path:
    dotenv.load_dotenv()
    init_db()
    db = SessionLocal()
    api_key = clean_env_var(os.environ.get("LLM_API_KEY"), "")

    # If no trace_id provided, look for an existing real LLM trace, or run one
    trace: Trace | None = None
    if trace_id is not None:
        trace = db.query(Trace).filter(Trace.id == trace_id).first()

    if trace is None:
        # Search for a recorded trace with an llm_call event
        candidates = (
            db.query(Trace)
            .order_by(Trace.id.desc())
            .limit(20)
            .all()
        )
        for c in candidates:
            has_llm = (
                db.query(Step)
                .join(Step.events)
                .filter(Step.trace_id == c.id, Step.node_name == "llm_decide")
                .first()
            )
            if has_llm:
                trace = c
                break

    if trace is None:
        print("[INFO] No existing real trace found. Running one now...")
        reset_ledger()
        run_refund_agent(
            customer_id="C001",
            order_id="O103",
            stale_policy=True,
            description="Export golden trace run",
        )
        trace = (
            db.query(Trace)
            .filter_by(description="Export golden trace run")
            .order_by(Trace.id.desc())
            .first()
        )

    assert trace is not None, "Failed to locate or record a trace for export"

    # Fetch steps and events
    steps = (
        db.query(Step)
        .filter(Step.trace_id == trace.id)
        .order_by(Step.index)
        .all()
    )

    steps_data = []
    for s in steps:
        events_data = []
        for e in sorted(s.events, key=lambda x: x.seq):
            events_data.append({
                "seq": e.seq,
                "kind": e.kind.value if hasattr(e.kind, "value") else str(e.kind),
                "input_hash": e.input_hash,
                "raw_input": redact_sensitive(e.raw_input),
                "output": redact_sensitive(e.output),
                "raw_output": redact_sensitive(e.raw_output),
                "error": e.error,
                "source": e.source.value if hasattr(e.source, "value") else str(e.source),
            })
        steps_data.append({
            "index": s.index,
            "node_name": s.node_name,
            "state_before": redact_sensitive(s.state_before),
            "state_after": redact_sensitive(s.state_after),
            "model": s.model,
            "input_tokens": s.input_tokens,
            "output_tokens": s.output_tokens,
            "cost": s.cost,
            "latency_ms": s.latency_ms,
            "events": events_data,
        })

    trace_data = {
        "run_id": trace.run_id,
        "agent_name": trace.agent_name,
        "description": trace.description,
        "meta": redact_sensitive(trace.meta),
        "status": trace.status.value if hasattr(trace.status, "value") else str(trace.status),
        "run_type": trace.run_type.value if hasattr(trace.run_type, "value") else str(trace.run_type),
        "final_output": redact_sensitive(trace.final_output),
        "agent_version": trace.agent_version,
        "steps": steps_data,
    }

    # Redact everything
    redacted_payload = redact_sensitive(trace_data)
    json_str = json.dumps(redacted_payload, indent=2, default=str)

    # Secret scanning
    scan_for_secrets(json_str, api_key=api_key)

    _GOLDEN_PATH.write_text(json_str, encoding="utf-8")
    print(f"[SUCCESS] Exported golden trace #{trace.id} to {_GOLDEN_PATH}")
    db.close()
    return _GOLDEN_PATH


if __name__ == "__main__":
    t_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    export_trace_to_golden(t_id)
