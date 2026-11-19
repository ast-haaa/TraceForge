# agentreplay/investigate.py
"""LLM Investigator for root-cause analysis of agent execution traces.

Analyzes trace steps and recorded events to diagnose the root-cause step
of a faulty agent run, ranking candidate steps with confidence and cited events.

LEAK PREVENTION:
- Sees ONLY step inputs, step outputs, tool calls, and LLM text.
- NEVER reads Trace.description, Trace.meta, or agent_version.
- NEVER imports, queries, or references EvalLabel.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from agentreplay.db import SessionLocal
from agentreplay.models import Event, Investigation, Step, Trace
from demo_agent.llm import clean_env_var

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas for Investigation Output
# ---------------------------------------------------------------------------


class StepHypothesis(BaseModel):
    """Ranked step hypothesis with evidence."""

    step_index: int = Field(description="1-based index of the execution step")
    node_name: str = Field(description="Name of the node (e.g. read_refund_policy)")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    hypothesis: str = Field(
        description="Detailed explanation of why this step was or was not the root cause"
    )
    cited_event_ids: list[int] = Field(
        default_factory=list,
        description="Database IDs of the events cited as evidence",
    )


class InvestigationResult(BaseModel):
    """Full structured diagnosis returned by the investigator."""

    summary: str = Field(
        description="High-level summary of the root cause and why the run failed"
    )
    ranked_steps: list[StepHypothesis] = Field(
        description="List of all executed steps ranked from most likely root cause to least likely"
    )


# ---------------------------------------------------------------------------
# Prompt Builder (strictly isolated from Trace metadata and EvalLabel)
# ---------------------------------------------------------------------------

USER_EXPECTATION = (
    "User Expectation & Policy Invariant:\n"
    "1. Refunds may ONLY be approved and issued for orders that are strictly within 7 days old (DAYS_OLD <= 7).\n"
    "2. Any order older than 7 days (DAYS_OLD > 7) must be DENIED, and refund must be SKIPPED.\n"
    "3. The refund must be processed for the actual customer requesting it, not credited to a different customer.\n"
    "4. If a tool fails, returns empty/null data, or returns an outdated policy (e.g. 30 days instead of 7 days), "
    "or if corrupted data is introduced, that step is faulty and caused the failure."
)


def build_investigator_prompt(
    steps: list[Step],
    events_by_step: dict[int, list[Event]],
    user_expectation: str = USER_EXPECTATION,
) -> str:
    """Construct the investigator prompt using ONLY steps and events.

    Strictly ignores Trace.description, Trace.meta, and Trace.agent_version.
    """
    lines: list[str] = [
        "You are an expert root-cause investigator for autonomous AI agent traces.",
        "Below is the execution trace of an agent that produced an incorrect or unexpected outcome.",
        "",
        "--- SYSTEM SPECIFICATION & USER EXPECTATION ---",
        user_expectation.strip(),
        "",
        "--- EXECUTION TRACE (STEPS & EVENTS) ---",
    ]

    for step in sorted(steps, key=lambda s: s.index):
        lines.append(f"\nStep {step.index} (Node: {step.node_name}):")
        if step.state_before:
            lines.append(f"  State Before: {json.dumps(step.state_before)}")

        step_events = events_by_step.get(step.id, [])
        for ev in sorted(step_events, key=lambda e: e.seq):
            kind_str = ev.kind.value if hasattr(ev.kind, "value") else str(ev.kind)
            lines.append(f"  - Event #{ev.id} [{kind_str}] (seq={ev.seq}):")
            if ev.raw_input:
                lines.append(f"      Input: {ev.raw_input}")
            if ev.output is not None:
                lines.append(f"      Output: {json.dumps(ev.output)}")
            if ev.error:
                lines.append(f"      Error: {ev.error}")

        if step.state_after:
            lines.append(f"  State After: {json.dumps(step.state_after)}")

    lines.extend(
        [
            "",
            "--- INSTRUCTIONS ---",
            "1. Carefully examine all steps and events in chronological order.",
            "2. Identify the ROOT CAUSE step where the first error, bad data, missing data, or flawed instruction was introduced.",
            "3. Rank all executed steps from most likely root cause (Rank #1) to least likely.",
            "4. For each step, provide:",
            "   - 'step_index': integer (1-based index)",
            "   - 'node_name': string",
            "   - 'confidence': float between 0.0 and 1.0",
            "   - 'hypothesis': concise explanation of why this step is or is not the root cause",
            "   - 'cited_event_ids': list of integer event IDs supporting your conclusion",
            "5. Return strictly a JSON object with keys 'summary' and 'ranked_steps'.",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline Deterministic Stub Investigator
# ---------------------------------------------------------------------------


def _stub_investigate(
    steps: list[Step], events_by_step: dict[int, list[Event]]
) -> tuple[InvestigationResult, str, int, int, float]:
    """Offline heuristic investigator for testing when LLM_PROVIDER=stub."""
    t0 = time.perf_counter()

    # Simple heuristic to identify suspect steps offline:
    # 1. Any step with an error
    # 2. Step 3 (read_refund_policy) if policy mentions 30 days or is empty
    # 3. Step 2 (get_orders) if order date is falsified
    # 4. Step 1 (get_customer) if customer mismatch
    # 5. Step 4 (llm_decide) if prompt mentions 30 days
    suspect_index = 4  # default suspect is decision step
    primary_event_id: int | None = None
    reason = "Decision step evaluated the inputs."

    for step in steps:
        step_events = events_by_step.get(step.id, [])
        for ev in step_events:
            ev_out = ev.output or {}
            ev_in = ev.raw_input or ""
            ev_in_str = json.dumps(ev_in) if isinstance(ev_in, dict) else str(ev_in)

            if ev.error:
                suspect_index = step.index
                primary_event_id = ev.id
                reason = f"Event encountered error: {ev.error}"
                break

            if step.node_name == "read_refund_policy":
                pol = ev_out.get("policy", "") if isinstance(ev_out, dict) else ""
                if not pol:
                    suspect_index = step.index
                    primary_event_id = ev.id
                    reason = "read_refund_policy returned empty/null policy text."
                    break
                if "30 days" in str(pol):
                    suspect_index = step.index
                    primary_event_id = ev.id
                    reason = "read_refund_policy returned stale 30-day policy text."
                    break

            if step.node_name == "llm_decide" and "30 days" in ev_in_str:
                suspect_index = step.index
                primary_event_id = ev.id
                reason = "llm_decide prompt incorrectly contained 30-day allowance."

    ranked: list[StepHypothesis] = []
    sorted_steps = sorted(steps, key=lambda s: s.index)

    # Place suspect first, then others
    suspect_step = next(
        (s for s in sorted_steps if s.index == suspect_index), sorted_steps[0]
    )
    c_events = [primary_event_id] if primary_event_id else []
    if not c_events:
        step_evs = events_by_step.get(suspect_step.id, [])
        if step_evs:
            c_events = [step_evs[0].id]

    ranked.append(
        StepHypothesis(
            step_index=suspect_step.index,
            node_name=suspect_step.node_name,
            confidence=0.92,
            hypothesis=f"Root cause identified: {reason}",
            cited_event_ids=c_events,
        )
    )

    for step in sorted_steps:
        if step.index != suspect_step.index:
            step_evs = events_by_step.get(step.id, [])
            ev_ids = [step_evs[0].id] if step_evs else []
            ranked.append(
                StepHypothesis(
                    step_index=step.index,
                    node_name=step.node_name,
                    confidence=0.20,
                    hypothesis=f"Step {step.node_name} executed normally.",
                    cited_event_ids=ev_ids,
                )
            )

    result = InvestigationResult(
        summary=f"Primary suspect is step {suspect_step.index} ({suspect_step.node_name}): {reason}",
        ranked_steps=ranked,
    )

    lat = (time.perf_counter() - t0) * 1000
    raw_str = result.model_dump_json()
    return result, raw_str, 250, 100, lat


# ---------------------------------------------------------------------------
# Live LLM Investigator (OpenAI-compatible)
# ---------------------------------------------------------------------------


def _live_investigate(
    prompt: str,
    steps: list[Step],
    events_by_step: dict[int, list[Event]],
    client: httpx.Client | None = None,
) -> tuple[InvestigationResult, str, str, int, int, float]:
    """Call OpenAI-compatible LLM to perform investigation."""
    base_url = clean_env_var(
        os.environ.get("LLM_BASE_URL"), "https://api.groq.com/openai/v1"
    )
    target_model = clean_env_var(os.environ.get("LLM_MODEL"), "openai/gpt-oss-120b")
    api_key = clean_env_var(os.environ.get("LLM_API_KEY"), "")

    if not api_key:
        raise ValueError("LLM_API_KEY is not configured for live investigation.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are an expert root-cause investigator for autonomous LLM agent execution traces. "
        "Diagnose the root-cause step that led to the agent's failure. Output strictly a JSON object "
        "with keys 'summary' (string) and 'ranked_steps' (array of step hypothesis objects)."
    )

    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    def _post(req_body: dict[str, Any]) -> tuple[int, str, dict[str, Any], float]:
        t0 = time.perf_counter()
        max_429_retries = 4
        backoff = 2.0
        resp = None

        for attempt in range(max_429_retries):
            if client is not None:
                resp = client.post(url, headers=headers, json=req_body, timeout=45.0)
            else:
                resp = httpx.post(url, headers=headers, json=req_body, timeout=45.0)

            if resp.status_code == 429 and attempt < max_429_retries - 1:
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after else backoff
                logger.warning(
                    "HTTP 429 Rate limited during investigation. Backing off for %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_429_retries,
                )
                time.sleep(delay)
                backoff *= 2
                continue
            break

        assert resp is not None
        lat = (time.perf_counter() - t0) * 1000
        try:
            data = resp.json()
        except Exception:
            data = {}
        return resp.status_code, resp.text, data, lat

    # --- Attempt 1 ---
    status_code, resp_text, data, lat = _post(payload)
    if status_code != 200:
        raise RuntimeError(
            f"Investigation LLM call failed with HTTP {status_code}: {resp_text[:200]}"
        )

    choices = data.get("choices", [])
    content = ""
    if choices and isinstance(choices, list):
        msg = choices[0].get("message", {})
        content = msg.get("content", "")

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Attempt 1 validation
    try:
        parsed_json = json.loads(content)
        result = InvestigationResult.model_validate(parsed_json)
        return result, content, target_model, prompt_tokens, completion_tokens, lat
    except (json.JSONDecodeError, ValidationError) as err1:
        logger.warning(
            "Investigation validation failed on Attempt 1: %s. Retrying once...", err1
        )

    # --- Attempt 2 (Retry once) ---
    retry_payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid JSON or did not match the schema. "
                    "Return strictly a JSON object with keys 'summary' (str) and 'ranked_steps' "
                    "(list of objects each having 'step_index' (int), 'node_name' (str), 'confidence' (float 0..1), "
                    "'hypothesis' (str), 'cited_event_ids' (list[int]))."
                ),
            },
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    r_status, r_text, r_data, r_lat = _post(retry_payload)
    total_lat = lat + r_lat
    r_usage = r_data.get("usage", {})
    prompt_tokens += r_usage.get("prompt_tokens", 0)
    completion_tokens += r_usage.get("completion_tokens", 0)

    if r_status != 200:
        raise RuntimeError(
            f"Investigation LLM retry failed with HTTP {r_status}: {r_text[:200]}"
        )

    r_choices = r_data.get("choices", [])
    r_content = ""
    if r_choices and isinstance(r_choices, list):
        r_msg = r_choices[0].get("message", {})
        r_content = r_msg.get("content", "")

    parsed_json = json.loads(r_content)
    result = InvestigationResult.model_validate(parsed_json)
    return result, r_content, target_model, prompt_tokens, completion_tokens, total_lat


# ---------------------------------------------------------------------------
# Main Public Interface
# ---------------------------------------------------------------------------


def investigate(
    trace_id: int,
    session: Session | None = None,
    client: httpx.Client | None = None,
) -> InvestigationResult:
    """Analyze a recorded trace to identify and rank the root-cause step.

    Parameters:
    - trace_id: ID of the trace in the database
    - session: Optional SQLAlchemy Session (opens SessionLocal if None)
    - client: Optional httpx.Client for testing/mocking

    Returns:
    - InvestigationResult containing summary and ranked steps.
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True

    try:
        trace = session.query(Trace).filter(Trace.id == trace_id).one_or_none()
        if trace is None:
            raise ValueError(f"Trace ID {trace_id} not found in database.")

        steps = (
            session.query(Step)
            .filter(Step.trace_id == trace_id)
            .order_by(Step.index)
            .all()
        )
        if not steps:
            raise ValueError(f"Trace ID {trace_id} contains no recorded steps.")

        step_ids = [s.id for s in steps]
        events = (
            session.query(Event)
            .filter(Event.step_id.in_(step_ids))
            .order_by(Event.seq)
            .all()
        )

        events_by_step: dict[int, list[Event]] = {s.id: [] for s in steps}
        for ev in events:
            events_by_step[ev.step_id].append(ev)

        # Build isolated prompt (sees ONLY steps, states, and events)
        prompt = build_investigator_prompt(steps, events_by_step)

        provider = clean_env_var(os.environ.get("LLM_PROVIDER"), "stub").lower()
        if provider == "openai_compatible":
            (
                result,
                raw_output,
                model_name,
                p_tokens,
                c_tokens,
                latency_ms,
            ) = _live_investigate(prompt, steps, events_by_step, client=client)
        else:
            result, raw_output, p_tokens, c_tokens, latency_ms = _stub_investigate(
                steps, events_by_step
            )
            model_name = "stub-investigator-v1"

        # Save to investigations table
        hypotheses_dump = [h.model_dump() for h in result.ranked_steps]
        inv_record = Investigation(
            trace_id=trace_id,
            hypotheses=hypotheses_dump,
            ranked_steps=hypotheses_dump,
            raw_llm_output=raw_output,
            summary=result.summary,
            model=model_name,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            latency_ms=latency_ms,
        )
        session.add(inv_record)
        session.commit()

        return result

    finally:
        if close_session:
            session.close()
