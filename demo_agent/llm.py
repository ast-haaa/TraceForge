# demo_agent/llm.py
"""LLM adapter supporting offline deterministic stub and OpenAI-compatible endpoints."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------


class DecisionOutput(BaseModel):
    """Structured decision output expected from the LLM."""

    decision: Literal["approve", "deny"]
    reason: str


class LLMResponse(BaseModel):
    """Normalized response returned by any LLM provider adapter."""

    decision: Literal["approve", "deny"]
    reason: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    raw_response: dict[str, Any] | str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def clean_env_var(val: str | None, default: str = "") -> str:
    """Clean environment variable values by stripping whitespace, quotes, and angle brackets."""
    if val is None:
        return default
    s = val.strip().strip("\"'").strip("<>").strip()
    return s if s else default


# ---------------------------------------------------------------------------
# Provider 1: Deterministic Stub LLM (Default for CI and tests)
# ---------------------------------------------------------------------------


def stub_llm(
    prompt: str,
    *,
    policy_days: int | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Simulate an LLM using rule-based evaluation of DAYS_OLD in prompt."""
    t0 = time.perf_counter()
    target_model = model or "stub-llm-v1"

    # Parse DAYS_OLD from prompt
    days_old: int | None = None
    for line in prompt.splitlines():
        if "DAYS_OLD:" in line:
            try:
                days_old = int(line.split("DAYS_OLD:")[-1].strip())
            except ValueError:
                pass

    if days_old is None:
        dec: Literal["approve", "deny"] = "approve"
        reason = "unable to determine order age"
    else:
        if policy_days is None:
            policy_line_match = re.search(
                r"^Policy:[ \t]*(.*)$", prompt, re.MULTILINE
            )
            policy_content = (
                policy_line_match.group(1).strip() if policy_line_match else ""
            )
            if not policy_content:
                dec = "deny"
                reason = "refund denied: no refund policy provided"
            else:
                policy_match = re.search(r"within (\d+) days", prompt, re.IGNORECASE)
                policy_days = int(policy_match.group(1)) if policy_match else 7
                if days_old <= policy_days:
                    dec = "approve"
                    reason = f"order is {days_old} day(s) old, within the {policy_days}-day window"
                else:
                    dec = "deny"
                    reason = f"order is {days_old} day(s) old, outside the {policy_days}-day window"
        else:
            if days_old <= policy_days:
                dec = "approve"
                reason = (
                    f"order is {days_old} day(s) old, within the {policy_days}-day window"
                )
            else:
                dec = "deny"
                reason = (
                    f"order is {days_old} day(s) old, outside the {policy_days}-day window"
                )

    latency = (time.perf_counter() - t0) * 1000
    p_tokens = len(prompt.split())
    c_tokens = len(reason.split())

    return LLMResponse(
        decision=dec,
        reason=reason,
        model=target_model,
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        total_tokens=p_tokens + c_tokens,
        latency_ms=latency,
        cost=0.0,
        raw_response={"decision": dec, "reason": reason},
    )


# ---------------------------------------------------------------------------
# Provider 2: OpenAI-Compatible Endpoint (Groq, vLLM, OpenAI, etc.)
# ---------------------------------------------------------------------------


def openai_compatible_llm(
    prompt: str,
    *,
    policy_days: int | None = None,
    model: str | None = None,
    client: httpx.Client | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """Call an OpenAI-compatible /chat/completions endpoint using httpx.

    Enforces:
    - temperature=0.0
    - response_format={"type": "json_object"}
    - Pydantic validation against DecisionOutput
    - Retry once on invalid JSON/schema
    - Timeout and backoff on HTTP 429
    - No API key or Authorization header stored in returned data
    """
    base_url = clean_env_var(
        os.environ.get("LLM_BASE_URL"), "https://api.groq.com/openai/v1"
    )
    target_model = model or clean_env_var(
        os.environ.get("LLM_MODEL"), "openai/gpt-oss-120b"
    )
    api_key = clean_env_var(os.environ.get("LLM_API_KEY"), "")

    if not api_key:
        return LLMResponse(
            decision="deny",
            reason="LLM_API_KEY is not configured",
            model=target_model,
            error="Missing LLM_API_KEY",
        )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "You are an automated refund decision agent. Evaluate the customer refund request against "
        "the provided refund policy strictly. You must output ONLY a JSON object with two fields: "
        "'decision' (which must be exactly 'approve' or 'deny') and 'reason' (a concise explanation)."
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
        max_429_retries = 3
        backoff = 1.0
        resp = None

        for attempt in range(max_429_retries):
            if client is not None:
                resp = client.post(url, headers=headers, json=req_body, timeout=30.0)
            else:
                resp = httpx.post(url, headers=headers, json=req_body, timeout=30.0)

            if resp.status_code == 429 and attempt < max_429_retries - 1:
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after else backoff
                logger.warning(
                    "HTTP 429 Rate limited. Backing off for %.1fs (attempt %d/%d)",
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
        return LLMResponse(
            decision="deny",
            reason=f"LLM API request failed with HTTP {status_code}",
            model=target_model,
            latency_ms=lat,
            error=f"HTTP {status_code}: {resp_text[:200]}",
        )

    choices = data.get("choices", [])
    content = ""
    if choices and isinstance(choices, list):
        msg = choices[0].get("message", {})
        content = msg.get("content", "")

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    # Validate Attempt 1
    try:
        parsed_json = json.loads(content)
        validated = DecisionOutput.model_validate(parsed_json)
        return LLMResponse(
            decision=validated.decision,
            reason=validated.reason,
            model=target_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=lat,
            cost=0.0,
            raw_response=data,
        )
    except (json.JSONDecodeError, ValidationError) as err1:
        logger.warning(
            "LLM output validation failed on Attempt 1: %s. Retrying once...", err1
        )

    # --- Attempt 2 (Retry once on invalid JSON/schema) ---
    retry_payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid JSON. Return strictly a JSON object with keys "
                    "'decision' ('approve' or 'deny') and 'reason'."
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
    total_tokens = prompt_tokens + completion_tokens

    if r_status != 200:
        return LLMResponse(
            decision="deny",
            reason=f"LLM API retry failed with HTTP {r_status}",
            model=target_model,
            latency_ms=total_lat,
            error=f"HTTP {r_status} on retry: {r_text[:200]}",
        )

    r_choices = r_data.get("choices", [])
    r_content = ""
    if r_choices and isinstance(r_choices, list):
        r_msg = r_choices[0].get("message", {})
        r_content = r_msg.get("content", "")

    try:
        r_parsed = json.loads(r_content)
        r_validated = DecisionOutput.model_validate(r_parsed)
        return LLMResponse(
            decision=r_validated.decision,
            reason=r_validated.reason,
            model=target_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=total_lat,
            cost=0.0,
            raw_response=r_data,
        )
    except (json.JSONDecodeError, ValidationError) as err2:
        return LLMResponse(
            decision="deny",
            reason="Invalid LLM output after retry",
            model=target_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=total_lat,
            cost=0.0,
            raw_response=r_data,
            error=f"Invalid output after retry: {err2}",
        )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def get_llm() -> Callable[..., LLMResponse]:
    """Return the configured LLM callable based on LLM_PROVIDER env var.

    Values:
    - 'stub': pure Python deterministic stub (default)
    - 'openai_compatible': httpx-based OpenAI-compatible endpoint
    """
    provider = clean_env_var(os.environ.get("LLM_PROVIDER"), "stub").lower()
    if provider == "openai_compatible":
        return openai_compatible_llm
    return stub_llm
