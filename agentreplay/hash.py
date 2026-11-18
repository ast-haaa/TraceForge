# agentreplay/hash.py
"""Deterministic input-hash used for replay event matching.

The hash is computed over the *canonical* JSON representation of the input:
  - sort_keys=True  → dictionary ordering is stable
  - separators=(',', ':') → no whitespace variation
  - ensure_ascii=False → keep Unicode as-is

Volatile fields (timestamps, UUIDs, request IDs) are automatically stripped
before hashing so callers do not have to clean payloads manually.
"""
import datetime
import hashlib
import json
import re
import uuid
from typing import Any

# Clearly volatile dict keys (timestamps, UUIDs, request IDs)
# Domain date/time keys (e.g. "date", "order_date") are NOT in this set.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "timestamps",
        "created_at",
        "updated_at",
        "issued_at",
        "request_id",
        "request-id",
        "req_id",
        "trace_id",
        "span_id",
        "correlation_id",
        "uuid",
        "run_id",
    }
)

_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_TIMESTAMP_PATTERN = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)


def strip_volatile_fields(value: Any) -> Any:
    """Recursively strip volatile fields (timestamps, UUIDs, request IDs).

    - Dict keys in VOLATILE_KEYS are removed.
    - String values with UUIDs or full ISO timestamps are normalized to sentinels.
    - datetime.datetime objects are normalized to <VOLATILE_TIMESTAMP>.
    - datetime.date objects are preserved (converted to ISO string for JSON serialization).
    - uuid.UUID objects are normalized to <VOLATILE_UUID>.
    - Nested dicts/lists/tuples/sets are walked recursively.
    """
    if isinstance(value, dict):
        return {
            k: strip_volatile_fields(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.lower() in VOLATILE_KEYS)
        }
    if isinstance(value, (list, tuple)):
        cleaned = [strip_volatile_fields(v) for v in value]
        return type(value)(cleaned)
    if isinstance(value, set):
        return {strip_volatile_fields(v) for v in value}
    if isinstance(value, datetime.datetime):
        return "<VOLATILE_TIMESTAMP>"
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return "<VOLATILE_UUID>"
    if isinstance(value, str):
        v = _UUID_PATTERN.sub("<VOLATILE_UUID>", value)
        v = _TIMESTAMP_PATTERN.sub("<VOLATILE_TIMESTAMP>", v)
        return v
    return value


def _canonical_json(value: Any) -> str:
    """Return deterministic JSON string with sorted keys and no whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_input_hash(value: Any) -> str:
    """SHA-256 hex digest of canonical JSON representation after stripping volatile fields."""
    cleaned = strip_volatile_fields(value)
    canonical = _canonical_json(cleaned)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
