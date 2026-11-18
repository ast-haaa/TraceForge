# agentreplay/redact.py
"""Remove secrets and PII from arbitrary Python data before storage.

Rules:
- Dict keys matching SECRET_KEYS (case-insensitive) are dropped entirely.
- Strings are scanned for email addresses and phone numbers and replaced
  with sentinel tokens.
- Nested structures (list, tuple, set) are walked recursively.
- Scalars that are not strings are returned unchanged.
"""
import re
from typing import Any

SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "api-key",
        "secret",
        "secret_key",
        "password",
        "passwd",
        "access_token",
        "auth_token",
        "bearer",
        "token",
        "session_id",
        "authorization",
        "x-api-key",
        "private_key",
        "credential",
    }
)

_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+")
_PHONE_RE = re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_API_KEY_RE = re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{8,}|key-[a-zA-Z0-9_\-]{8,})\b")


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _redact_value(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.lower() in SECRET_KEYS)
        }
    if isinstance(value, (list, tuple)):
        redacted = [_redact_value(v) for v in value]
        return type(value)(redacted)
    if isinstance(value, set):
        return {_redact_value(v) for v in value}
    if isinstance(value, str):
        v = _EMAIL_RE.sub("<REDACTED_EMAIL>", value)
        v = _PHONE_RE.sub("<REDACTED_PHONE>", v)
        v = _API_KEY_RE.sub("<REDACTED_API_KEY>", v)
        return v
    return value


def redact_sensitive(data: Any) -> Any:
    """Return a deep copy of *data* with secrets and PII removed.

    ``None`` is returned unchanged.
    """
    if data is None:
        return None
    return _redact_value(data)
