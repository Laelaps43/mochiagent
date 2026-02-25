"""
Redaction helpers for sensitive data in logs and persisted metadata.
"""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
_MASK = "***"

_SENSITIVE_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "proxy_authorization",
    "x_api_key",
    "x-api-key",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "client_secret",
    "password",
    "private_key",
}

_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|x[_-]?api[_-]?key|secret)\b"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,}]{6,})\3"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._\-]{6,})")


def mask_secret(value: Any) -> Any:
    """Mask full secret values while keeping very small context for debugging."""
    if value is None:
        return None
    if not isinstance(value, str):
        return REDACTED
    if not value:
        return value
    if value == REDACTED:
        return value
    if len(value) <= 8:
        return _MASK
    return f"{value[:4]}{_MASK}{value[-3:]}"


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_FIELD_NAMES


def redact_dict(data: Any) -> Any:
    """Recursively redact sensitive fields in dict/list trees."""
    if isinstance(data, dict):
        redacted: dict[Any, Any] = {}
        for key, value in data.items():
            if _is_sensitive_key(key):
                redacted[key] = mask_secret(value)
            else:
                redacted[key] = redact_dict(value)
        return redacted
    if isinstance(data, list):
        return [redact_dict(item) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_dict(item) for item in data)
    return data


def redact_text(text: Any) -> str:
    """Redact common key/token patterns from free-form text."""
    if text is None:
        return ""
    value = str(text)
    value = _BEARER_RE.sub("Bearer " + REDACTED, value)
    value = _KEY_VALUE_RE.sub(r"\1\2" + REDACTED, value)
    return value
