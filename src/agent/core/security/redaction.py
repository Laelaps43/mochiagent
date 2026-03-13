"""
Redaction helpers for sensitive data in logs and persisted metadata.
"""

from __future__ import annotations

import re

from typing import cast

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
    "credential",
    "credentials",
    "session_token",
    "auth_token",
}

_KEY_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|x[_-]?api[_-]?key|secret|password|credential)\b(\s*[:=]\s*)([\"']?)([^\"'\s,}]{6,})\3"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._\-]{6,})")


def mask_secret(value: object) -> object:
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


def _is_sensitive_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _SENSITIVE_FIELD_NAMES:
        return True
    for substr in ("key", "secret", "token", "password", "credential"):
        if substr in normalized:
            return True
    return False


def redact_dict(data: object) -> object:
    """Recursively redact sensitive fields in dict/list trees."""
    if isinstance(data, dict):
        d = cast(dict[object, object], data)
        redacted: dict[object, object] = {}
        for key, value in d.items():
            if _is_sensitive_key(key):
                redacted[key] = mask_secret(value)
            else:
                redacted[key] = redact_dict(value)
        return redacted
    if isinstance(data, list):
        return [redact_dict(item) for item in cast(list[object], data)]
    if isinstance(data, tuple):
        return tuple(redact_dict(item) for item in cast(tuple[object, ...], data))
    return data


def redact_text(text: object) -> str:
    """Redact common key/token patterns from free-form text."""
    if text is None:
        return ""
    value = str(text)
    value = _BEARER_RE.sub("Bearer " + REDACTED, value)
    value = _KEY_VALUE_RE.sub(r"\1\2" + REDACTED, value)
    return value
