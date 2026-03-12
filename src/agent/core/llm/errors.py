"""
LLM provider error models.
"""

from __future__ import annotations


class LLMProviderError(RuntimeError):
    """Base exception for provider-facing failures."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        hint: str | None = None,
        retriable: bool = False,
        status_code: int | None = None,
        provider_code: str | None = None,
        x_log_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code: str = code
        self.message: str = message
        self.hint: str | None = hint
        self.retriable: bool = retriable
        self.status_code: int | None = status_code
        self.provider_code: str | None = provider_code
        self.x_log_id: str | None = x_log_id
        self.provider: str | None = provider
        self.model: str | None = model
        self.base_url: str | None = base_url


class LLMRateLimitError(LLMProviderError):
    """Raised when upstream enforces rate limits."""


class LLMProtocolError(LLMProviderError):
    """Raised when provider returns incompatible protocol payload."""


class LLMTransportError(LLMProviderError):
    """Raised for non-protocol transport or API errors."""


_CONTEXT_OVERFLOW_KEYWORDS = [
    "context_length",
    "context length",
    "context window",
    "maximum context",
    "prompt is too long",
    "too many tokens",
    "token limit",
    "input is too long",
]


def is_context_overflow_error(exc: Exception) -> bool:
    """Check whether *exc* signals a context-window overflow.

    Works with both :class:`LLMProviderError` (structured fields) and
    plain exceptions (string matching on ``str(exc)``).
    """
    provider_error: LLMProviderError | None = None
    current: BaseException | None = exc
    while current:
        if isinstance(current, LLMProviderError):
            provider_error = current
            break
        current = current.__cause__

    if provider_error:
        code = (provider_error.code or "").lower()
        provider_code = (provider_error.provider_code or "").lower()
        message = (provider_error.message or "").lower()
        status = provider_error.status_code
    else:
        code = ""
        provider_code = ""
        message = str(exc).lower()
        status = None

    if any(kw in code for kw in _CONTEXT_OVERFLOW_KEYWORDS):
        return True
    if any(kw in provider_code for kw in _CONTEXT_OVERFLOW_KEYWORDS):
        return True
    if any(kw in message for kw in _CONTEXT_OVERFLOW_KEYWORDS):
        return True
    return bool(status == 400 and "context" in message and "token" in message)
