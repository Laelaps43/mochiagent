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
        self.code = code
        self.message = message
        self.hint = hint
        self.retriable = retriable
        self.status_code = status_code
        self.provider_code = provider_code
        self.x_log_id = x_log_id
        self.provider = provider
        self.model = model
        self.base_url = base_url


class LLMRateLimitError(LLMProviderError):
    """Raised when upstream enforces rate limits."""


class LLMProtocolError(LLMProviderError):
    """Raised when provider returns incompatible protocol payload."""


class LLMTransportError(LLMProviderError):
    """Raised for non-protocol transport or API errors."""
