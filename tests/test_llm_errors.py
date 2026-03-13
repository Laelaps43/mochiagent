from __future__ import annotations

from agent.core.llm.errors import (
    LLMProviderError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMTransportError,
    is_context_overflow_error,
)


def _make_error(**kwargs: object) -> LLMProviderError:
    return LLMProviderError(
        code=str(kwargs.get("code", "ERR")),
        message=str(kwargs.get("message", "error")),
        hint=str(kwargs["hint"]) if "hint" in kwargs else None,
        retriable=bool(kwargs.get("retriable", False)),
        status_code=int(str(kwargs["status_code"])) if "status_code" in kwargs else None,
        provider_code=str(kwargs["provider_code"]) if "provider_code" in kwargs else None,
        x_log_id=str(kwargs["x_log_id"]) if "x_log_id" in kwargs else None,
        provider=str(kwargs["provider"]) if "provider" in kwargs else None,
        model=str(kwargs["model"]) if "model" in kwargs else None,
        base_url=str(kwargs["base_url"]) if "base_url" in kwargs else None,
    )


class TestLLMProviderError:
    def test_basic_fields(self):
        e = _make_error(
            code="TEST",
            message="something failed",
            hint="check config",
            retriable=True,
            status_code=500,
            provider_code="P500",
            x_log_id="xlog1",
            provider="openai",
            model="gpt-4",
            base_url="https://api.openai.com",
        )
        assert e.code == "TEST"
        assert e.message == "something failed"
        assert str(e) == "something failed"
        assert e.hint == "check config"
        assert e.retriable is True
        assert e.status_code == 500
        assert e.provider_code == "P500"
        assert e.x_log_id == "xlog1"
        assert e.provider == "openai"
        assert e.model == "gpt-4"
        assert e.base_url == "https://api.openai.com"

    def test_optional_fields_default_none(self):
        e = _make_error(code="X", message="m")
        assert e.hint is None
        assert e.status_code is None
        assert e.provider_code is None
        assert e.x_log_id is None
        assert e.provider is None
        assert e.model is None
        assert e.base_url is None
        assert e.retriable is False

    def test_is_runtime_error(self):
        e = _make_error(code="X", message="m")
        assert isinstance(e, RuntimeError)

    def test_subclasses_inherit(self):
        rate = LLMRateLimitError(code="RL", message="rate limited")
        proto = LLMProtocolError(code="PR", message="protocol error")
        transport = LLMTransportError(code="TP", message="transport error")
        assert isinstance(rate, LLMProviderError)
        assert isinstance(proto, LLMProviderError)
        assert isinstance(transport, LLMProviderError)


class TestIsContextOverflowError:
    def test_returns_false_for_plain_error(self):
        assert is_context_overflow_error(ValueError("something went wrong")) is False

    def test_detects_context_length_in_code(self):
        e = _make_error(code="context_length_exceeded", message="fine")
        assert is_context_overflow_error(e) is True

    def test_detects_context_window_in_code(self):
        e = _make_error(code="context window exceeded", message="fine")
        assert is_context_overflow_error(e) is True

    def test_detects_too_many_tokens_in_provider_code(self):
        e = _make_error(code="GENERIC", message="fine", provider_code="too many tokens limit")
        assert is_context_overflow_error(e) is True

    def test_detects_prompt_is_too_long_in_message(self):
        e = _make_error(code="ERR", message="prompt is too long for this model")
        assert is_context_overflow_error(e) is True

    def test_detects_token_limit_in_message(self):
        e = _make_error(code="ERR", message="you exceeded the token limit")
        assert is_context_overflow_error(e) is True

    def test_detects_input_is_too_long(self):
        e = _make_error(code="ERR", message="input is too long")
        assert is_context_overflow_error(e) is True

    def test_detects_maximum_context_in_message(self):
        e = _make_error(code="ERR", message="maximum context reached")
        assert is_context_overflow_error(e) is True

    def test_status_400_with_context_and_token(self):
        e = _make_error(
            code="ERR",
            message="context length exceeded: too many token",
            status_code=400,
        )
        assert is_context_overflow_error(e) is True

    def test_status_400_without_keywords(self):
        e = _make_error(code="ERR", message="bad request", status_code=400)
        assert is_context_overflow_error(e) is False

    def test_chained_exception_detected(self):
        inner = _make_error(code="context_length", message="overflow")
        outer = ValueError("wrapper")
        outer.__cause__ = inner
        assert is_context_overflow_error(outer) is True

    def test_plain_exception_message_matching(self):
        e = ValueError("context length exceeded")
        assert is_context_overflow_error(e) is True

    def test_plain_exception_no_match(self):
        e = ValueError("network timeout")
        assert is_context_overflow_error(e) is False
