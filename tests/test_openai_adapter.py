from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Callable, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from agent.core.llm.adapters.openai import OpenAIAdapter
from agent.core.llm.errors import (
    LLMProtocolError,
    LLMRateLimitError,
    LLMTransportError,
)
from agent.types import LLMConfig, LLMStreamChunk, ToolDefinition


def _make_config(**kwargs: object) -> LLMConfig:
    config = LLMConfig(
        adapter="openai_compatible",
        provider="test",
        model="test-model",
        api_key=SecretStr("dummy-key"),
    )
    for key, val in kwargs.items():
        object.__setattr__(config, key, val)
    return config


def _make_chunk(
    content: str | None = None,
    finish_reason: str | None = None,
    tool_calls: list[object] | None = None,
    usage: object | None = None,
    thinking: str | None = None,
) -> object:
    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
    if thinking is not None:
        delta.reasoning_content = thinking
        delta.reasoning = None
    else:
        delta.reasoning_content = None
        delta.reasoning = None
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


class _ErrorWithResponse(RuntimeError):
    response: object

    def __init__(self, msg: str, response: object) -> None:
        super().__init__(msg)
        self.response = response


def _make_stream(*chunks: object) -> MagicMock:
    async def _iter() -> AsyncIterator[object]:
        for c in chunks:
            yield c

    def _aiter(_self: object) -> AsyncIterator[object]:
        return _iter()

    stream = MagicMock()
    stream.__aiter__ = _aiter
    return stream


def _make_completion(
    content: str = "",
    finish_reason: str = "stop",
    tool_calls: list[object] | None = None,
    usage: object | None = None,
) -> MagicMock:
    comp = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    choice.message = message
    choice.finish_reason = finish_reason
    comp.choices = [choice]
    comp.usage = usage
    return comp


async def test_stream_chat_yields_content_chunks() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    chunk1 = _make_chunk(content="hello ")
    chunk2 = _make_chunk(content="world", finish_reason="stop")
    stream = _make_stream(chunk1, chunk2)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    assert any(r.content == "hello " for r in results)
    assert any(r.finish_reason == "stop" for r in results)


async def test_stream_chat_yields_thinking_chunk() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    chunk = _make_chunk(thinking="I think...", finish_reason="stop")
    stream = _make_stream(chunk)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    assert any(r.thinking == "I think..." for r in results)


async def test_stream_chat_thinking_via_reasoning_attr() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    chunk = MagicMock()
    choice = MagicMock()
    delta = MagicMock()
    delta.content = None
    delta.tool_calls = []
    delta.reasoning_content = None
    delta.reasoning = "alt thinking"
    choice.delta = delta
    choice.finish_reason = "stop"
    chunk.choices = [choice]
    chunk.usage = None
    stream = _make_stream(chunk)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    assert any(r.thinking == "alt thinking" for r in results)


async def test_stream_chat_accumulates_tool_calls() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    tc1 = MagicMock()
    tc1.index = 0
    tc1.id = "call_1"
    fn1 = MagicMock()
    fn1.name = "my_tool"
    fn1.arguments = '{"a":'
    tc1.function = fn1

    tc2 = MagicMock()
    tc2.index = 0
    tc2.id = None
    fn2 = MagicMock()
    fn2.name = None
    fn2.arguments = '"val"}'
    tc2.function = fn2

    chunk1 = _make_chunk(tool_calls=[tc1])
    chunk2 = _make_chunk(tool_calls=[tc2], finish_reason="tool_calls")
    stream = _make_stream(chunk1, chunk2)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    finish_chunks = [r for r in results if r.finish_reason == "tool_calls"]
    assert len(finish_chunks) == 1
    assert finish_chunks[0].tool_calls is not None
    assert len(finish_chunks[0].tool_calls) == 1
    assert finish_chunks[0].tool_calls[0].function.name == "my_tool"
    assert finish_chunks[0].tool_calls[0].function.arguments == '{"a":"val"}'


async def test_stream_chat_tool_call_no_function() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    tc = MagicMock()
    tc.index = 0
    tc.id = "call_y"
    tc.function = None

    chunk = _make_chunk(tool_calls=[tc], finish_reason="tool_calls")
    stream = _make_stream(chunk)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    finish_chunks = [r for r in results if r.finish_reason == "tool_calls"]
    assert len(finish_chunks) == 1
    assert finish_chunks[0].tool_calls is not None
    assert finish_chunks[0].tool_calls[0].id == "call_y"


async def test_stream_chat_with_usage() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    details = MagicMock()
    details.reasoning_tokens = 5
    usage.completion_tokens_details = details

    chunk = _make_chunk(finish_reason="stop", usage=usage)
    stream = _make_stream(chunk)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    usage_chunks = [r for r in results if r.usage is not None]
    assert len(usage_chunks) >= 1
    assert usage_chunks[0].usage is not None
    assert usage_chunks[0].usage.input_tokens == 10
    assert usage_chunks[0].usage.output_tokens == 20
    assert usage_chunks[0].usage.reasoning_tokens == 5


async def test_stream_chat_usage_with_none_details() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    usage = MagicMock()
    usage.prompt_tokens = 3
    usage.completion_tokens = 7
    usage.completion_tokens_details = None

    chunk = _make_chunk(finish_reason="stop", usage=usage)
    stream = _make_stream(chunk)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    usage_chunks = [r for r in results if r.usage is not None]
    assert usage_chunks[0].usage is not None
    assert usage_chunks[0].usage.reasoning_tokens == 0


async def test_stream_chat_maps_exception_to_transport_error() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    adapter.client.chat.completions.create = AsyncMock(side_effect=RuntimeError("network error"))  # type: ignore[assignment]

    with pytest.raises(LLMTransportError):
        async for _ in adapter.stream_chat([]):
            pass


async def test_stream_chat_maps_rate_limit_exception() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    class _FakeRateLimitError(Exception):
        pass

    _FakeRateLimitError.__name__ = "RateLimitError"
    adapter.client.chat.completions.create = AsyncMock(
        side_effect=_FakeRateLimitError("rate limited")
    )  # type: ignore[assignment]

    with pytest.raises(LLMRateLimitError):
        async for _ in adapter.stream_chat([]):
            pass


async def test_stream_chat_maps_rate_limit_by_provider_code() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.text = json.dumps({"error": {"code": "1302", "message": "rate limit"}})
    exc = _ErrorWithResponse("provider error", response)
    adapter.client.chat.completions.create = AsyncMock(side_effect=exc)  # type: ignore[assignment]

    with pytest.raises(LLMRateLimitError):
        async for _ in adapter.stream_chat([]):
            pass


async def test_stream_chat_maps_transport_error_with_status() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    response = MagicMock()
    response.status_code = 503
    response.headers = {}
    response.text = ""
    exc = _ErrorWithResponse("server error", response)
    adapter.client.chat.completions.create = AsyncMock(side_effect=exc)  # type: ignore[assignment]

    with pytest.raises(LLMTransportError) as exc_info:
        async for _ in adapter.stream_chat([]):
            pass
    assert exc_info.value.retriable is True


async def _bad_stream_helper(items: list[object]) -> AsyncIterator[object]:
    for item in items:
        yield item
    raise KeyError("choices")


async def test_stream_chat_maps_key_error_to_protocol_error() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    def _aiter(_self: object) -> AsyncIterator[object]:
        return _bad_stream_helper([])

    stream = MagicMock()
    stream.__aiter__ = _aiter
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    with pytest.raises(LLMProtocolError) as exc_info:
        async for _ in adapter.stream_chat([]):
            pass
    assert exc_info.value.code == "NON_STANDARD_SSE"


async def test_stream_chat_propagates_cancelled_error() -> None:
    import asyncio

    config = _make_config()
    adapter = OpenAIAdapter(config)
    adapter.client.chat.completions.create = AsyncMock(side_effect=asyncio.CancelledError())  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        async for _ in adapter.stream_chat([]):
            pass


async def test_stream_chat_empty_stream_yields_nothing() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    stream = _make_stream()
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([]):
        results.append(c)

    assert results == []


async def test_complete_returns_content() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    completion = _make_completion(content="response text", finish_reason="stop")
    adapter.client.chat.completions.create = AsyncMock(return_value=completion)  # type: ignore[assignment]

    result = await adapter.complete([])

    assert result.content == "response text"
    assert result.finish_reason == "stop"


async def test_complete_with_real_tool_calls() -> None:
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )

    config = _make_config()
    adapter = OpenAIAdapter(config)

    tc = ChatCompletionMessageFunctionToolCall(
        id="call_x",
        function=Function(name="tool_a", arguments='{"x": 1}'),
        type="function",
    )
    completion = _make_completion(tool_calls=[tc], finish_reason="tool_calls")
    adapter.client.chat.completions.create = AsyncMock(return_value=completion)  # type: ignore[assignment]

    result = await adapter.complete([])

    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_x"
    assert result.tool_calls[0].function.name == "tool_a"


async def test_complete_with_usage() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 15
    details = MagicMock()
    details.reasoning_tokens = 0
    usage.completion_tokens_details = details
    completion = _make_completion(content="ok", usage=usage)
    adapter.client.chat.completions.create = AsyncMock(return_value=completion)  # type: ignore[assignment]

    result = await adapter.complete([])

    assert result.usage is not None
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 15


async def test_complete_usage_none_details() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    usage = MagicMock()
    usage.prompt_tokens = 1
    usage.completion_tokens = 2
    usage.completion_tokens_details = None
    completion = _make_completion(content="hi", usage=usage)
    cast(MagicMock, cast(MagicMock, completion.choices[0]).message).tool_calls = None
    adapter.client.chat.completions.create = AsyncMock(return_value=completion)  # type: ignore[assignment]

    result = await adapter.complete([])

    assert result.usage is not None
    assert result.usage.reasoning_tokens == 0


async def test_complete_empty_choices() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    completion = MagicMock()
    completion.choices = []
    completion.usage = None
    adapter.client.chat.completions.create = AsyncMock(return_value=completion)  # type: ignore[assignment]

    result = await adapter.complete([])

    assert result.content == ""


async def test_complete_maps_exception_to_transport_error() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    adapter.client.chat.completions.create = AsyncMock(side_effect=ValueError("bad value"))  # type: ignore[assignment]

    with pytest.raises(LLMTransportError):
        _ = await adapter.complete([])


async def test_complete_maps_key_error_to_protocol_error() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    adapter.client.chat.completions.create = AsyncMock(side_effect=KeyError("choices"))  # type: ignore[assignment]

    with pytest.raises(LLMProtocolError) as exc_info:
        _ = await adapter.complete([])
    assert exc_info.value.code == "PROVIDER_PROTOCOL_ERROR"


async def test_complete_maps_timeout_to_retriable_error() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    adapter.client.chat.completions.create = AsyncMock(side_effect=TimeoutError("timed out"))  # type: ignore[assignment]

    with pytest.raises(LLMTransportError) as exc_info:
        _ = await adapter.complete([])
    assert exc_info.value.retriable is True


async def test_complete_propagates_cancelled_error() -> None:
    import asyncio

    config = _make_config()
    adapter = OpenAIAdapter(config)
    adapter.client.chat.completions.create = AsyncMock(side_effect=asyncio.CancelledError())  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        _ = await adapter.complete([])


async def test_complete_already_provider_error_reraises_unchanged() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    original = LLMRateLimitError(code="RATE_LIMITED", message="rate limited")
    adapter.client.chat.completions.create = AsyncMock(side_effect=original)  # type: ignore[assignment]

    with pytest.raises(LLMRateLimitError) as exc_info:
        _ = await adapter.complete([])
    assert exc_info.value is original


async def test_stream_chat_already_provider_error_reraises_unchanged() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)
    original = LLMRateLimitError(code="RATE_LIMITED", message="rate limited")
    adapter.client.chat.completions.create = AsyncMock(side_effect=original)  # type: ignore[assignment]

    with pytest.raises(LLMRateLimitError) as exc_info:
        async for _ in adapter.stream_chat([]):
            pass
    assert exc_info.value is original


async def test_stream_chat_with_max_tokens_and_tools() -> None:
    config = _make_config(max_tokens=128)
    adapter = OpenAIAdapter(config)
    chunk = _make_chunk(content="ok", finish_reason="stop")
    stream = _make_stream(chunk)
    adapter.client.chat.completions.create = AsyncMock(return_value=stream)  # type: ignore[assignment]
    tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]

    results: list[LLMStreamChunk] = []
    async for c in adapter.stream_chat([], tools=tools):
        results.append(c)

    assert any(r.content == "ok" for r in results)
    create_mock: AsyncMock = adapter.client.chat.completions.create  # type: ignore[assignment]
    call_kwargs = cast(dict[str, object], create_mock.call_args[1])
    assert call_kwargs["max_tokens"] == 128
    assert "tools" in call_kwargs


async def test_complete_rate_limit_by_status_429() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    response = MagicMock()
    response.status_code = 429
    response.headers = {}
    response.text = ""
    exc = _ErrorWithResponse("rate limit", response)
    adapter.client.chat.completions.create = AsyncMock(side_effect=exc)  # type: ignore[assignment]

    with pytest.raises(LLMRateLimitError):
        _ = await adapter.complete([])


async def test_complete_error_response_with_hint_logged() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    response = MagicMock()
    response.status_code = 503
    response.headers = {"content-type": "application/json", "x-log-id": "xyz"}
    response.text = json.dumps({"error": {"code": "SERVICE_DOWN", "message": "maintenance"}})
    exc = _ErrorWithResponse("error with hint", response)
    adapter.client.chat.completions.create = AsyncMock(side_effect=exc)  # type: ignore[assignment]

    with pytest.raises(LLMTransportError) as exc_info:
        _ = await adapter.complete([])
    assert exc_info.value.provider_code == "SERVICE_DOWN"
    assert exc_info.value.x_log_id == "xyz"


async def test_stream_chat_response_meta_body_truncated() -> None:
    config = _make_config()
    adapter = OpenAIAdapter(config)

    response = MagicMock()
    response.status_code = 503
    response.headers = {}
    response.text = "x" * 2000
    exc = _ErrorWithResponse("long error", response)
    adapter.client.chat.completions.create = AsyncMock(side_effect=exc)  # type: ignore[assignment]

    with pytest.raises(LLMTransportError):
        async for _ in adapter.stream_chat([]):
            pass


async def test_parse_error_body_valid_json() -> None:
    parse_error_body = cast(
        Callable[[str], tuple[str | None, str | None]],
        vars(OpenAIAdapter)["_parse_error_body"].__func__,
    )
    body = json.dumps({"error": {"code": "1302", "message": "rate limited"}})
    code, msg = parse_error_body(body)
    assert code == "1302"
    assert msg is not None


async def test_parse_error_body_missing_error_key() -> None:
    parse_error_body = cast(
        Callable[[str], tuple[str | None, str | None]],
        vars(OpenAIAdapter)["_parse_error_body"].__func__,
    )
    code, msg = parse_error_body(json.dumps({"something": "else"}))
    assert code is None
    assert msg is None


async def test_parse_error_body_non_dict_error() -> None:
    parse_error_body = cast(
        Callable[[str], tuple[str | None, str | None]],
        vars(OpenAIAdapter)["_parse_error_body"].__func__,
    )
    code, msg = parse_error_body(json.dumps({"error": "string error"}))
    assert code is None
    assert msg is None
