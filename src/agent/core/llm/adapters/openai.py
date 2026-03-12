"""
OpenAI Compatible Adapter - 使用OpenAI SDK支持OpenAI兼容的厂商
支持DeepSeek, OpenAI, Azure, Kimi, Qwen等
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Awaitable
from typing import NoReturn, cast, override

from loguru import logger
from openai import AsyncOpenAI

from agent.constants import OPENAI_MAX_RETRIES
from agent.core.llm.base import LLMProvider
from agent.core.message import Message as InternalMessage
from agent.core.llm.errors import (
    LLMProtocolError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTransportError,
)
from agent.core.security import redact_text
from agent.types import (
    LLMConfig,
    LLMStreamChunk,
    ProviderUsage,
    ToolCallPayload,
    ToolDefinition,
    ToolFunctionPayload,
)


class _ResponseMeta:
    __slots__: tuple[str, ...] = (
        "status_code",
        "content_type",
        "x_log_id",
        "provider_code",
        "provider_message",
        "response_body",
    )

    def __init__(
        self,
        status_code: int | None = None,
        content_type: str | None = None,
        x_log_id: str | None = None,
        provider_code: str | None = None,
        provider_message: str | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code: int | None = status_code
        self.content_type: str | None = content_type
        self.x_log_id: str | None = x_log_id
        self.provider_code: str | None = provider_code
        self.provider_message: str | None = provider_message
        self.response_body: str | None = response_body


def _gstr(obj: object, attr: str, default: str = "") -> str:
    val: object = cast(object, getattr(obj, attr, default))
    return str(val) if val is not None else default


def _gint(obj: object, attr: str, default: int = 0) -> int:
    val: object = cast(object, getattr(obj, attr, default))
    if val is None:
        return default
    try:
        return int(str(val))
    except (TypeError, ValueError):
        return default


def _gobj(obj: object, attr: str) -> object:
    return cast(object, getattr(obj, attr, None))


class OpenAIAdapter(LLMProvider):
    """
    OpenAI兼容适配器
    使用官方OpenAI SDK，支持所有兼容OpenAI接口的LLM
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.openai_max_retries: int = self._resolve_max_retries(config.openai_max_retries)
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=self.openai_max_retries,
        )

    @staticmethod
    def _resolve_max_retries(configured_retries: int | None) -> int:
        if configured_retries is None or configured_retries < 0:
            return OPENAI_MAX_RETRIES
        return configured_retries

    @staticmethod
    def _prepare_tools(tools: list[ToolDefinition]) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _build_request_params(
        self,
        *,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None,
        stream: bool,
        **kwargs: object,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "model": self.config.model,
            "messages": self.prepare_messages(messages),
            "stream": stream,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            params["max_tokens"] = self.config.max_tokens
        if tools:
            params["tools"] = self._prepare_tools(tools)
            params["tool_choice"] = "auto"
        params.update(self.config.extra_params)
        params.update(kwargs)
        if stream:
            params["stream_options"] = {"include_usage": True}
        return params

    @staticmethod
    def _extract_response_meta(exc: Exception) -> _ResponseMeta:
        meta = _ResponseMeta()

        response: object = cast(object, getattr(exc, "response", None))
        if response is None:
            return meta

        try:
            meta.status_code = cast("int | None", _gobj(response, "status_code"))
            headers = cast(dict[str, str], _gobj(response, "headers") or {})
            meta.content_type = headers.get("content-type", "")
            meta.x_log_id = headers.get("x-log-id", "")

            body_text: str = _gstr(response, "text")
            if body_text:
                if len(body_text) > 1200:
                    body_text = body_text[:1200] + "...(truncated)"
                meta.response_body = body_text

                try:
                    payload = cast("dict[str, object]", json.loads(body_text))
                    error_obj = cast("dict[str, object] | None", payload.get("error"))
                    if isinstance(error_obj, dict):
                        code = error_obj.get("code")
                        if code is not None:
                            meta.provider_code = str(code)
                        message = error_obj.get("message")
                        if message:
                            meta.provider_message = redact_text(str(message))
                except Exception:
                    pass
        except Exception:
            pass

        return meta

    def _map_provider_exception(self, operation: str, exc: Exception) -> LLMProviderError:
        if isinstance(exc, LLMProviderError):
            return exc

        provider = self.config.provider
        model = self.config.model
        base_url = self.config.base_url

        if isinstance(exc, KeyError):
            missing_key = str(cast(object, exc.args[0])) if exc.args else "unknown"
            is_stream = operation == "stream_chat"
            parse_subject = (
                "OpenAI-compatible streaming parse failed "
                if is_stream
                else "OpenAI-compatible response parse failed "
            )
            payload_subject = (
                "Provider may be returning a non-standard SSE error payload. "
                if is_stream
                else "Provider may be returning a non-standard response payload. "
            )
            return LLMProtocolError(
                code="NON_STANDARD_SSE" if is_stream else "PROVIDER_PROTOCOL_ERROR",
                message=(
                    parse_subject
                    + f"(missing key: {missing_key!r}). "
                    + payload_subject
                    + f"model={self.config.model}, base_url={self.config.base_url}"
                ),
                hint=(
                    "供应商返回了非标准流式错误包。建议重试，或切换更稳定的模型/网关。"
                    if is_stream
                    else "供应商返回了非标准响应包。建议重试，或切换更稳定的模型/网关。"
                ),
                retriable=True,
                provider=provider,
                model=model,
                base_url=base_url,
            )

        meta = self._extract_response_meta(exc)
        status_code = meta.status_code
        provider_code = meta.provider_code
        provider_message = meta.provider_message
        x_log_id = meta.x_log_id

        is_rate_limit = (
            type(exc).__name__ == "RateLimitError" or status_code == 429 or provider_code == "1302"
        )
        if is_rate_limit:
            detail = provider_message or redact_text(str(exc))
            return LLMRateLimitError(
                code="RATE_LIMITED",
                message=(
                    "provider rate limited request"
                    f" (status={status_code}, provider_code={provider_code}, x_log_id={x_log_id}): "
                    f"{detail}"
                ),
                hint="触发模型供应商限流（429/1302）。请稍后重试，或降低请求频率/并发。",
                retriable=True,
                status_code=status_code,
                provider_code=provider_code,
                x_log_id=x_log_id,
                provider=provider,
                model=model,
                base_url=base_url,
            )

        retriable = bool(
            isinstance(exc, TimeoutError)
            or status_code in {408, 409, 429}
            or (isinstance(status_code, int) and status_code >= 500)
        )
        detail = redact_text(f"{type(exc).__name__}: {exc}")
        if status_code is not None:
            message = (
                f"{operation} failed"
                f" (status={status_code}, provider_code={provider_code}, x_log_id={x_log_id}): "
                f"{detail}"
            )
        else:
            message = f"{operation} failed: {detail}"

        return LLMTransportError(
            code="LLM_ERROR",
            message=message,
            retriable=retriable,
            status_code=status_code,
            provider_code=provider_code,
            x_log_id=x_log_id,
            provider=provider,
            model=model,
            base_url=base_url,
        )

    @staticmethod
    def _log_provider_exception(
        stage: str,
        provider_error: LLMProviderError,
        exc: Exception,
    ) -> None:
        logger.error(
            "[LLM][{}] code={} retriable={} status={} provider_code={} x_log_id={} message={}",
            stage,
            provider_error.code,
            provider_error.retriable,
            provider_error.status_code,
            provider_error.provider_code,
            provider_error.x_log_id,
            redact_text(provider_error.message),
        )
        if provider_error.hint:
            logger.error("[LLM][{}] hint={}", stage, provider_error.hint)
        logger.error(
            "[LLM][{}] provider={} model={} base_url={}",
            stage,
            provider_error.provider,
            provider_error.model,
            provider_error.base_url,
        )
        logger.error(
            "[LLM][{}] exception type={} repr={}",
            stage,
            type(exc).__name__,
            redact_text(repr(exc)),
        )

        response: object = cast(object, getattr(exc, "response", None))
        if response is not None:
            try:
                headers = cast(dict[str, str], _gobj(response, "headers") or {})
                content_type: str = headers.get("content-type", "")
                body_text: str = _gstr(response, "text")
                if len(body_text) > 1200:
                    body_text = body_text[:1200] + "...(truncated)"
                logger.error(
                    "[LLM][{}] response content-type={} body={}",
                    stage,
                    content_type,
                    redact_text(body_text),
                )
            except Exception:
                pass

    def _raise_mapped_error(self, operation: str, exc: Exception) -> NoReturn:
        provider_error = self._map_provider_exception(operation, exc)
        self._log_provider_exception(f"{operation}.error", provider_error, exc)
        raise provider_error from exc

    @staticmethod
    def _merge_tool_call_delta(
        tool_call: object,
        accumulated_tool_calls: dict[int, ToolCallPayload],
    ) -> None:
        tc_index = _gobj(tool_call, "index")
        idx = int(tc_index) if isinstance(tc_index, int) else len(accumulated_tool_calls)
        if idx not in accumulated_tool_calls:
            accumulated_tool_calls[idx] = ToolCallPayload()

        entry = accumulated_tool_calls[idx]

        tc_id = _gstr(tool_call, "id")
        if tc_id:
            entry.id = tc_id

        tc_function = _gobj(tool_call, "function")
        if not tc_function:
            return

        fn_name = _gstr(tc_function, "name")
        if fn_name:
            entry.function.name = fn_name

        fn_arguments = _gstr(tc_function, "arguments")
        if fn_arguments:
            entry.function.arguments += fn_arguments

    @staticmethod
    def _parse_stream_chunk(
        chunk: object,
        accumulated_tool_calls: dict[int, ToolCallPayload],
    ) -> LLMStreamChunk | None:
        choices = _gobj(chunk, "choices")
        choice = cast(list[object], choices)[0] if choices else None
        if not choice:
            return None

        result = LLMStreamChunk()
        delta = _gobj(choice, "delta")
        has_data = False

        if delta:
            delta_content = _gstr(delta, "content")
            if delta_content:
                result.content = delta_content
                has_data = True

            reasoning_content = _gobj(delta, "reasoning_content") or _gobj(delta, "reasoning")
            if reasoning_content:
                result.thinking = str(reasoning_content)
                has_data = True

            delta_tool_calls = _gobj(delta, "tool_calls")
            if delta_tool_calls:
                for tool_call in cast(list[object], delta_tool_calls):
                    OpenAIAdapter._merge_tool_call_delta(tool_call, accumulated_tool_calls)

        finish_reason = _gstr(choice, "finish_reason")
        if finish_reason:
            result.finish_reason = finish_reason
            has_data = True
            if accumulated_tool_calls:
                ordered_calls = [accumulated_tool_calls[k] for k in sorted(accumulated_tool_calls)]
                result.tool_calls = ordered_calls
        usage = _gobj(chunk, "usage")
        if usage is not None:
            details = _gobj(usage, "completion_tokens_details")
            result.usage = ProviderUsage(
                input_tokens=_gint(usage, "prompt_tokens"),
                output_tokens=_gint(usage, "completion_tokens"),
                reasoning_tokens=_gint(details, "reasoning_tokens") if details else 0,
            )
            has_data = True

        return result if has_data else None

    @staticmethod
    def _parse_complete_response(response: object) -> LLMStreamChunk:
        choices = _gobj(response, "choices")
        choice = cast(list[object], choices)[0] if choices else None
        message = _gobj(choice, "message")
        result = LLMStreamChunk(
            content=_gstr(message, "content"),
            finish_reason=_gstr(choice, "finish_reason"),
        )

        message_tool_calls = _gobj(message, "tool_calls")
        if message_tool_calls:
            result.tool_calls = [
                ToolCallPayload(
                    id=_gstr(tc, "id"),
                    function=ToolFunctionPayload(
                        name=_gstr(_gobj(tc, "function"), "name"),
                        arguments=_gstr(_gobj(tc, "function"), "arguments"),
                    ),
                )
                for tc in cast(list[object], message_tool_calls)
            ]

        usage = _gobj(response, "usage")
        if usage is not None:
            details = _gobj(usage, "completion_tokens_details")
            result.usage = ProviderUsage(
                input_tokens=_gint(usage, "prompt_tokens"),
                output_tokens=_gint(usage, "completion_tokens"),
                reasoning_tokens=_gint(details, "reasoning_tokens") if details else 0,
            )

        return result

    @override
    async def stream_chat(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        params = self._build_request_params(
            messages=messages,
            tools=tools,
            stream=True,
            **kwargs,
        )

        logger.debug(
            "[LLM] Calling model={}, base_url={}, messages={}, tools={}",
            self.config.model,
            self.config.base_url,
            len(messages),
            len(tools) if tools else 0,
        )
        start_time = time.time()

        try:
            _create = cast(Callable[..., Awaitable[object]], self.client.chat.completions.create)
            _response = await _create(**params)
            response = cast(AsyncIterator[object], _response)

            first_chunk_time = None
            chunk_count = 0
            accumulated_tool_calls: dict[int, ToolCallPayload] = {}

            async for chunk in response:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    logger.debug(
                        "[LLM] First chunk received (TTFB: {:.2f}s)",
                        first_chunk_time - start_time,
                    )

                chunk_count += 1
                result = self._parse_stream_chunk(chunk, accumulated_tool_calls)
                if result:
                    if result.finish_reason:
                        logger.debug(
                            "[LLM] Stream completed: chunks={}, total_time={:.2f}s, finish_reason={}",
                            chunk_count,
                            time.time() - start_time,
                            result.finish_reason,
                        )
                    yield result

        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            self._raise_mapped_error("stream_chat", exc)

    @override
    async def complete(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> LLMStreamChunk:
        params = self._build_request_params(
            messages=messages,
            tools=tools,
            stream=False,
            **kwargs,
        )

        try:
            _create = cast(Callable[..., Awaitable[object]], self.client.chat.completions.create)
            _response = await _create(**params)
            return self._parse_complete_response(_response)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            self._raise_mapped_error("complete", exc)
