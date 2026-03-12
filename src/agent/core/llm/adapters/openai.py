"""
OpenAI Compatible Adapter - 使用OpenAI SDK支持OpenAI兼容的厂商
支持DeepSeek, OpenAI, Azure, Kimi, Qwen等
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import NoReturn, cast, override

from loguru import logger
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from agent.core.llm.base import LLMProvider
from agent.core.llm.errors import (
    LLMProtocolError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTransportError,
)
from agent.core.message import Message as InternalMessage
from agent.core.security import redact_text
from agent.types import (
    LLMConfig,
    LLMStreamChunk,
    ProviderUsage,
    ToolCallPayload,
    ToolDefinition,
    ToolFunctionPayload,
)


@dataclass(slots=True)
class _ResponseMeta:
    status_code: int | None = None
    content_type: str | None = None
    x_log_id: str | None = None
    provider_code: str | None = None
    provider_message: str | None = None
    response_body: str | None = None


class OpenAIAdapter(LLMProvider):
    """
    OpenAI兼容适配器
    使用官方OpenAI SDK，支持所有兼容OpenAI接口的LLM
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.openai_max_retries: int = max(0, config.openai_max_retries)
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=self.openai_max_retries,
        )

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
            params["tools"] = self.prepare_tools(tools)
            params["tool_choice"] = "auto"
        params.update(self.config.extra_params)
        params.update(kwargs)
        if stream:
            params["stream_options"] = {"include_usage": True}
        return params

    @staticmethod
    def _parse_error_body(body_text: str) -> tuple[str | None, str | None]:
        """从 JSON error body 中提取 provider_code 和 provider_message。"""
        try:
            payload = cast("dict[str, object]", json.loads(body_text))
            error_obj = payload.get("error")
            if not isinstance(error_obj, dict):
                return None, None
            error = cast("dict[str, object]", error_obj)
            code: object | None = error.get("code")
            message: object | None = error.get("message")
            return (
                str(code) if code is not None else None,
                redact_text(str(message)) if message else None,
            )
        except Exception:
            return None, None

    @staticmethod
    def _extract_response_meta(exc: Exception) -> _ResponseMeta:
        response: object = getattr(exc, "response", None)
        if response is None:
            return _ResponseMeta()

        try:
            headers = cast("dict[str, str]", getattr(response, "headers", None) or {})
            body_text = str(getattr(response, "text", "") or "")
            if len(body_text) > 1200:
                body_text = body_text[:1200] + "...(truncated)"

            provider_code, provider_message = OpenAIAdapter._parse_error_body(body_text)

            return _ResponseMeta(
                status_code=cast("int | None", getattr(response, "status_code", None)),
                content_type=headers.get("content-type", ""),
                x_log_id=headers.get("x-log-id", ""),
                provider_code=provider_code,
                provider_message=provider_message,
                response_body=body_text or None,
            )
        except Exception:
            return _ResponseMeta()

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

        response: object = getattr(exc, "response", None)
        if response is not None:
            try:
                headers = cast("dict[str, str]", getattr(response, "headers", None) or {})
                content_type: str = headers.get("content-type", "")
                body_text: str = str(getattr(response, "text", "") or "")
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
        tool_call: ChoiceDeltaToolCall,
        accumulated_tool_calls: dict[int, ToolCallPayload],
    ) -> None:
        idx = tool_call.index
        if idx not in accumulated_tool_calls:
            accumulated_tool_calls[idx] = ToolCallPayload()

        entry = accumulated_tool_calls[idx]

        if tool_call.id:
            entry.id = tool_call.id

        fn = tool_call.function
        if not fn:
            return

        if fn.name:
            entry.function.name = fn.name

        if fn.arguments:
            entry.function.arguments += fn.arguments

    @staticmethod
    def _parse_stream_chunk(
        chunk: ChatCompletionChunk,
        accumulated_tool_calls: dict[int, ToolCallPayload],
    ) -> LLMStreamChunk | None:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            return None

        result = LLMStreamChunk()
        delta = choice.delta
        has_data = False

        if delta:
            if delta.content:
                result.content = delta.content
                has_data = True

            # reasoning_content / reasoning 是部分厂商（DeepSeek 等）的扩展字段
            reasoning_content: object | None = getattr(delta, "reasoning_content", None) or getattr(
                delta, "reasoning", None
            )
            if reasoning_content:
                result.thinking = str(reasoning_content)
                has_data = True

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    OpenAIAdapter._merge_tool_call_delta(tc, accumulated_tool_calls)

        if choice.finish_reason:
            result.finish_reason = choice.finish_reason
            has_data = True
            if accumulated_tool_calls:
                ordered_calls = [accumulated_tool_calls[k] for k in sorted(accumulated_tool_calls)]
                result.tool_calls = ordered_calls

        usage = chunk.usage
        if usage is not None:
            details = usage.completion_tokens_details
            result.usage = ProviderUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                reasoning_tokens=details.reasoning_tokens or 0 if details else 0,
            )
            has_data = True

        return result if has_data else None

    @staticmethod
    def _parse_complete_response(response: ChatCompletion) -> LLMStreamChunk:
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        result = LLMStreamChunk(
            content=message.content or "" if message else "",
            finish_reason=choice.finish_reason if choice else "",
        )

        if message and message.tool_calls:
            result.tool_calls = [
                ToolCallPayload(
                    id=tc.id,
                    function=ToolFunctionPayload(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in message.tool_calls
                if isinstance(tc, ChatCompletionMessageFunctionToolCall)
            ]

        usage = response.usage
        if usage is not None:
            details = usage.completion_tokens_details
            result.usage = ProviderUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                reasoning_tokens=details.reasoning_tokens or 0 if details else 0,
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
            _create = cast(
                Callable[..., Awaitable[AsyncStream[ChatCompletionChunk]]],
                self.client.chat.completions.create,
            )
            response = await _create(**params)

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
            _create = cast(
                Callable[..., Awaitable[ChatCompletion]],
                self.client.chat.completions.create,
            )
            response = await _create(**params)
            return self._parse_complete_response(response)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            self._raise_mapped_error("complete", exc)
