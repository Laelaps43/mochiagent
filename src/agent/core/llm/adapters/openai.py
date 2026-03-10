"""
OpenAI Compatible Adapter - 使用OpenAI SDK支持OpenAI兼容的厂商
支持DeepSeek, OpenAI, Azure, Kimi, Qwen等
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, NoReturn

from loguru import logger
from openai import AsyncOpenAI

from agent.constants import OPENAI_MAX_RETRIES
from agent.core.llm.base import LLMMessageInput, LLMProvider
from agent.core.llm.errors import (
    LLMProtocolError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTransportError,
)
from agent.core.security import redact_text
from agent.types import LLMConfig, LLMStreamChunk, Message as ChatMessage, ToolCallPayload, ToolDefinition

try:
    from openai import RateLimitError
except Exception:  # pragma: no cover

    class RateLimitError(Exception):
        pass


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
        self.openai_max_retries = self._resolve_max_retries(config.openai_max_retries)
        self.client = AsyncOpenAI(
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
    def _prepare_tools(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
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

    @staticmethod
    def _normalize_messages(messages: list[LLMMessageInput]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                normalized.append(message.model_dump(exclude_none=True, mode="json"))
                continue

            raise TypeError(
                f"Unsupported message type: {type(message).__name__}. "
                "Expected agent.types.Message."
            )
        return normalized

    def _build_request_params(
        self,
        *,
        messages: list[LLMMessageInput],
        tools: list[ToolDefinition] | None,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, object]:
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._normalize_messages(messages),
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

        response = exc.response if hasattr(exc, "response") else None
        if response is None:
            return meta

        try:
            meta.status_code = response.status_code if hasattr(response, "status_code") else None
            headers = response.headers if hasattr(response, "headers") else {}
            meta.content_type = headers.get("content-type", "")
            meta.x_log_id = headers.get("x-log-id", "")

            body_text = (response.text if hasattr(response, "text") else "") or ""
            if body_text:
                if len(body_text) > 1200:
                    body_text = body_text[:1200] + "...(truncated)"
                meta.response_body = body_text

                try:
                    payload = json.loads(body_text)
                    if isinstance(payload, dict):
                        error_obj = payload.get("error")
                        if isinstance(error_obj, dict):
                            if error_obj.get("code") is not None:
                                meta.provider_code = str(error_obj.get("code"))
                            if error_obj.get("message"):
                                meta.provider_message = redact_text(error_obj.get("message"))
                except Exception:
                    pass
        except Exception:
            pass

        return meta

    def _map_provider_exception(self, operation: str, exc: Exception) -> LLMProviderError:
        if isinstance(exc, LLMProviderError):
            return exc

        context = {
            "provider": self.config.provider,
            "model": self.config.model,
            "base_url": self.config.base_url,
        }

        if isinstance(exc, KeyError):
            missing_key = str(exc.args[0]) if exc.args else "unknown"
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
                **context,
            )

        meta = self._extract_response_meta(exc)
        status_code = meta.status_code
        provider_code = meta.provider_code
        provider_message = meta.provider_message
        x_log_id = meta.x_log_id

        if isinstance(exc, RateLimitError) or status_code == 429 or provider_code == "1302":
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
                **context,
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
            **context,
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

        response = exc.response if hasattr(exc, "response") else None
        if response is not None:
            try:
                content_type = response.headers.get("content-type", "")
                body_text = (response.text if hasattr(response, "text") else "") or ""
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
        tool_call: Any,
        accumulated_tool_calls: dict[int, ToolCallPayload],
    ) -> None:
        idx = tool_call.index if tool_call.index is not None else len(accumulated_tool_calls)
        if idx not in accumulated_tool_calls:
            accumulated_tool_calls[idx] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }

        entry = accumulated_tool_calls[idx]

        if tool_call.id:
            entry["id"] = tool_call.id

        if not tool_call.function:
            return

        if tool_call.function.name:
            entry["function"]["name"] = tool_call.function.name

        if tool_call.function.arguments:
            entry["function"]["arguments"] += tool_call.function.arguments

    @staticmethod
    def _parse_stream_chunk(
        chunk: Any,
        accumulated_tool_calls: dict[int, ToolCallPayload],
    ) -> LLMStreamChunk | None:
        choice = chunk.choices[0] if chunk.choices else None
        if not choice:
            return None

        result: LLMStreamChunk = {}
        delta = choice.delta

        if delta and delta.content:
            result["content"] = delta.content

        if delta and delta.tool_calls:
            for tool_call in delta.tool_calls:
                OpenAIAdapter._merge_tool_call_delta(tool_call, accumulated_tool_calls)

        if choice.finish_reason:
            result["finish_reason"] = choice.finish_reason
            if accumulated_tool_calls:
                ordered_calls = [accumulated_tool_calls[k] for k in sorted(accumulated_tool_calls)]
                result["tool_calls"] = ordered_calls
        usage = chunk.usage if hasattr(chunk, "usage") else None
        if usage is not None:
            if hasattr(usage, "model_dump"):
                result["usage"] = usage.model_dump(exclude_none=True)
            elif isinstance(usage, dict):
                result["usage"] = usage

        return result or None

    @staticmethod
    def _parse_complete_response(response: Any) -> LLMStreamChunk:
        choice = response.choices[0]
        message = choice.message
        result: LLMStreamChunk = {
            "content": message.content or "",
            "finish_reason": choice.finish_reason,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return result

    async def stream_chat(
        self,
        messages: list[LLMMessageInput],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
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
            response = await self.client.chat.completions.create(**params)

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
                    if "finish_reason" in result:
                        logger.debug(
                            "[LLM] Stream completed: chunks={}, total_time={:.2f}s, finish_reason={}",
                            chunk_count,
                            time.time() - start_time,
                            result.get("finish_reason"),
                        )
                    yield result

        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            self._raise_mapped_error("stream_chat", exc)

    async def complete(
        self,
        messages: list[LLMMessageInput],
        tools: list[ToolDefinition] | None = None,
        **kwargs: Any,
    ) -> LLMStreamChunk:
        params = self._build_request_params(
            messages=messages,
            tools=tools,
            stream=False,
            **kwargs,
        )

        try:
            response = await self.client.chat.completions.create(**params)
            return self._parse_complete_response(response)
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except Exception as exc:
            self._raise_mapped_error("complete", exc)
