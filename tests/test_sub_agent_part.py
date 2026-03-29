"""Tests for SubAgentPart and LLM message conversion."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import override

import pytest

from agent.core.message.part import (
    SubAgentPart,
    SubAgentStateCompleted,
    SubAgentStateError,
    SubAgentStateRunning,
    TimeInfo,
)
from agent.types import TokenUsage


def test_create_running() -> None:
    part = SubAgentPart.create_running(
        session_id="sess_1",
        message_id="msg_1",
        call_id="call_1",
        agent_name="explorer",
        prompt="find the bug",
        depth=2,
    )
    assert part.type == "subagent"
    assert part.agent_name == "explorer"
    assert part.depth == 2
    assert part.call_id == "call_1"
    assert isinstance(part.state, SubAgentStateRunning)
    assert part.state.prompt == "find the bug"


def test_update_to_completed() -> None:
    part = SubAgentPart.create_running(
        session_id="s",
        message_id="m",
        call_id="c",
        agent_name="sub1",
        prompt="hello",
    )
    tokens = TokenUsage(input_tokens=10, output_tokens=5)
    completed = part.update_to_completed(
        output="result text",
        child_session_id="sub_999",
        tokens=tokens,
    )
    assert isinstance(completed.state, SubAgentStateCompleted)
    assert completed.state.output == "result text"
    assert completed.state.child_session_id == "sub_999"
    assert completed.state.tokens.input_tokens == 10
    assert completed.state.time.end is not None
    assert completed.call_id == "c"
    assert completed.agent_name == "sub1"


def test_update_to_error() -> None:
    part = SubAgentPart.create_running(
        session_id="s",
        message_id="m",
        call_id="c",
        agent_name="sub1",
        prompt="hello",
    )
    errored = part.update_to_error(
        error="depth exceeded",
        child_session_id="sub_err",
    )
    assert isinstance(errored.state, SubAgentStateError)
    assert errored.state.error == "depth exceeded"
    assert errored.state.child_session_id == "sub_err"


def test_update_to_completed_rejects_non_running() -> None:
    part = SubAgentPart.create_running(
        session_id="s", message_id="m", call_id="c", agent_name="sub1", prompt="hello"
    )
    completed = part.update_to_completed(output="ok", child_session_id="sub_1")
    with pytest.raises(ValueError, match="expected 'running'"):
        _ = completed.update_to_completed(output="again", child_session_id="sub_2")


def test_update_to_error_rejects_non_running() -> None:
    part = SubAgentPart.create_running(
        session_id="s", message_id="m", call_id="c", agent_name="sub1", prompt="hello"
    )
    errored = part.update_to_error(error="fail")
    with pytest.raises(ValueError, match="expected 'running'"):
        _ = errored.update_to_error(error="fail again")


def test_prepare_messages_subagent_completed() -> None:
    """SubAgentPart converts to tool_call + tool_result in prepare_messages."""
    from agent.core.llm.base import LLMProvider
    from agent.core.llm.types import LLMStreamChunk
    from agent.core.message import Message
    from agent.core.message.info import AssistantMessageInfo
    from agent.types import LLMConfig, ToolDefinition

    config = LLMConfig(adapter="openai_compatible", provider="openai", model="gpt-4o")

    class _DummyProvider(LLMProvider):
        @override
        def stream_chat(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            **_kwargs: object,
        ) -> AsyncGenerator[LLMStreamChunk, None]:
            raise NotImplementedError

        @override
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            **_kwargs: object,
        ) -> LLMStreamChunk:
            raise NotImplementedError

    provider = _DummyProvider(config)

    part = SubAgentPart(
        session_id="s",
        message_id="m",
        call_id="call_sub",
        agent_name="explorer",
        state=SubAgentStateCompleted(
            prompt="find bugs",
            output="found 3 bugs",
            child_session_id="sub_x",
            tokens=TokenUsage(),
            time=TimeInfo(start=100, end=200),
        ),
    )
    msg = Message(
        info=AssistantMessageInfo(
            id="m",
            session_id="s",
            created_at=100,
            parent_id="p",
            model_id="gpt-4o",
            provider_id="openai",
        ),
        parts=[part],
    )

    result = provider.prepare_messages([msg])

    assert len(result) == 2
    assistant_msg = result[0]
    assert assistant_msg["role"] == "assistant"
    from typing import cast

    tool_calls_typed = cast(list[dict[str, object]], assistant_msg.get("tool_calls", []))
    assert len(tool_calls_typed) == 1
    tc = tool_calls_typed[0]
    func = cast(dict[str, object], tc["function"])
    assert func["name"] == "task"
    args = cast(dict[str, object], json.loads(str(func["arguments"])))
    assert args["agent_name"] == "explorer"
    assert args["prompt"] == "find bugs"

    tool_result_msg = result[1]
    assert tool_result_msg["role"] == "tool"
    assert tool_result_msg["content"] == "found 3 bugs"
    assert tool_result_msg["tool_call_id"] == "call_sub"


def test_prepare_messages_subagent_error() -> None:
    """SubAgentPart in error state converts to error tool_result."""
    from agent.core.llm.base import LLMProvider
    from agent.core.llm.types import LLMStreamChunk
    from agent.core.message import Message
    from agent.core.message.info import AssistantMessageInfo
    from agent.types import LLMConfig, ToolDefinition

    config = LLMConfig(adapter="openai_compatible", provider="openai", model="gpt-4o")

    class _DummyProvider(LLMProvider):
        @override
        def stream_chat(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            **_kwargs: object,
        ) -> AsyncGenerator[LLMStreamChunk, None]:
            raise NotImplementedError

        @override
        async def complete(
            self,
            messages: list[Message],
            tools: list[ToolDefinition] | None = None,
            **_kwargs: object,
        ) -> LLMStreamChunk:
            raise NotImplementedError

    provider = _DummyProvider(config)

    part = SubAgentPart(
        session_id="s",
        message_id="m",
        call_id="call_err",
        agent_name="sub1",
        state=SubAgentStateError(
            prompt="do thing",
            error="max depth",
            time=TimeInfo(start=100, end=200),
        ),
    )
    msg = Message(
        info=AssistantMessageInfo(
            id="m",
            session_id="s",
            created_at=100,
            parent_id="p",
            model_id="gpt-4o",
            provider_id="openai",
        ),
        parts=[part],
    )

    result = provider.prepare_messages([msg])
    assert len(result) == 2
    tool_result_msg = result[1]
    assert "Error: max depth" in str(tool_result_msg["content"])
