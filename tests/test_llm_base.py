from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import final, override

from agent.core.llm.base import LLMProvider
from agent.core.message import UserTextInput
from agent.core.message.message import Message
from agent.core.message.part import (
    TimeInfo,
    ToolInput,
    ToolPart,
    ToolStateCompleted,
    ToolStateError,
)
from agent.core.session.context import SessionContext
from agent.types import (
    LLMConfig,
    LLMStreamChunk,
    ToolCallPayload,
    ToolDefinition,
    ToolFunctionPayload,
)


def _make_config() -> LLMConfig:
    return LLMConfig(adapter="openai_compatible", provider="test", model="m1")


@final
class _DummyProvider(LLMProvider):
    @override
    async def stream_chat(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        del messages, tools, kwargs
        yield LLMStreamChunk(content="chunk", finish_reason="stop")

    @override
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> LLMStreamChunk:
        del messages, tools, kwargs
        return LLMStreamChunk(content="done", finish_reason="stop")


def _build_messages() -> list[Message]:
    ctx = SessionContext(session_id="s1", model_profile_id="test:m1", agent_name="agent")
    user_message = ctx.build_user_message([UserTextInput(text="hello")])
    empty_assistant = ctx.build_assistant_message(
        user_message.message_id,
        provider_id="test",
        model_id="m1",
    )
    assistant_with_tools = ctx.build_assistant_message(
        user_message.message_id,
        provider_id="test",
        model_id="m1",
    )
    assistant_with_tools.add_part(
        UserTextInput(text="assistant text").to_part(
            ctx.session_id, assistant_with_tools.message_id
        )
    )

    running = ToolPart.create_running(
        session_id=ctx.session_id,
        message_id=assistant_with_tools.message_id,
        tool_call=ToolCallPayload(
            id="call-running",
            function=ToolFunctionPayload(name="run_tool", arguments='{"value": 1}'),
        ),
    )
    assistant_with_tools.add_part(running)
    assistant_with_tools.add_part(
        ToolPart(
            session_id=ctx.session_id,
            message_id=assistant_with_tools.message_id,
            call_id="call-completed",
            tool="done_tool",
            state=ToolStateCompleted(
                input=ToolInput(arguments='{"value": 2}'),
                output="raw output",
                summary="summary output",
                title="done_tool",
                time=TimeInfo(start=1, end=2),
            ),
        )
    )
    assistant_with_tools.add_part(
        ToolPart(
            session_id=ctx.session_id,
            message_id=assistant_with_tools.message_id,
            call_id="call-error",
            tool="bad_tool",
            state=ToolStateError(
                input=ToolInput(arguments='{"value": 3}'),
                error="boom",
                time=TimeInfo(start=3, end=4),
            ),
        )
    )
    compaction = Message.create_compaction(session_id="s1", summary="compacted summary")
    return [user_message, empty_assistant, compaction, assistant_with_tools]


def test_prepare_messages_covers_text_tool_states_empty_and_compaction_role() -> None:
    provider = _DummyProvider(_make_config())

    result = provider.prepare_messages(_build_messages())

    assert len(result) == 5
    assert result[0] == {"role": "user", "content": "hello"}
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "COMPACTION_SUMMARY\ncompacted summary"
    assert result[2]["role"] == "assistant"
    assert result[2]["content"] == "assistant text"
    assert result[2]["tool_calls"] == [
        {
            "id": "call-running",
            "type": "function",
            "function": {"name": "run_tool", "arguments": '{"value": 1}'},
        },
        {
            "id": "call-completed",
            "type": "function",
            "function": {"name": "done_tool", "arguments": '{"value": 2}'},
        },
        {
            "id": "call-error",
            "type": "function",
            "function": {"name": "bad_tool", "arguments": '{"value": 3}'},
        },
    ]
    assert result[3] == {
        "role": "tool",
        "content": "summary output",
        "tool_call_id": "call-completed",
    }
    assert result[4] == {
        "role": "tool",
        "content": "Error: boom",
        "tool_call_id": "call-error",
    }


def test_prepare_tools_converts_definitions() -> None:
    tools = [
        ToolDefinition(
            name="calculator",
            description="Add numbers",
            parameters={"type": "object", "properties": {"x": {"type": "number"}}},
        )
    ]

    result = LLMProvider.prepare_tools(tools)

    assert result == [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Add numbers",
                "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
            },
        }
    ]


async def test_llm_provider_subclass_instantiation_and_base_abstract_bodies() -> None:
    provider = _DummyProvider(_make_config())

    stream = provider.stream_chat([])
    first = await anext(stream)

    assert provider.config.provider == "test"
    assert first.content == "chunk"
    assert first.finish_reason == "stop"
    assert await provider.complete([]) == LLMStreamChunk(content="done", finish_reason="stop")
    assert LLMProvider.stream_chat(provider, []) is None
    assert await LLMProvider.complete(provider, []) is None


def test_llm_provider_declares_expected_abstract_methods() -> None:
    assert LLMProvider.__abstractmethods__ == frozenset({"stream_chat", "complete"})
