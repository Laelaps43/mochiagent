from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import cast, final, override
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from agent.core.compression import CompactionPayload, CompactionStage
from agent.core.llm import AdapterRegistry
from agent.core.llm.base import LLMProvider
from agent.core.llm.errors import LLMProviderError
from agent.core.loop._framework_protocol import FrameworkProtocol
from agent.core.loop.llm_turn_handler import LLMTurnHandler
from agent.core.message import Message as InternalMessage
from agent.core.message import TextPart, UserTextInput
from agent.core.message.info import AssistantMessageInfo
from agent.core.runtime.strategy_manager import AgentStrategyManager
from agent.core.session import SessionManager
from agent.core.session.context import SessionContext
from agent.types import (
    ContextBudget,
    Event,
    EventType,
    LLMConfig,
    LLMStreamChunk,
    ProviderUsage,
    SessionState,
    ToolDefinition,
    ToolCallPayload,
    ToolFunctionPayload,
)


type _StreamOutcome = Exception | list[LLMStreamChunk]


@final
class _FakeStorage:
    def __init__(self) -> None:
        self.saved_messages: list[tuple[str, InternalMessage]] = []

    async def save_message(self, session_id: str, msg: InternalMessage) -> None:
        self.saved_messages.append((session_id, msg))


@final
class _FakeSessionManager:
    def __init__(self, context: SessionContext) -> None:
        self._context = context
        self.storage = _FakeStorage()
        self.saved_metadata_session_ids: list[str] = []
        self.updated_states: list[tuple[str, SessionState]] = []

    async def get_session(self, session_id: str) -> SessionContext:
        if session_id != self._context.session_id:
            raise ValueError("session not found")
        return self._context

    async def save_session_metadata(self, session_id: str) -> None:
        self.saved_metadata_session_ids.append(session_id)

    async def update_state(self, session_id: str, new_state: SessionState) -> None:
        self.updated_states.append((session_id, new_state))


@final
class _FakeLLMProvider(LLMProvider):
    def __init__(self, config: LLMConfig, outcomes: list[_StreamOutcome]) -> None:
        super().__init__(config)
        self._outcomes = outcomes
        self.stream_calls: list[tuple[list[InternalMessage], list[ToolDefinition]]] = []

    @override
    async def stream_chat(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **_kwargs: object,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        self.stream_calls.append((messages, tools or []))
        if not self._outcomes:
            return
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        for chunk in outcome:
            yield chunk

    @override
    async def complete(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **_kwargs: object,
    ) -> LLMStreamChunk:
        del messages, tools
        return LLMStreamChunk(content="")


@final
class _FakeAdapterRegistry:
    def __init__(self, provider: _FakeLLMProvider) -> None:
        self.provider = provider
        self.seen_configs: list[LLMConfig] = []

    def get(self, config: LLMConfig) -> _FakeLLMProvider:
        self.seen_configs.append(config)
        return self.provider


@final
class _FakeStrategyManager:
    def __init__(self, payloads: list[CompactionPayload]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, object]] = []

    async def run_compaction(
        self,
        *,
        session_context: SessionContext,
        budget: ContextBudget,
        llm_config: LLMConfig,
        llm_provider: object,
        agent_name: str | None,
        stage: CompactionStage,
        error: str | None,
        emit_event: Callable[[Event], Awaitable[None]] | None,
    ) -> CompactionPayload:
        self.calls.append(
            {
                "session_context": session_context,
                "budget": budget,
                "llm_config": llm_config,
                "llm_provider": llm_provider,
                "agent_name": agent_name,
                "stage": stage,
                "error": error,
                "emit_event": emit_event,
            }
        )
        if self.payloads:
            return self.payloads.pop(0)
        return CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)


@final
class _FakeToolRegistry:
    def get_definitions(self) -> list[ToolDefinition]:
        return []


@final
class _FakeAgentContext:
    def __init__(self, llm_config: LLMConfig, strategy_manager: _FakeStrategyManager) -> None:
        self.llm_config = llm_config
        self.strategy_manager = strategy_manager
        self.resolve_calls: list[tuple[str, str]] = []

    def resolve_llm_config_for_agent(self, agent_name: str, profile_id: str) -> LLMConfig:
        self.resolve_calls.append((agent_name, profile_id))
        return self.llm_config


@final
class _FakeAgent:
    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        strategy_manager: _FakeStrategyManager,
        system_prompt: str | None = None,
    ) -> None:
        self.tool_registry = _FakeToolRegistry()
        self.context = _FakeAgentContext(llm_config, strategy_manager)
        self._system_prompt = system_prompt

    def get_system_prompt(self, _context: SessionContext) -> str | None:
        return self._system_prompt


@final
class _FakeFramework:
    def __init__(self, agent: _FakeAgent | None) -> None:
        self.agent = agent

    def get_agent(self, _agent_name: str) -> _FakeAgent | None:
        return self.agent


def _make_context(
    *, session_id: str = "sess-1", model_profile_id: str = "openai:gpt-4o-mini"
) -> SessionContext:
    context = SessionContext(
        session_id=session_id, model_profile_id=model_profile_id, agent_name="test_agent"
    )
    _ = context.build_user_message(parts=[UserTextInput(text="hello")])
    return context


def _make_llm_config(*, max_overflow_retries: int = 2) -> LLMConfig:
    return LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model="gpt-4o-mini",
        api_key=SecretStr("test-key"),
        context_window_tokens=100,
        max_overflow_retries=max_overflow_retries,
    )


def _make_handler(
    *,
    context: SessionContext,
    llm_outcomes: list[_StreamOutcome],
    compaction_payloads: list[CompactionPayload],
    system_prompt: str | None = None,
) -> tuple[LLMTurnHandler, _FakeSessionManager, _FakeLLMProvider, list[Event]]:
    session_manager = _FakeSessionManager(context)
    llm_config = _make_llm_config()
    provider = _FakeLLMProvider(llm_config, llm_outcomes)
    adapter_registry = _FakeAdapterRegistry(provider)
    strategy_manager = _FakeStrategyManager(compaction_payloads)
    agent = _FakeAgent(
        llm_config=llm_config,
        strategy_manager=strategy_manager,
        system_prompt=system_prompt,
    )
    framework = _FakeFramework(agent)
    events: list[Event] = []

    async def emit_event(event: Event) -> None:
        events.append(event)

    handler = LLMTurnHandler(
        session_manager=cast(SessionManager, cast(object, session_manager)),
        adapter_registry=cast(AdapterRegistry, cast(object, adapter_registry)),
        framework=cast(FrameworkProtocol, cast(object, framework)),
        emit_event=emit_event,
    )
    return handler, session_manager, provider, events


@pytest.mark.asyncio
async def test_run_raises_when_model_profile_id_missing() -> None:
    context = _make_context()
    context.model_profile_id = None
    handler, _, _, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="ok", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    with pytest.raises(ValueError, match="has no model_profile_id"):
        _ = await handler.run(context.session_id)


@pytest.mark.asyncio
async def test_run_raises_when_agent_not_found() -> None:
    context = _make_context()
    session_manager = _FakeSessionManager(context)
    provider = _FakeLLMProvider(
        _make_llm_config(),
        [[LLMStreamChunk(content="ok", finish_reason="stop")]],
    )
    adapter_registry = _FakeAdapterRegistry(provider)
    framework = _FakeFramework(None)

    async def emit_event(_event: Event) -> None:
        return None

    handler = LLMTurnHandler(
        session_manager=cast(SessionManager, cast(object, session_manager)),
        adapter_registry=cast(AdapterRegistry, cast(object, adapter_registry)),
        framework=cast(FrameworkProtocol, cast(object, framework)),
        emit_event=emit_event,
    )

    with pytest.raises(ValueError, match="not found"):
        _ = await handler.run(context.session_id)


@pytest.mark.asyncio
async def test_run_happy_path_streams_text_and_returns_content() -> None:
    context = _make_context()
    handler, session_manager, _, events = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="hello", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    result = await handler.run(context.session_id)

    assert result.content == "hello"
    assert result.finish_reason == "stop"
    assert result.thinking == ""
    assert context.current_message is not None
    current = context.current_message
    assert isinstance(current.parts[-1], TextPart)
    assert current.parts[-1].text == "hello"
    assert result.context_budget.source == "estimated"
    assert session_manager.saved_metadata_session_ids == [context.session_id]
    assert events[-1].type == EventType.PART_CREATED


@pytest.mark.asyncio
async def test_run_emits_reasoning_before_content() -> None:
    context = _make_context()
    handler, _, _, events = _make_handler(
        context=context,
        llm_outcomes=[
            [
                LLMStreamChunk(thinking="pondering "),
                LLMStreamChunk(content="answer", finish_reason="stop"),
            ]
        ],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    result = await handler.run(context.session_id)

    assert result.content == "answer"
    assert result.thinking == "pondering "
    assert context.current_message is not None
    current_parts = context.current_message.parts
    assert current_parts[0].type == "reasoning"
    assert current_parts[1].type == "text"
    assert any(event.type == EventType.LLM_THINKING for event in events)


@pytest.mark.asyncio
async def test_run_emits_reasoning_when_stream_ends_with_thinking() -> None:
    context = _make_context()
    handler, _, _, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(thinking="tail-thinking", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    result = await handler.run(context.session_id)

    assert result.content == ""
    assert result.thinking == "tail-thinking"
    assert context.current_message is not None
    assert context.current_message.parts[0].type == "reasoning"


@pytest.mark.asyncio
async def test_run_accumulates_tool_calls_and_provider_usage() -> None:
    context = _make_context()
    tool_call_1 = ToolCallPayload(
        id="call-1",
        function=ToolFunctionPayload(name="read", arguments='{"path":"a"}'),
    )
    tool_call_2 = ToolCallPayload(
        id="call-2",
        function=ToolFunctionPayload(name="write", arguments='{"path":"b"}'),
    )
    handler, _, _, _ = _make_handler(
        context=context,
        llm_outcomes=[
            [
                LLMStreamChunk(tool_calls=[tool_call_1]),
                LLMStreamChunk(
                    content="done",
                    tool_calls=[tool_call_2],
                    finish_reason="tool_calls",
                    usage=ProviderUsage(input_tokens=11, output_tokens=22, reasoning_tokens=3),
                ),
            ]
        ],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    result = await handler.run(context.session_id)

    assert [tc.id for tc in result.tool_calls] == ["call-1", "call-2"]
    assert result.tokens.input == 11
    assert result.tokens.output == 22
    assert result.tokens.reasoning == 3
    assert result.context_budget.source == "provider"


@pytest.mark.asyncio
async def test_run_invokes_on_compaction_applied_for_pre_compaction() -> None:
    context = _make_context()
    handler, _, _, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="ok", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload(applied=True, reason="applied", stage="pre_call")],
    )
    on_compaction_applied_mock = AsyncMock()
    setattr(handler, "_on_compaction_applied", on_compaction_applied_mock)

    _ = await handler.run(context.session_id)

    on_compaction_applied_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_retries_on_context_overflow_when_compaction_applied() -> None:
    context = _make_context()
    overflow = LLMProviderError(code="context_length_exceeded", message="overflow")
    handler, _, provider, _ = _make_handler(
        context=context,
        llm_outcomes=[overflow, [LLMStreamChunk(content="after-retry", finish_reason="stop")]],
        compaction_payloads=[
            CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value),
            CompactionPayload(
                applied=True, reason="overflow", stage=CompactionStage.OVERFLOW_ERROR.value
            ),
        ],
    )

    result = await handler.run(context.session_id)

    assert result.content == "after-retry"
    assert len(provider.stream_calls) == 2


@pytest.mark.asyncio
async def test_run_reraises_on_overflow_when_retry_limit_reached() -> None:
    context = _make_context()
    overflow = LLMProviderError(code="context_length_exceeded", message="overflow")

    session_manager = _FakeSessionManager(context)
    provider = _FakeLLMProvider(_make_llm_config(max_overflow_retries=0), [overflow])
    adapter_registry = _FakeAdapterRegistry(provider)
    strategy_manager = _FakeStrategyManager(
        [CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)]
    )
    agent = _FakeAgent(
        llm_config=_make_llm_config(max_overflow_retries=0),
        strategy_manager=strategy_manager,
    )
    framework = _FakeFramework(agent)

    async def emit_event(_event: Event) -> None:
        return None

    handler = LLMTurnHandler(
        session_manager=cast(SessionManager, cast(object, session_manager)),
        adapter_registry=cast(AdapterRegistry, cast(object, adapter_registry)),
        framework=cast(FrameworkProtocol, cast(object, framework)),
        emit_event=emit_event,
    )

    with pytest.raises(LLMProviderError, match="overflow"):
        _ = await handler.run(context.session_id)


@pytest.mark.asyncio
async def test_run_reraises_on_overflow_when_compaction_not_applied() -> None:
    context = _make_context()
    overflow = LLMProviderError(code="context_length_exceeded", message="overflow")
    handler, _, _, _ = _make_handler(
        context=context,
        llm_outcomes=[overflow],
        compaction_payloads=[
            CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value),
            CompactionPayload.noop(stage=CompactionStage.OVERFLOW_ERROR.value),
        ],
    )

    with pytest.raises(LLMProviderError, match="overflow"):
        _ = await handler.run(context.session_id)


@pytest.mark.asyncio
async def test_run_reraises_non_overflow_error() -> None:
    context = _make_context()
    handler, _, _, _ = _make_handler(
        context=context,
        llm_outcomes=[RuntimeError("network-down")],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    with pytest.raises(RuntimeError, match="network-down"):
        _ = await handler.run(context.session_id)


def test_extract_provider_error_from_exception_and_cause_chain() -> None:
    provider_error = LLMProviderError(code="X", message="provider")
    direct = LLMTurnHandler.extract_provider_error(provider_error)
    assert direct is provider_error

    wrapped = RuntimeError("outer")
    wrapped.__cause__ = provider_error
    from_cause = LLMTurnHandler.extract_provider_error(cast(Exception, wrapped))
    assert from_cause is provider_error

    missing = LLMTurnHandler.extract_provider_error(ValueError("plain"))
    assert missing is None


def test_resolve_error_payload_with_and_without_provider_error() -> None:
    provider_error = LLMProviderError(code="RATE", message="rate limited", hint="retry later")
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = provider_error

    msg, code, hint = LLMTurnHandler.resolve_error_payload(cast(Exception, wrapped))
    assert (msg, code, hint) == ("rate limited", "RATE", "retry later")

    msg2, code2, hint2 = LLMTurnHandler.resolve_error_payload(ValueError("bad"))
    assert msg2 == "ValueError: bad"
    assert code2 is None
    assert hint2 is None


def test_is_context_overflow_error_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Exception] = []

    def _fake_checker(exc: Exception) -> bool:
        seen.append(exc)
        return True

    monkeypatch.setattr("agent.core.loop.llm_turn_handler._is_context_overflow", _fake_checker)

    exc = RuntimeError("x")
    assert LLMTurnHandler.is_context_overflow_error(cast(Exception, exc)) is True
    assert seen == [exc]


@pytest.mark.asyncio
async def test_persist_session_metadata_calls_session_manager() -> None:
    context = _make_context()
    handler, session_manager, _, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="ok", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    persist_session_metadata = cast(
        Callable[[str], Awaitable[None]],
        getattr(handler, "_persist_session_metadata"),
    )
    await persist_session_metadata(context.session_id)

    assert session_manager.saved_metadata_session_ids == [context.session_id]


@pytest.mark.asyncio
async def test_on_compaction_applied_saves_compaction_message_and_metadata() -> None:
    context = _make_context()
    compaction_message = InternalMessage.create_compaction(
        session_id=context.session_id,
        summary="compressed",
        compacted_count=1,
    )
    context.messages.append(compaction_message)

    handler, session_manager, _, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="ok", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    on_compaction_applied = cast(
        Callable[[SessionContext], Awaitable[None]],
        getattr(handler, "_on_compaction_applied"),
    )
    await on_compaction_applied(context)

    assert len(session_manager.storage.saved_messages) == 1
    saved_session_id, saved_message = session_manager.storage.saved_messages[0]
    assert saved_session_id == context.session_id
    assert saved_message.message_id == compaction_message.message_id
    assert session_manager.saved_metadata_session_ids == [context.session_id]


@pytest.mark.asyncio
async def test_run_context_compaction_forwards_correct_arguments() -> None:
    context = _make_context()
    llm_config = _make_llm_config()
    payload = CompactionPayload(applied=True, reason="ok", stage=CompactionStage.PRE_CALL.value)

    strategy_manager = AgentStrategyManager()
    run_compaction_mock = AsyncMock(return_value=payload)
    setattr(strategy_manager, "run_compaction", run_compaction_mock)

    handler, _, provider, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="ok", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
    )

    run_context_compaction = cast(
        Callable[..., Awaitable[CompactionPayload]],
        getattr(handler, "_run_context_compaction"),
    )
    result = await run_context_compaction(
        context=context,
        budget=ContextBudget(),
        llm_config=llm_config,
        llm=provider,
        strategy_manager=strategy_manager,
        stage=CompactionStage.OVERFLOW_ERROR,
        error="too long",
    )

    assert result == payload
    run_compaction_mock.assert_awaited_once()
    kwargs = cast(dict[str, object], run_compaction_mock.call_args.kwargs)
    assert kwargs["session_context"] is context
    assert kwargs["llm_config"] == llm_config
    assert kwargs["llm_provider"] is provider
    assert kwargs["agent_name"] == context.agent_name
    assert kwargs["stage"] == CompactionStage.OVERFLOW_ERROR
    assert kwargs["error"] == "too long"
    assert callable(kwargs["emit_event"])


@pytest.mark.asyncio
async def test_run_reuses_existing_current_assistant_message_and_includes_system_prompt() -> None:
    context = _make_context()
    existing = context.build_assistant_message(
        parent_id=context.messages[0].message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    assert isinstance(existing.info, AssistantMessageInfo)

    handler, _, provider, _ = _make_handler(
        context=context,
        llm_outcomes=[[LLMStreamChunk(content="reused", finish_reason="stop")]],
        compaction_payloads=[CompactionPayload.noop(stage=CompactionStage.PRE_CALL.value)],
        system_prompt="system-guidance",
    )

    result = await handler.run(context.session_id)

    assert result.message_id == existing.message_id
    assert provider.stream_calls
    messages, _tools = provider.stream_calls[0]
    assert messages[0].role == "system"
