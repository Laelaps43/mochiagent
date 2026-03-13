from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast, final
from unittest.mock import MagicMock, patch

import pytest

from agent.core.bus import MessageBus
from agent.core.compression import CompactionPayload
from agent.core.llm import AdapterRegistry
from agent.core.loop._framework_protocol import FrameworkProtocol
from agent.core.loop.event_loop import AgentEventLoop
from agent.core.loop.turn_result import LLMTurnResult
from agent.core.message import ToolPart, UserTextInput
from agent.core.session import SessionManager
from agent.core.session.context import SessionContext
from agent.types import (
    ContextBudget,
    Event,
    EventType,
    SessionState,
    TokenUsage,
    ToolCallPayload,
    ToolFunctionPayload,
    ToolResult,
)


def _make_tool_call(
    *,
    call_id: str = "call_1",
    name: str = "echo",
    arguments: str = "{}",
) -> ToolCallPayload:
    return ToolCallPayload(
        id=call_id,
        type="function",
        function=ToolFunctionPayload(name=name, arguments=arguments),
    )


def _make_turn_result(
    *,
    message_id: str,
    finish_reason: str | None,
    tool_calls: list[ToolCallPayload] | None = None,
) -> LLMTurnResult:
    return LLMTurnResult(
        content="assistant-content",
        thinking="assistant-thinking",
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        tokens=TokenUsage(input_tokens=3, output_tokens=5, reasoning_tokens=1),
        context_budget=ContextBudget(total_tokens=1024, used_tokens=9, remaining_tokens=1015),
        context_compaction=CompactionPayload.noop(stage="turn"),
        context_compaction_events=[CompactionPayload.noop(stage="turn")],
        message_id=message_id,
    )


def _make_context(*, session_id: str = "sess_1", agent_name: str = "test_agent") -> SessionContext:
    return SessionContext(
        session_id=session_id,
        model_profile_id="openai:gpt-4o-mini",
        agent_name=agent_name,
    )


@final
class _StubSessionManager:
    def __init__(self, contexts: dict[str, SessionContext]):
        self._contexts: dict[str, SessionContext] = contexts
        self._get_session_side_effects: list[SessionContext | Exception] = []
        self.storage: object = object()
        self.emitted_events: list[Event] = []
        self.updated_states: list[tuple[str, SessionState]] = []
        self.finished_messages: list[tuple[str, str, TokenUsage | None]] = []

    async def emit_to_session_listeners(self, session_id: str, event: Event) -> None:
        _ = session_id
        self.emitted_events.append(event)

    async def get_session(self, session_id: str) -> SessionContext:
        if self._get_session_side_effects:
            next_item = self._get_session_side_effects.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

        context = self._contexts.get(session_id)
        if context is None:
            raise ValueError(f"Session {session_id} not found")
        return context

    async def update_state(self, session_id: str, new_state: SessionState) -> None:
        context = self._contexts[session_id]
        context.update_state(new_state)
        self.updated_states.append((session_id, new_state))

    async def finish_assistant_message(
        self,
        *,
        session_id: str,
        tokens: TokenUsage | None = None,
        finish: str = "stop",
    ) -> None:
        context = self._contexts[session_id]
        context.finish_current_message(tokens=tokens, finish=finish)
        self.finished_messages.append((session_id, finish, tokens))

    def queue_get_session_side_effects(self, effects: list[SessionContext | Exception]) -> None:
        self._get_session_side_effects = effects


@final
class _StaticTurnHandler:
    def __init__(self, results: list[LLMTurnResult] | None = None, error: Exception | None = None):
        self._results: list[LLMTurnResult] = results or []
        self._error: Exception | None = error
        self.calls: int = 0

    async def run(self, _session_id: str) -> LLMTurnResult:
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self.calls <= len(self._results):
            return self._results[self.calls - 1]
        return self._results[-1]


@final
class _FakeToolExecutor:
    def __init__(self, results: list[ToolResult]):
        self._results: list[ToolResult] = results
        self.calls: int = 0

    async def execute_batch(self, _tool_calls: list[ToolCallPayload]) -> list[ToolResult]:
        self.calls += 1
        return self._results


@final
class _FakeStrategyManager:
    async def run_postprocess(
        self,
        *,
        agent_name: str,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: dict[str, object],
        storage: object,
    ) -> ToolResult:
        _ = (agent_name, session_id, tool_arguments, storage)
        return tool_result


@final
class _FakeAgentContext:
    def __init__(self):
        self.strategy_manager: _FakeStrategyManager = _FakeStrategyManager()


@final
class _FakeAgent:
    def __init__(self, results: list[ToolResult]):
        self.tool_executor: _FakeToolExecutor = _FakeToolExecutor(results)
        self.context: _FakeAgentContext = _FakeAgentContext()


def _make_framework(get_agent_result: object | None) -> FrameworkProtocol:
    framework = MagicMock()
    framework.get_agent = MagicMock(return_value=get_agent_result)
    return cast(FrameworkProtocol, framework)


def _make_loop(
    session_manager: _StubSessionManager,
    *,
    framework: FrameworkProtocol | None = None,
    max_iterations: int = 5,
) -> AgentEventLoop:
    bus = MessageBus(max_concurrent=4)
    adapter_registry = MagicMock()
    selected_framework = framework if framework is not None else _make_framework(None)
    return AgentEventLoop(
        bus=bus,
        session_manager=cast(SessionManager, cast(object, session_manager)),
        adapter_registry=cast(AdapterRegistry, adapter_registry),
        framework=selected_framework,
        max_iterations=max_iterations,
    )


async def _call_emit_event(loop: AgentEventLoop, event: Event) -> None:
    method = cast(Callable[[Event], Awaitable[None]], getattr(loop, "_emit_event"))
    await method(event)


async def _call_handle_session_terminated(loop: AgentEventLoop, event: Event) -> None:
    method = cast(Callable[[Event], Awaitable[None]], getattr(loop, "_handle_session_terminated"))
    await method(event)


async def _call_get_session_lock(loop: AgentEventLoop, session_id: str) -> object:
    method = cast(Callable[[str], Awaitable[object]], getattr(loop, "_get_session_lock"))
    return await method(session_id)


async def _call_emit_error_and_done(
    loop: AgentEventLoop,
    *,
    session_id: str,
    error_message: str,
    code: str | None = None,
    hint: str | None = None,
) -> None:
    method = cast(
        Callable[..., Awaitable[None]],
        getattr(loop, "_emit_error_and_done"),
    )
    await method(session_id=session_id, error_message=error_message, code=code, hint=hint)


async def _call_handle_user_message(loop: AgentEventLoop, event: Event) -> None:
    method = cast(Callable[[Event], Awaitable[None]], getattr(loop, "_handle_user_message"))
    await method(event)


async def _call_conversation_loop(loop: AgentEventLoop, session_id: str) -> None:
    method = cast(Callable[[str], Awaitable[None]], getattr(loop, "_conversation_loop"))
    await method(session_id)


async def _call_execute_tools(
    loop: AgentEventLoop,
    session_id: str,
    tool_calls: list[ToolCallPayload],
) -> list[ToolResult]:
    method = cast(
        Callable[[str, list[ToolCallPayload]], Awaitable[list[ToolResult]]],
        getattr(loop, "_execute_tools"),
    )
    return await method(session_id, tool_calls)


def test_init_subscribes_handlers_and_normalizes_max_iterations() -> None:
    context = _make_context()
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager, max_iterations=0)

    assert loop.max_iterations == 1
    assert loop.bus is not None
    subscribers = cast(
        dict[EventType, list[Callable[[Event], Awaitable[None]]]],
        getattr(loop.bus, "_subscribers"),
    )
    user_handler = cast(Callable[[Event], Awaitable[None]], getattr(loop, "_handle_user_message"))
    terminated_handler = cast(
        Callable[[Event], Awaitable[None]],
        getattr(loop, "_handle_session_terminated"),
    )
    assert user_handler in subscribers[EventType.MESSAGE_RECEIVED]
    assert terminated_handler in subscribers[EventType.SESSION_TERMINATED]


async def test_emit_event_with_and_without_session_id() -> None:
    context = _make_context()
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    await _call_emit_event(loop, Event(type=EventType.MESSAGE_DONE, session_id="sess_1", data={}))
    await _call_emit_event(loop, Event(type=EventType.MESSAGE_DONE, session_id="", data={}))

    assert len(session_manager.emitted_events) == 1
    assert session_manager.emitted_events[0].session_id == "sess_1"


async def test_get_and_remove_session_lock_and_handle_session_terminated() -> None:
    context = _make_context()
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    first_lock = await _call_get_session_lock(loop, "sess_1")
    second_lock = await _call_get_session_lock(loop, "sess_1")
    assert first_lock is second_lock

    await loop.remove_session_lock("sess_1")
    await loop.remove_session_lock("missing")

    lock_map = cast(dict[str, object], getattr(loop, "_session_locks"))
    assert "sess_1" not in lock_map

    _ = await _call_get_session_lock(loop, "sess_1")
    await _call_handle_session_terminated(
        loop, Event(type=EventType.SESSION_TERMINATED, session_id="sess_1", data={})
    )
    assert "sess_1" not in lock_map


async def test_emit_error_and_done_with_code_and_hint() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="hi")])
    assistant_message = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    await _call_emit_error_and_done(
        loop,
        session_id="sess_1",
        error_message="provider failure",
        code="PROVIDER_ERROR",
        hint="retry later",
    )

    assert len(session_manager.emitted_events) == 2
    llm_error, message_done = session_manager.emitted_events
    assert llm_error.type == EventType.LLM_ERROR
    assert llm_error.data["error"] == "provider failure"
    assert llm_error.data["code"] == "PROVIDER_ERROR"
    assert llm_error.data["hint"] == "retry later"
    assert message_done.type == EventType.MESSAGE_DONE
    assert message_done.data["message_id"] == assistant_message.message_id
    assert message_done.data["finish"] == "error"


async def test_emit_error_and_done_when_get_session_fails() -> None:
    context = _make_context()
    session_manager = _StubSessionManager({"sess_1": context})
    session_manager.queue_get_session_side_effects([RuntimeError("broken store")])
    loop = _make_loop(session_manager)

    await _call_emit_error_and_done(loop, session_id="sess_1", error_message="boom")

    assert len(session_manager.emitted_events) == 2
    llm_error, message_done = session_manager.emitted_events
    assert llm_error.type == EventType.LLM_ERROR
    assert "code" not in llm_error.data
    assert "hint" not in llm_error.data
    assert message_done.type == EventType.MESSAGE_DONE
    assert message_done.data["message_id"] is None
    assert message_done.data["finish"] == "error"


async def test_handle_user_message_happy_path_calls_conversation_loop() -> None:
    context = _make_context()
    _ = context.build_user_message([UserTextInput(text="hello")])
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    calls = {"count": 0}

    async def fake_conversation_loop(session_id: str) -> None:
        calls["count"] += 1
        assert session_id == "sess_1"

    setattr(loop, "_conversation_loop", fake_conversation_loop)

    await _call_handle_user_message(
        loop, Event(type=EventType.MESSAGE_RECEIVED, session_id="sess_1", data={})
    )
    assert calls["count"] == 1


async def test_handle_user_message_without_user_message_emits_invalid_state() -> None:
    context = _make_context()
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    await _call_handle_user_message(
        loop, Event(type=EventType.MESSAGE_RECEIVED, session_id="sess_1", data={})
    )

    assert len(session_manager.emitted_events) == 2
    llm_error, message_done = session_manager.emitted_events
    assert llm_error.type == EventType.LLM_ERROR
    assert llm_error.data["code"] == "INVALID_STATE"
    assert "No user message found" in str(llm_error.data["error"])
    assert message_done.type == EventType.MESSAGE_DONE


async def test_handle_user_message_general_exception_uses_resolved_payload() -> None:
    context = _make_context()
    _ = context.build_user_message([UserTextInput(text="hello")])
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    async def failing_loop(_session_id: str) -> None:
        raise RuntimeError("unexpected")

    setattr(loop, "_conversation_loop", failing_loop)

    with patch(
        "agent.core.loop.event_loop.LLMTurnHandler.resolve_error_payload",
        return_value=("handled failure", "HANDLED", "check model"),
    ):
        await _call_handle_user_message(
            loop, Event(type=EventType.MESSAGE_RECEIVED, session_id="sess_1", data={})
        )

    assert len(session_manager.emitted_events) == 2
    llm_error, message_done = session_manager.emitted_events
    assert llm_error.type == EventType.LLM_ERROR
    assert llm_error.data["error"] == "handled failure"
    assert llm_error.data["code"] == "HANDLED"
    assert llm_error.data["hint"] == "check model"
    assert message_done.type == EventType.MESSAGE_DONE


async def test_conversation_loop_single_turn_emits_done_and_sets_idle() -> None:
    context = _make_context()
    _ = context.build_user_message([UserTextInput(text="prompt")])
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)
    setattr(
        loop,
        "_llm_turn_handler",
        _StaticTurnHandler(
            [
                _make_turn_result(
                    message_id="msg_1",
                    finish_reason="stop",
                    tool_calls=[],
                )
            ]
        ),
    )

    await _call_conversation_loop(loop, "sess_1")

    assert context.state == SessionState.IDLE
    assert session_manager.finished_messages
    assert session_manager.finished_messages[-1][1] == "stop"
    assert len(session_manager.emitted_events) == 1
    assert session_manager.emitted_events[0].type == EventType.MESSAGE_DONE


async def test_conversation_loop_tool_calls_then_continue_and_done() -> None:
    context = _make_context()
    _ = context.build_user_message([UserTextInput(text="prompt")])
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)
    tool_call = _make_tool_call(call_id="call_1")
    setattr(
        loop,
        "_llm_turn_handler",
        _StaticTurnHandler(
            [
                _make_turn_result(
                    message_id="msg_a",
                    finish_reason="tool_calls",
                    tool_calls=[tool_call],
                ),
                _make_turn_result(message_id="msg_b", finish_reason="stop", tool_calls=[]),
            ]
        ),
    )

    execute_calls = {"count": 0}

    async def fake_execute_tools(
        session_id: str,
        tool_calls: list[ToolCallPayload],
    ) -> list[ToolResult]:
        execute_calls["count"] += 1
        assert session_id == "sess_1"
        assert len(tool_calls) == 1
        return []

    setattr(loop, "_execute_tools", fake_execute_tools)

    await _call_conversation_loop(loop, "sess_1")

    assert execute_calls["count"] == 1
    assert context.state == SessionState.IDLE
    assert session_manager.emitted_events[-1].type == EventType.MESSAGE_DONE


async def test_conversation_loop_max_iterations_exceeded_emits_error_and_done() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager, max_iterations=1)
    tool_call = _make_tool_call(call_id="call_1")
    setattr(
        loop,
        "_llm_turn_handler",
        _StaticTurnHandler(
            [
                _make_turn_result(
                    message_id="msg_tool",
                    finish_reason="tool_calls",
                    tool_calls=[tool_call],
                )
            ]
        ),
    )

    async def fake_execute_tools(
        _session_id: str,
        _tool_calls: list[ToolCallPayload],
    ) -> list[ToolResult]:
        return []

    setattr(loop, "_execute_tools", fake_execute_tools)

    await _call_conversation_loop(loop, "sess_1")

    assert context.state == SessionState.IDLE
    assert session_manager.finished_messages
    assert session_manager.finished_messages[-1][1] == "max_iterations_exceeded"
    assert len(session_manager.emitted_events) == 2
    assert session_manager.emitted_events[0].type == EventType.LLM_ERROR
    assert session_manager.emitted_events[0].data["code"] == "MAX_ITERATIONS_EXCEEDED"
    assert session_manager.emitted_events[1].type == EventType.MESSAGE_DONE
    assert session_manager.emitted_events[1].data["finish"] == "max_iterations_exceeded"


async def test_conversation_loop_exception_sets_error_finishes_and_reraises() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)
    setattr(loop, "_llm_turn_handler", _StaticTurnHandler(error=RuntimeError("loop crash")))

    with pytest.raises(RuntimeError, match="loop crash"):
        await _call_conversation_loop(loop, "sess_1")

    assert context.state == SessionState.ERROR
    assert session_manager.finished_messages
    assert session_manager.finished_messages[-1][1] == "error"
    assert context.current_message is None


async def test_conversation_loop_exception_cleanup_get_session_failure_not_raised() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    session_manager = _StubSessionManager({"sess_1": context})
    session_manager.queue_get_session_side_effects(
        [
            context,
            RuntimeError("cleanup failure"),
            context,
        ]
    )
    loop = _make_loop(session_manager)
    setattr(loop, "_llm_turn_handler", _StaticTurnHandler(error=RuntimeError("loop failure")))

    with pytest.raises(RuntimeError, match="loop failure"):
        await _call_conversation_loop(loop, "sess_1")

    assert context.state == SessionState.ERROR


async def test_execute_tools_no_current_message_returns_empty() -> None:
    context = _make_context()
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager)

    results = await _call_execute_tools(loop, "sess_1", [_make_tool_call()])
    assert results == []


async def test_execute_tools_agent_not_found_raises_runtime_error() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    session_manager = _StubSessionManager({"sess_1": context})
    framework = _make_framework(None)
    loop = _make_loop(session_manager, framework=framework)

    with pytest.raises(RuntimeError, match="not found"):
        _ = await _call_execute_tools(loop, "sess_1", [_make_tool_call()])


async def test_execute_tools_happy_path_emits_running_and_completed() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    call = _make_tool_call(call_id="call_1", name="echo", arguments='{"text":"hi"}')
    tool_result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result={"ok": True},
        success=True,
        summary="done",
    )
    agent = _FakeAgent([tool_result])
    framework = _make_framework(agent)
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager, framework=framework)

    results = await _call_execute_tools(loop, "sess_1", [call])

    assert len(results) == 1
    assert results[0].success is True
    part_events = [
        event for event in session_manager.emitted_events if event.type == EventType.PART_CREATED
    ]
    assert len(part_events) == 2
    assert part_events[0].data["type"] == "tool"
    assert cast(dict[str, object], part_events[0].data["state"])["status"] == "running"
    assert cast(dict[str, object], part_events[1].data["state"])["status"] == "completed"
    assert context.current_message is not None
    assert any(
        isinstance(part, ToolPart) and part.call_id == "call_1" and part.state.status == "completed"
        for part in context.current_message.parts
    )


async def test_execute_tools_failure_result_updates_to_error() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    call = _make_tool_call(call_id="call_1", name="echo", arguments="{}")
    tool_result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result=None,
        success=False,
        error="denied",
    )
    agent = _FakeAgent([tool_result])
    framework = _make_framework(agent)
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager, framework=framework)

    _ = await _call_execute_tools(loop, "sess_1", [call])

    part_events = [
        event for event in session_manager.emitted_events if event.type == EventType.PART_CREATED
    ]
    assert len(part_events) == 2
    assert cast(dict[str, object], part_events[1].data["state"])["status"] == "error"


async def test_execute_tools_skips_result_when_call_id_not_in_map() -> None:
    context = _make_context()
    user_message = context.build_user_message([UserTextInput(text="prompt")])
    _ = context.build_assistant_message(
        parent_id=user_message.message_id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    call = _make_tool_call(call_id="call_1", name="echo", arguments="{}")
    unknown_result = ToolResult(
        tool_call_id="call_unknown",
        tool_name="echo",
        result="ignored",
        success=True,
    )
    agent = _FakeAgent([unknown_result])
    framework = _make_framework(agent)
    session_manager = _StubSessionManager({"sess_1": context})
    loop = _make_loop(session_manager, framework=framework)

    results = await _call_execute_tools(loop, "sess_1", [call])

    assert len(results) == 1
    part_events = [
        event for event in session_manager.emitted_events if event.type == EventType.PART_CREATED
    ]
    assert len(part_events) == 1
    assert cast(dict[str, object], part_events[0].data["state"])["status"] == "running"
