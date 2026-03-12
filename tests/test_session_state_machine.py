from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

import pytest

from agent.core.session.state import SessionStateMachine
from agent.types import SessionState


def _terminate(sm: SessionStateMachine) -> Awaitable[None]:
    return cast(Callable[[], Awaitable[None]], getattr(sm, "terminate"))()


@pytest.fixture
def sm() -> SessionStateMachine:
    return SessionStateMachine(session_id="test-session")


def test_initial_state_is_idle(sm: SessionStateMachine):
    assert sm.state == SessionState.IDLE.value
    assert sm.current_state == SessionState.IDLE


async def test_idle_to_processing(sm: SessionStateMachine):
    result = await sm.transition_to(SessionState.PROCESSING)
    assert result is True
    assert sm.current_state == SessionState.PROCESSING


async def test_processing_to_streaming(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    result = await sm.transition_to(SessionState.STREAMING)
    assert result is True
    assert sm.current_state == SessionState.STREAMING


async def test_processing_to_waiting_tool(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    result = await sm.transition_to(SessionState.WAITING_TOOL)
    assert result is True
    assert sm.current_state == SessionState.WAITING_TOOL


async def test_processing_to_idle_via_complete(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    result = await sm.transition_to(SessionState.IDLE)
    assert result is True
    assert sm.current_state == SessionState.IDLE


async def test_processing_to_error(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    result = await sm.transition_to(SessionState.ERROR)
    assert result is True
    assert sm.current_state == SessionState.ERROR


async def test_streaming_to_idle(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.STREAMING)
    result = await sm.transition_to(SessionState.IDLE)
    assert result is True
    assert sm.current_state == SessionState.IDLE


async def test_streaming_to_waiting_tool(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.STREAMING)
    result = await sm.transition_to(SessionState.WAITING_TOOL)
    assert result is True
    assert sm.current_state == SessionState.WAITING_TOOL


async def test_waiting_tool_to_processing(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.WAITING_TOOL)
    result = await sm.transition_to(SessionState.PROCESSING)
    assert result is True
    assert sm.current_state == SessionState.PROCESSING


async def test_error_to_idle_via_reset(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.ERROR)
    result = await sm.transition_to(SessionState.IDLE)
    assert result is True
    assert sm.current_state == SessionState.IDLE


async def test_invalid_transition_returns_false(sm: SessionStateMachine):
    result = await sm.transition_to(SessionState.WAITING_TOOL)
    assert result is False
    assert sm.current_state == SessionState.IDLE


async def test_same_state_returns_true(sm: SessionStateMachine):
    result = await sm.transition_to(SessionState.IDLE)
    assert result is True
    assert sm.current_state == SessionState.IDLE


async def test_terminate_from_idle(sm: SessionStateMachine):
    await _terminate(sm)
    assert sm.current_state == SessionState.TERMINATED


async def test_terminate_from_processing(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    await _terminate(sm)
    assert sm.current_state == SessionState.TERMINATED


async def test_terminate_from_streaming(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.STREAMING)
    await _terminate(sm)
    assert sm.current_state == SessionState.TERMINATED


async def test_terminate_from_error(sm: SessionStateMachine):
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.ERROR)
    await _terminate(sm)
    assert sm.current_state == SessionState.TERMINATED


async def test_on_state_change_callback_called():
    calls: list[tuple[str, str, str]] = []

    async def callback(session_id: str, from_state: str, to_state: str) -> None:
        calls.append((session_id, from_state, to_state))

    sm = SessionStateMachine(session_id="cb-session", on_state_change=callback)
    _ = await sm.transition_to(SessionState.PROCESSING)

    assert len(calls) == 1
    assert calls[0] == ("cb-session", SessionState.IDLE.value, SessionState.PROCESSING.value)


async def test_on_state_change_callback_multiple_transitions():
    calls: list[tuple[str, str, str]] = []

    async def callback(session_id: str, from_state: str, to_state: str) -> None:
        calls.append((session_id, from_state, to_state))

    sm = SessionStateMachine(session_id="multi-cb", on_state_change=callback)
    _ = await sm.transition_to(SessionState.PROCESSING)
    _ = await sm.transition_to(SessionState.STREAMING)
    _ = await sm.transition_to(SessionState.IDLE)

    assert len(calls) == 3
    assert calls[1] == ("multi-cb", SessionState.PROCESSING.value, SessionState.STREAMING.value)
    assert calls[2] == ("multi-cb", SessionState.STREAMING.value, SessionState.IDLE.value)


def test_can_transition_true_for_valid_trigger(sm: SessionStateMachine):
    assert sm.can_transition("start_processing") is True


def test_can_transition_false_for_invalid_trigger(sm: SessionStateMachine):
    assert sm.can_transition("wait_for_tool") is False


def test_can_transition_false_for_unknown_trigger(sm: SessionStateMachine):
    assert sm.can_transition("nonexistent_trigger") is False
