from __future__ import annotations

import pytest

from agent.core.bus import MessageBus
from agent.core.session import SessionManager
from agent.core.storage.memory import MemoryStorage
from agent.session import Session
from agent.types import Event, EventType, SessionState


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus(max_concurrent=4)


@pytest.fixture
def manager(bus: MessageBus, storage: MemoryStorage) -> SessionManager:
    return SessionManager(bus=bus, storage=storage)


async def _make_session(manager: SessionManager) -> Session:
    context = await manager.get_or_create_session(
        session_id="sess_1",
        model_profile_id="openai:gpt-4",
        agent_name="agent",
    )
    return Session(context, manager)


async def test_session_id_set(manager: SessionManager):
    session = await _make_session(manager)
    assert session.session_id == "sess_1"


async def test_session_state_idle(manager: SessionManager):
    session = await _make_session(manager)
    assert session.state == SessionState.IDLE


async def test_session_agent_name(manager: SessionManager):
    session = await _make_session(manager)
    assert session.agent_name == "agent"


async def test_session_model_profile_id(manager: SessionManager):
    session = await _make_session(manager)
    assert session.model_profile_id == "openai:gpt-4"


async def test_session_messages_empty(manager: SessionManager):
    session = await _make_session(manager)
    assert session.messages == []


async def test_add_and_remove_listener(manager: SessionManager):
    session = await _make_session(manager)
    received: list[Event] = []

    async def listener(event: Event) -> None:
        received.append(event)

    session.add_listener(listener)
    session.remove_listener(listener)


async def test_listener_receives_events_via_emit(manager: SessionManager):
    session = await _make_session(manager)
    received: list[Event] = []

    async def listener(event: Event) -> None:
        received.append(event)

    session.add_listener(listener)

    event = Event(type=EventType.PART_CREATED, session_id="sess_1", data={})
    await manager.emit_to_session_listeners("sess_1", event)

    import asyncio

    await asyncio.sleep(0.05)

    assert len(received) == 1
    session.remove_listener(listener)
