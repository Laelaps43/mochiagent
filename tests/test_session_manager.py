from __future__ import annotations

import asyncio

import pytest

from agent.core.bus import MessageBus
from agent.core.session import SessionManager
from agent.core.storage import MemoryStorage
from agent.types import Event, EventType, SessionState


MODEL_PROFILE = "openai:gpt-4o-mini"


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus(max_concurrent=4)


@pytest.fixture
def mgr(bus: MessageBus, storage: MemoryStorage) -> SessionManager:
    return SessionManager(bus=bus, storage=storage)


async def test_create_session_returns_context(mgr: SessionManager):
    ctx = await mgr.create_session(session_id="sess-1", model_profile_id=MODEL_PROFILE)
    assert ctx.session_id == "sess-1"
    assert ctx.model_profile_id == MODEL_PROFILE


async def test_create_session_persists_to_storage(mgr: SessionManager, storage: MemoryStorage):
    _ = await mgr.create_session(session_id="sess-persist", model_profile_id=MODEL_PROFILE)
    assert await storage.session_exists("sess-persist")


async def test_create_session_duplicate_raises(mgr: SessionManager):
    _ = await mgr.create_session(session_id="dup", model_profile_id=MODEL_PROFILE)
    with pytest.raises(ValueError, match="already exists"):
        _ = await mgr.create_session(session_id="dup", model_profile_id=MODEL_PROFILE)


async def test_create_session_without_model_profile_raises(mgr: SessionManager):
    with pytest.raises(ValueError, match="model_profile_id is required"):
        _ = await mgr.create_session(session_id="no-profile")


async def test_get_session_cache_hit(mgr: SessionManager):
    _ = await mgr.create_session(session_id="cached", model_profile_id=MODEL_PROFILE)
    ctx = await mgr.get_session("cached")
    assert ctx.session_id == "cached"


async def test_get_session_cold_load(bus: MessageBus, storage: MemoryStorage):
    warm_mgr = SessionManager(bus=bus, storage=storage)
    _ = await warm_mgr.create_session(session_id="cold", model_profile_id=MODEL_PROFILE)

    cold_mgr = SessionManager(bus=bus, storage=storage)
    ctx = await cold_mgr.get_session("cold")
    assert ctx.session_id == "cold"


async def test_get_session_missing_raises(mgr: SessionManager):
    with pytest.raises(ValueError, match="not found"):
        _ = await mgr.get_session("ghost")


async def test_get_or_create_creates_when_absent(mgr: SessionManager):
    ctx = await mgr.get_or_create_session(
        "new-sess", model_profile_id=MODEL_PROFILE, agent_name="bot"
    )
    assert ctx.session_id == "new-sess"


async def test_get_or_create_returns_existing(mgr: SessionManager):
    _ = await mgr.create_session(session_id="existing", model_profile_id=MODEL_PROFILE)
    ctx = await mgr.get_or_create_session("existing", model_profile_id=MODEL_PROFILE)
    assert ctx.session_id == "existing"


async def test_get_or_create_loads_from_storage(bus: MessageBus, storage: MemoryStorage):
    warm_mgr = SessionManager(bus=bus, storage=storage)
    _ = await warm_mgr.create_session(session_id="stored", model_profile_id=MODEL_PROFILE)

    cold_mgr = SessionManager(bus=bus, storage=storage)
    ctx = await cold_mgr.get_or_create_session("stored", model_profile_id=MODEL_PROFILE)
    assert ctx.session_id == "stored"


async def test_delete_session_removes_from_cache_and_storage(
    mgr: SessionManager, storage: MemoryStorage
):
    _ = await mgr.create_session(session_id="del-me", model_profile_id=MODEL_PROFILE)
    await mgr.delete_session("del-me")
    assert not await storage.session_exists("del-me")
    with pytest.raises(ValueError):
        _ = await mgr.get_session("del-me")


async def test_delete_nonexistent_session_does_not_raise(mgr: SessionManager):
    await mgr.delete_session("no-such-session")


async def test_concurrent_get_or_create_no_duplicates(mgr: SessionManager):
    results = await asyncio.gather(
        *[mgr.get_or_create_session("race", model_profile_id=MODEL_PROFILE) for _ in range(10)]
    )
    ids = {ctx.session_id for ctx in results}
    assert ids == {"race"}


async def test_add_and_remove_listener(mgr: SessionManager):
    received: list[Event] = []

    async def listener(event: Event) -> None:
        received.append(event)

    mgr.add_session_listener("ls-sess", listener)
    event = Event(type=EventType.MESSAGE_DONE, session_id="ls-sess")
    await mgr.emit_to_session_listeners("ls-sess", event)
    await asyncio.sleep(0.05)
    assert len(received) == 1

    mgr.remove_session_listener("ls-sess", listener)
    await mgr.emit_to_session_listeners("ls-sess", event)
    await asyncio.sleep(0.05)
    assert len(received) == 1


async def test_emit_to_session_listeners_delivers_event(mgr: SessionManager):
    received: list[Event] = []

    async def listener(event: Event) -> None:
        received.append(event)

    mgr.add_session_listener("emit-sess", listener)
    event = Event(type=EventType.MESSAGE_DONE, session_id="emit-sess")
    await mgr.emit_to_session_listeners("emit-sess", event)

    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].session_id == "emit-sess"


async def test_emit_with_no_listeners_does_not_raise(mgr: SessionManager):
    event = Event(type=EventType.MESSAGE_DONE, session_id="orphan")
    await mgr.emit_to_session_listeners("orphan", event)


async def test_update_state_transitions_successfully(mgr: SessionManager):
    _ = await mgr.create_session(session_id="state-sess", model_profile_id=MODEL_PROFILE)
    await mgr.update_state("state-sess", SessionState.PROCESSING)
    ctx = await mgr.get_session("state-sess")
    assert ctx.state == SessionState.PROCESSING
