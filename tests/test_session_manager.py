from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from agent.core.bus import MessageBus
from agent.core.message import Message
from agent.core.message.part import TextPart, TimeInfo, UserTextInput
from agent.core.session import SessionManager
from agent.core.storage import MemoryStorage
from agent.types import Event, EventType, SessionState, TokenUsage


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


async def test_create_session_auto_generates_id(mgr: SessionManager):
    ctx = await mgr.create_session(model_profile_id=MODEL_PROFILE)
    assert ctx.session_id != ""
    assert len(ctx.session_id) == 36


async def test_create_session_storage_exists_raises(mgr: SessionManager):
    _ = await mgr.create_session(session_id="dupe-storage", model_profile_id=MODEL_PROFILE)
    cache = cast("dict[str, object]", getattr(mgr, "_cache"))
    del cache["dupe-storage"]
    with pytest.raises(ValueError, match="already exists"):
        _ = await mgr.create_session(session_id="dupe-storage", model_profile_id=MODEL_PROFILE)


async def test_create_session_rollback_on_failure(bus: MessageBus, storage: MemoryStorage):
    mgr = SessionManager(bus=bus, storage=storage)
    with patch.object(
        storage, "save_session", new_callable=AsyncMock, side_effect=RuntimeError("fail")
    ):
        with pytest.raises(RuntimeError, match="fail"):
            _ = await mgr.create_session(session_id="rollback", model_profile_id=MODEL_PROFILE)
    cache = cast("dict[str, object]", getattr(mgr, "_cache"))
    state_machines = cast("dict[str, object]", getattr(mgr, "_state_machines"))
    assert "rollback" not in cache
    assert "rollback" not in state_machines


async def test_get_or_create_without_model_profile_raises(mgr: SessionManager):
    with pytest.raises(ValueError, match="model_profile_id is required"):
        _ = await mgr.get_or_create_session("s", model_profile_id="")


async def test_get_or_create_rollback_on_failure(bus: MessageBus, storage: MemoryStorage):
    mgr = SessionManager(bus=bus, storage=storage)
    with patch.object(
        storage, "save_session", new_callable=AsyncMock, side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError, match="boom"):
            _ = await mgr.get_or_create_session("fail-sess", model_profile_id=MODEL_PROFILE)
    cache = cast("dict[str, object]", getattr(mgr, "_cache"))
    state_machines = cast("dict[str, object]", getattr(mgr, "_state_machines"))
    assert "fail-sess" not in cache
    assert "fail-sess" not in state_machines


async def test_refresh_model_profile_updates_storage(bus: MessageBus, storage: MemoryStorage):
    mgr = SessionManager(bus=bus, storage=storage)
    _ = await mgr.create_session(session_id="refresh-sess", model_profile_id="openai:gpt-4o-mini")
    cache = cast("dict[str, object]", getattr(mgr, "_cache"))
    del cache["refresh-sess"]
    ctx = await mgr.get_or_create_session("refresh-sess", model_profile_id="openai:gpt-4o")
    assert ctx.model_profile_id == "openai:gpt-4o"


async def test_delete_session_cancels_listener_task(mgr: SessionManager):
    received: list[Event] = []

    async def slow_listener(event: Event) -> None:
        await asyncio.sleep(999)
        received.append(event)

    mgr.add_session_listener("cancel-sess", slow_listener)
    _ = await mgr.create_session(session_id="cancel-sess", model_profile_id=MODEL_PROFILE)
    event = Event(type=EventType.MESSAGE_DONE, session_id="cancel-sess")
    await mgr.emit_to_session_listeners("cancel-sess", event)
    await asyncio.sleep(0.01)
    await mgr.delete_session("cancel-sess")


async def test_start_assistant_message(mgr: SessionManager):
    _ = await mgr.create_session(session_id="asst-sess", model_profile_id=MODEL_PROFILE)
    ctx = await mgr.get_session("asst-sess")
    user_msg = await mgr.add_user_message("asst-sess", [UserTextInput(text="hello")])
    msg = await mgr.start_assistant_message(
        "asst-sess",
        parent_id=user_msg.info.id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    assert isinstance(msg, Message)
    assert ctx.current_message is msg


async def test_finish_assistant_message_with_current(mgr: SessionManager):
    _ = await mgr.create_session(session_id="finish-sess", model_profile_id=MODEL_PROFILE)
    user_msg = await mgr.add_user_message("finish-sess", [UserTextInput(text="hi")])
    _ = await mgr.start_assistant_message(
        "finish-sess",
        parent_id=user_msg.info.id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    await mgr.finish_assistant_message(
        "finish-sess",
        tokens=TokenUsage(input=10, output=5),
        finish="stop",
    )
    ctx = await mgr.get_session("finish-sess")
    assert ctx.current_message is None


async def test_save_session_metadata(mgr: SessionManager, storage: MemoryStorage):
    _ = await mgr.create_session(session_id="meta-sess", model_profile_id=MODEL_PROFILE)
    await mgr.save_session_metadata("meta-sess")
    data = await storage.load_session("meta-sess")
    assert data is not None


async def test_add_part_to_current(mgr: SessionManager):
    _ = await mgr.create_session(session_id="part-sess", model_profile_id=MODEL_PROFILE)
    user_msg = await mgr.add_user_message("part-sess", [UserTextInput(text="q")])
    _ = await mgr.start_assistant_message(
        "part-sess",
        parent_id=user_msg.info.id,
        provider_id="openai",
        model_id="gpt-4o-mini",
    )
    part = TextPart(
        session_id="part-sess",
        message_id=user_msg.info.id,
        text="answer",
        time=TimeInfo(start=0),
    )
    await mgr.add_part_to_current("part-sess", part)


async def test_update_state_no_state_machine(mgr: SessionManager):
    _ = await mgr.create_session(session_id="no-sm-sess", model_profile_id=MODEL_PROFILE)
    state_machines = cast("dict[str, object]", getattr(mgr, "_state_machines"))
    del state_machines["no-sm-sess"]
    await mgr.update_state("no-sm-sess", SessionState.PROCESSING)
    ctx = await mgr.get_session("no-sm-sess")
    assert ctx.state == SessionState.IDLE


async def test_update_state_invalid_transition(mgr: SessionManager):
    _ = await mgr.create_session(session_id="bad-trans", model_profile_id=MODEL_PROFILE)
    await mgr.update_state("bad-trans", SessionState.WAITING_TOOL)
    ctx = await mgr.get_session("bad-trans")
    assert ctx.state == SessionState.IDLE


async def test_remove_session_listener_not_found_does_not_raise(mgr: SessionManager):
    async def listener(_event: Event) -> None:
        pass

    mgr.remove_session_listener("no-sess", listener)


async def test_emit_to_session_listeners_exception_swallowed(mgr: SessionManager):
    async def bad_listener(_event: Event) -> None:
        raise RuntimeError("listener error")

    mgr.add_session_listener("err-sess", bad_listener)
    event = Event(type=EventType.MESSAGE_DONE, session_id="err-sess")
    await mgr.emit_to_session_listeners("err-sess", event)
    await asyncio.sleep(0.05)


async def test_switch_session_agent(mgr: SessionManager):
    _ = await mgr.create_session(
        session_id="switch-sess", model_profile_id=MODEL_PROFILE, agent_name="bot-a"
    )
    await mgr.switch_session_agent("switch-sess", "bot-b")
    ctx = await mgr.get_session("switch-sess")
    assert ctx.agent_name == "bot-b"
