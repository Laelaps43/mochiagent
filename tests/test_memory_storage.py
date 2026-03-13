from __future__ import annotations

import pytest

from agent.core.storage.memory import MemoryStorage
from agent.core.message import Message, UserMessageInfo, TextPart, TimeInfo
from agent.types import SessionMetadataData, ContextBudget


def _make_session_data(session_id: str = "sess_1") -> SessionMetadataData:
    return SessionMetadataData(
        session_id=session_id,
        state="idle",
        model_profile_id="openai:gpt-4",
        agent_name="agent",
        context_budget=ContextBudget(),
        created_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-01-01T00:00:00+00:00",
    )


def _make_message(session_id: str = "sess_1", msg_id: str = "msg_1") -> Message:
    return Message(
        info=UserMessageInfo(
            id=msg_id,
            session_id=session_id,
            created_at=0,
            agent="agent",
        ),
        parts=[
            TextPart(
                session_id=session_id,
                message_id=msg_id,
                text="hello",
                time=TimeInfo(start=0),
            )
        ],
    )


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


async def test_save_and_load_session(storage: MemoryStorage):
    data = _make_session_data()
    await storage.save_session("sess_1", data)
    loaded = await storage.load_session("sess_1")
    assert loaded is not None
    assert loaded.session_id == "sess_1"


async def test_load_missing_session_returns_none(storage: MemoryStorage):
    result = await storage.load_session("nonexistent")
    assert result is None


async def test_session_exists(storage: MemoryStorage):
    assert await storage.session_exists("sess_1") is False
    await storage.save_session("sess_1", _make_session_data())
    assert await storage.session_exists("sess_1") is True


async def test_list_sessions(storage: MemoryStorage):
    await storage.save_session("a", _make_session_data("a"))
    await storage.save_session("b", _make_session_data("b"))
    sessions = await storage.list_sessions()
    assert set(sessions) == {"a", "b"}


async def test_delete_session(storage: MemoryStorage):
    await storage.save_session("sess_1", _make_session_data())
    await storage.delete_session("sess_1")
    assert await storage.session_exists("sess_1") is False


async def test_delete_session_also_removes_messages(storage: MemoryStorage):
    await storage.save_session("sess_1", _make_session_data())
    msg = _make_message()
    await storage.save_message("sess_1", msg)
    await storage.delete_session("sess_1")
    msgs = await storage.load_messages("sess_1")
    assert msgs == []


async def test_save_and_load_messages(storage: MemoryStorage):
    msg1 = _make_message(msg_id="msg_1")
    msg2 = _make_message(msg_id="msg_2")
    await storage.save_message("sess_1", msg1)
    await storage.save_message("sess_1", msg2)
    msgs = await storage.load_messages("sess_1")
    assert len(msgs) == 2


async def test_load_messages_from_message_id(storage: MemoryStorage):
    for i in range(4):
        await storage.save_message("sess_1", _make_message(msg_id=f"msg_{i}"))
    msgs = await storage.load_messages("sess_1", from_message_id="msg_2")
    ids = [m.message_id for m in msgs]
    assert "msg_2" in ids
    assert "msg_0" not in ids
    assert "msg_1" not in ids


async def test_load_messages_from_id_not_found_returns_all(storage: MemoryStorage):
    for i in range(3):
        await storage.save_message("sess_1", _make_message(msg_id=f"msg_{i}"))
    msgs = await storage.load_messages("sess_1", from_message_id="msg_99")
    assert len(msgs) == 3


async def test_delete_messages(storage: MemoryStorage):
    await storage.save_message("sess_1", _make_message())
    await storage.delete_messages("sess_1")
    msgs = await storage.load_messages("sess_1")
    assert msgs == []


async def test_load_messages_empty_session(storage: MemoryStorage):
    msgs = await storage.load_messages("no_such_session")
    assert msgs == []


async def test_save_artifact(storage: MemoryStorage):
    meta = await storage.save_artifact(
        session_id="sess_1",
        kind="tool_result",
        content="big content here",
        metadata={"tool_name": "echo"},
    )
    assert meta.artifact_ref.startswith("artifact://sess_1/")
    assert meta.session_id == "sess_1"
    assert meta.kind == "tool_result"
    assert meta.size == len("big content here")


async def test_read_artifact(storage: MemoryStorage):
    meta = await storage.save_artifact("sess_1", "tool_result", "hello world")
    result = await storage.read_artifact(meta.artifact_ref)
    assert result.success is True
    assert result.content == "hello world"
    assert result.eof is True


async def test_read_artifact_with_offset(storage: MemoryStorage):
    meta = await storage.save_artifact("sess_1", "tool_result", "0123456789")
    result = await storage.read_artifact(meta.artifact_ref, offset=5, limit=3)
    assert result.content == "567"
    assert result.offset == 5


async def test_read_artifact_not_found(storage: MemoryStorage):
    result = await storage.read_artifact("artifact://sess_1/nonexistent_id")
    assert result.success is False
    assert "not found" in result.error


async def test_delete_artifacts(storage: MemoryStorage):
    meta = await storage.save_artifact("sess_1", "tool_result", "content")
    await storage.delete_artifacts("sess_1")
    result = await storage.read_artifact(meta.artifact_ref)
    assert result.success is False


async def test_parse_artifact_ref_invalid():
    with pytest.raises(ValueError, match="Invalid artifact_ref"):
        _ = MemoryStorage.parse_artifact_ref("bad://ref")


async def test_parse_artifact_ref_missing_parts():
    with pytest.raises(ValueError, match="Invalid artifact_ref"):
        _ = MemoryStorage.parse_artifact_ref("artifact://only_one_part")


async def test_session_count_warning_logged(storage: MemoryStorage):
    threshold = MemoryStorage.SESSION_COUNT_WARNING_THRESHOLD
    for i in range(threshold):
        await storage.save_session(f"s_{i}", _make_session_data(f"s_{i}"))
