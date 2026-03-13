from __future__ import annotations

from agent.core.message.message import Message
from agent.core.message.info import (
    UserMessageInfo,
    AssistantMessageInfo,
    SystemMessageInfo,
    CompactionMessageInfo,
)
from agent.core.message.part import TextPart


def _user_msg() -> Message:
    return Message(
        info=UserMessageInfo(id="msg_1", session_id="sess_1"),
    )


def test_message_id_property() -> None:
    msg = _user_msg()
    assert msg.message_id == "msg_1"


def test_session_id_property() -> None:
    msg = _user_msg()
    assert msg.session_id == "sess_1"


def test_role_user() -> None:
    msg = _user_msg()
    assert msg.role == "user"


def test_role_assistant() -> None:
    msg = Message(
        info=AssistantMessageInfo(
            id="msg_2",
            session_id="sess_1",
            parent_id="msg_1",
            model_id="gpt-4o",
            provider_id="openai",
        )
    )
    assert msg.role == "assistant"


def test_role_system() -> None:
    msg = Message(info=SystemMessageInfo())
    assert msg.role == "system"


def test_role_compaction() -> None:
    msg = Message(info=CompactionMessageInfo(id="cmp_1", session_id="sess_1"))
    assert msg.role == "compaction"


def test_add_part() -> None:
    msg = _user_msg()
    part = TextPart(session_id="sess_1", message_id="msg_1", text="hello")
    msg.add_part(part)
    assert len(msg.parts) == 1
    assert msg.parts[0] is part


def test_create_system() -> None:
    msg = Message.create_system("You are a helpful assistant.")
    assert msg.role == "system"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], TextPart)
    assert msg.parts[0].text == "You are a helpful assistant."


def test_create_system_empty_session_ids() -> None:
    msg = Message.create_system("sys prompt")
    part = msg.parts[0]
    assert isinstance(part, TextPart)
    assert part.session_id == ""
    assert part.message_id == ""


def test_create_compaction_basic() -> None:
    msg = Message.create_compaction(
        session_id="sess_1",
        summary="Work in progress: step 1 done.",
    )
    assert msg.role == "compaction"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], TextPart)
    assert "Work in progress" in msg.parts[0].text
    assert "COMPACTION_SUMMARY" in msg.parts[0].text


def test_create_compaction_metadata() -> None:
    msg = Message.create_compaction(
        session_id="sess_1",
        summary="summary text",
        compacted_count=5,
        compaction_metadata={"key": "val"},
    )
    assert isinstance(msg.info, CompactionMessageInfo)
    assert msg.info.compacted_count == 5
    assert msg.info.compaction_metadata == {"key": "val"}


def test_parts_default_empty() -> None:
    msg = _user_msg()
    assert msg.parts == []
