"""Tests for system message creation via Message.create_system()."""

from agent.core.message import Message, UserMessageInfo, TextPart


def test_create_system_message_has_correct_role():
    msg = Message.create_system("You are a helpful assistant.")
    assert msg.role == "system"
    assert len(msg.parts) == 1
    assert msg.parts[0].text == "You are a helpful assistant."


def test_create_system_prepended_to_messages():
    user_msg = Message(
        info=UserMessageInfo(id="u1", session_id="s1", agent="general"),
        parts=[TextPart.create_fast(session_id="s1", message_id="u1", text="hi")],
    )
    system_msg = Message.create_system("system prompt")
    messages = [system_msg, user_msg]
    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_no_system_when_prompt_is_none():
    """When system_prompt is None/empty, no system message should be prepended."""
    user_msg = Message(
        info=UserMessageInfo(id="u1", session_id="s1", agent="general"),
        parts=[TextPart.create_fast(session_id="s1", message_id="u1", text="hi")],
    )
    system_prompt = None
    messages = (
        [Message.create_system(system_prompt)] + [user_msg]
        if system_prompt
        else [user_msg]
    )
    assert len(messages) == 1
    assert messages[0].role == "user"
