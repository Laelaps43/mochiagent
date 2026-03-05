import pytest

from agent.core.llm.adapters.openai import OpenAIAdapter
from agent.types import Message, MessageRole


def test_normalize_messages_accepts_public_message_model() -> None:
    payload = [Message(role=MessageRole.USER, content="hello")]
    assert OpenAIAdapter._normalize_messages(payload) == [{"role": "user", "content": "hello"}]


def test_normalize_messages_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="Unsupported message type"):
        OpenAIAdapter._normalize_messages([123])  # type: ignore[list-item]


def test_normalize_messages_rejects_dict_payload() -> None:
    with pytest.raises(TypeError, match="Expected agent.types.Message"):
        OpenAIAdapter._normalize_messages([{"role": "user", "content": "hello"}])  # type: ignore[list-item]
