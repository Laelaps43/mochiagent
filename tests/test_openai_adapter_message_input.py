"""Tests for LLMProvider.prepare_messages() — the unified Message→dict conversion."""

from agent.core.llm.base import LLMProvider
from agent.core.message import Message, UserMessageInfo, AssistantMessageInfo, TextPart


class _DummyProvider(LLMProvider):
    """Minimal concrete provider for testing base class methods."""

    async def stream_chat(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    async def complete(self, messages, tools=None, **kwargs):
        raise NotImplementedError


def _provider() -> _DummyProvider:
    from agent.types import LLMConfig

    return _DummyProvider(LLMConfig(adapter="test", provider="test", model="test"))


def test_prepare_messages_converts_internal_message() -> None:
    msg = Message(
        info=UserMessageInfo(id="u1", session_id="s1", agent="general"),
        parts=[TextPart.create_fast(session_id="s1", message_id="u1", text="hello")],
    )
    result = _provider().prepare_messages([msg])
    assert result == [{"role": "user", "content": "hello"}]


def test_prepare_messages_handles_system_message() -> None:
    msg = Message.create_system("You are helpful.")
    result = _provider().prepare_messages([msg])
    assert result == [{"role": "system", "content": "You are helpful."}]


def test_prepare_messages_skips_empty_assistant_message() -> None:
    """An assistant message with no parts produces no dict entries."""
    msg = Message(
        info=AssistantMessageInfo(
            id="a1", session_id="s1", parent_id="u1",
            agent="general", model_id="m", provider_id="p",
        ),
        parts=[],
    )
    result = _provider().prepare_messages([msg])
    assert result == []
