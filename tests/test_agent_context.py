from __future__ import annotations

import pytest

from agent.context import AgentContext
from agent.core.bus import MessageBus
from agent.core.runtime import AgentStrategyManager
from agent.core.session import SessionManager
from agent.core.storage.memory import MemoryStorage
from agent.types import LLMConfig


def _make_llm_config(provider: str = "openai", model: str = "gpt-4") -> LLMConfig:
    return LLMConfig(
        adapter="openai_compatible",
        provider=provider,
        model=model,
        api_key=None,
    )


@pytest.fixture
def context() -> AgentContext:
    storage = MemoryStorage()
    bus = MessageBus(max_concurrent=4)
    manager = SessionManager(bus=bus, storage=storage)
    strategy = AgentStrategyManager()
    return AgentContext(
        session_manager=manager,
        message_bus=bus,
        strategy_manager=strategy,
        agent_name="agent1",
        llm_profiles={"openai:gpt-4": _make_llm_config()},
    )


def test_resolve_llm_config_success(context: AgentContext):
    cfg = context.resolve_llm_config_for_agent("agent1", "openai:gpt-4")
    assert cfg.model == "gpt-4"
    assert cfg.provider == "openai"


def test_resolve_llm_config_wrong_agent(context: AgentContext):
    with pytest.raises(ValueError, match="context mismatch"):
        _ = context.resolve_llm_config_for_agent("wrong_agent", "openai:gpt-4")


def test_resolve_llm_config_missing_profile(context: AgentContext):
    with pytest.raises(ValueError, match="not available"):
        _ = context.resolve_llm_config_for_agent("agent1", "openai:gpt-3")


def test_resolve_llm_config_lists_available(context: AgentContext):
    with pytest.raises(ValueError, match="openai:gpt-4"):
        _ = context.resolve_llm_config_for_agent("agent1", "openai:gpt-3")


async def test_get_session_creates_session(context: AgentContext):
    session = await context.get_session("agent1", "sess_test", "openai:gpt-4")
    assert session.session_id == "sess_test"


async def test_send_message_publishes_event(context: AgentContext):
    from agent.core.message import UserTextInput
    from agent.types import Event, EventType

    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    context.message_bus.subscribe(EventType.MESSAGE_RECEIVED, handler)

    _ = await context.get_session("agent1", "sess_send", "openai:gpt-4")

    await context.message_bus.start()
    await context.send_message("sess_send", [UserTextInput(text="hi")])
    await context.message_bus.wait_empty()
    await context.message_bus.stop()

    assert any(e.type == EventType.MESSAGE_RECEIVED for e in received)


async def test_switch_session_agent(context: AgentContext):
    _ = await context.get_session("agent1", "sess_switch", "openai:gpt-4")
    await context.switch_session_agent("sess_switch", "agent2")
    ctx = await context.session_manager.get_or_create_session(
        "sess_switch", model_profile_id="openai:gpt-4", agent_name="agent2"
    )
    assert ctx.agent_name == "agent2"
