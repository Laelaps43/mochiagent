from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.core.bus import MessageBus
from agent.core.loop import AgentEventLoop
from agent.core.session.context import SessionContext
from agent.types import LLMConfig, SessionState


class _SessionManagerStub:
    def __init__(self, context: SessionContext):
        self._context = context
        self.finish_calls: list[dict] = []
        self.states: list[SessionState] = []

    async def get_session(self, session_id: str) -> SessionContext:
        return self._context

    async def finish_assistant_message(
        self,
        session_id: str,
        cost: float = 0.0,
        tokens: dict | None = None,
        finish: str = "stop",
    ) -> None:
        self.finish_calls.append(
            {
                "session_id": session_id,
                "cost": cost,
                "tokens": tokens or {},
                "finish": finish,
            }
        )
        self._context.finish_current_message(cost=cost, tokens=tokens, finish=finish)

    async def emit_to_session_listeners(self, session_id: str, event) -> None:
        return None

    async def update_state(self, session_id: str, new_state: SessionState) -> None:
        self.states.append(new_state)
        self._context.update_state(new_state)


class _LLMNoToolStub:
    async def stream_chat(self, messages, tools=None):
        yield {"content": "hello"}
        yield {"finish_reason": "stop"}


class _AdapterRegistryStub:
    def get(self, config):
        return _LLMNoToolStub()


@pytest.mark.asyncio
async def test_call_llm_turn_uses_finish_assistant_message_for_persistence():
    context = SessionContext(
        session_id="sess_no_tool",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[{"type": "text", "text": "hi"}])
    session_manager = _SessionManagerStub(context)

    framework = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
        ),
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
        )
    )

    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=_AdapterRegistryStub(),
        framework=framework,
    )

    result = await loop._call_llm_turn("sess_no_tool")

    assert result["finish_reason"] == "stop"
    assert len(session_manager.finish_calls) == 1
    assert session_manager.finish_calls[0]["session_id"] == "sess_no_tool"


@pytest.mark.asyncio
async def test_conversation_loop_tool_branch_uses_finish_assistant_message():
    context = SessionContext(
        session_id="sess_tool",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[{"type": "text", "text": "hi"}])
    session_manager = _SessionManagerStub(context)

    framework = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
        ),
        get_agent=lambda _: None,
    )
    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=_AdapterRegistryStub(),
        framework=framework,
    )

    call_count = {"n": 0}

    async def _fake_call_llm_turn(session_id: str):
        if call_count["n"] == 0:
            call_count["n"] += 1
            context.build_assistant_message(
                parent_id=context.messages[-1].message_id,
                provider_id="openai",
                model_id="mock",
            )
            return {
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}
                ],
                "finish_reason": "tool_calls",
                "cost": 0.0,
                "tokens": {},
                "message_id": context.current_message.message_id if context.current_message else "",
            }

        return {
            "content": "done",
            "tool_calls": [],
            "finish_reason": "stop",
            "cost": 0.0,
            "tokens": {},
            "message_id": "msg_done",
        }

    async def _fake_execute_tools(session_id: str, tool_calls: list):
        return []

    loop._call_llm_turn = _fake_call_llm_turn  # type: ignore[method-assign]
    loop._execute_tools = _fake_execute_tools  # type: ignore[method-assign]

    await loop._conversation_loop("sess_tool")

    assert len(session_manager.finish_calls) >= 1
    assert session_manager.finish_calls[0]["finish"] == "tool_calls"
