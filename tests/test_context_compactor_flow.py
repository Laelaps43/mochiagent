from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.core.bus import MessageBus
from agent.core.compression import CompactionPayload, CompactionStage
from agent.core.llm.errors import LLMTransportError
from agent.core.loop import AgentEventLoop
from agent.core.runtime import StrategyKind
from agent.core.message import UserTextPart
from agent.core.session.context import SessionContext
from agent.types import LLMConfig, SessionState


class _SessionManagerStub:
    def __init__(self, context: SessionContext):
        self._context = context
        self.finish_calls: list[dict] = []
        self.states: list[SessionState] = []
        self.metadata_save_calls = 0

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

    async def save_session_metadata(self, session_id: str) -> None:
        self.metadata_save_calls += 1


class _LLMNoToolStub:
    async def stream_chat(self, messages, tools=None):
        yield {"content": "ok"}
        yield {"finish_reason": "stop"}


class _OverflowOnceLLMStub:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            raise LLMTransportError(
                code="LLM_ERROR",
                message="maximum context length exceeded",
                status_code=400,
            )
        yield {"content": "ok"}
        yield {"finish_reason": "stop"}


@pytest.mark.asyncio
async def test_context_compactor_runs_before_llm_call():
    context = SessionContext(session_id="sess_pre", model_profile_id="safe", agent_name="agent_stub")
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    stages: list[str] = []

    async def _run_strategy(kind, **kwargs):
        assert kind == StrategyKind.CONTEXT_COMPACTION
        stage = kwargs["stage"]
        stages.append(stage.value)
        return CompactionPayload.invalid(stage=stage.value, reason="skip", name="custom")

    framework = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
            context_window_tokens=1024,
        ),
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
        ),
        strategy_manager=SimpleNamespace(run=_run_strategy),
    )
    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=SimpleNamespace(get=lambda _cfg: _LLMNoToolStub()),
        framework=framework,
    )

    result = await loop._llm_turn_handler.run("sess_pre")

    assert stages == [CompactionStage.PRE_CALL.value]
    assert result.context_compaction_events[0].stage == "pre_call"


@pytest.mark.asyncio
async def test_context_overflow_triggers_compaction_and_retry():
    context = SessionContext(
        session_id="sess_overflow",
        model_profile_id="safe",
        agent_name="agent_stub",
    )
    context.build_user_message(parts=[UserTextPart(text="hi")])
    session_manager = _SessionManagerStub(context)

    stages: list[str] = []

    async def _run_strategy(kind, **kwargs):
        assert kind == StrategyKind.CONTEXT_COMPACTION
        stage = kwargs["stage"]
        stage_value = stage.value
        stages.append(stage_value)
        applied = stage == CompactionStage.OVERFLOW_ERROR
        return CompactionPayload(
            applied=applied,
            reason="compressed" if applied else "skip",
            metadata={},
            stats={},
            artifacts=[],
            name="custom",
            stage=stage_value,
        )

    llm = _OverflowOnceLLMStub()
    framework = SimpleNamespace(
        resolve_llm_config_for_agent=lambda _agent, _profile: LLMConfig(
            adapter="openai_compatible",
            provider="openai",
            model="mock",
            context_window_tokens=1024,
        ),
        get_agent=lambda _: SimpleNamespace(
            tool_registry=SimpleNamespace(get_definitions=lambda: []),
            get_system_prompt=lambda _ctx: None,
        ),
        strategy_manager=SimpleNamespace(run=_run_strategy),
    )
    loop = AgentEventLoop(
        bus=MessageBus(),
        session_manager=session_manager,
        adapter_registry=SimpleNamespace(get=lambda _cfg: llm),
        framework=framework,
    )

    result = await loop._llm_turn_handler.run("sess_overflow")

    assert llm.calls == 2
    assert stages == [CompactionStage.PRE_CALL.value, CompactionStage.OVERFLOW_ERROR.value]
    assert result.context_compaction.stage == "overflow_error"
    assert result.context_compaction.applied is True
    assert session_manager.metadata_save_calls >= 2
