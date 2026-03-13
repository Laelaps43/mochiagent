from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import cast, override
from unittest.mock import AsyncMock

from agent.core.compression import CompactionStage, CompactorRunOptions, DefaultContextCompactor
from agent.core.llm.base import LLMProvider
from agent.core.message import Message as InternalMessage, TextPart, UserTextInput
from agent.core.runtime.strategy_kind import StrategyKind
from agent.core.runtime.strategy_manager import AgentStrategyManager, StrategySlot
from agent.core.session.context import SessionContext
from agent.core.storage.memory import MemoryStorage
from agent.core.storage.provider import StorageProvider
from agent.core.tools import ToolResultPostProcessor
from agent.core.tools.result_postprocessor import (
    ToolResultPostProcessConfig,
    ToolResultPostProcessorStrategy,
)
from agent.types import (
    ContextBudget,
    Event,
    EventType,
    LLMConfig,
    LLMStreamChunk,
    ToolDefinition,
    ToolResult,
)


class _MockLLMProvider(LLMProvider):
    complete_mock: Callable[..., Awaitable[object]]
    complete_await_count: int
    _response: LLMStreamChunk | Exception

    def __init__(self, config: LLMConfig, response: LLMStreamChunk | Exception) -> None:
        super().__init__(config)
        self._response = response
        self.complete_mock = cast(Callable[..., Awaitable[object]], AsyncMock())
        self.complete_await_count = 0

    @override
    async def stream_chat(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        _ = messages
        _ = tools
        _ = kwargs
        chunks: list[LLMStreamChunk] = []
        for chunk in chunks:
            yield chunk

    @override
    async def complete(
        self,
        messages: list[InternalMessage],
        tools: list[ToolDefinition] | None = None,
        **kwargs: object,
    ) -> LLMStreamChunk:
        self.complete_await_count += 1
        _ = await self.complete_mock(messages=messages, tools=tools, **kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _ExplodingPostProcessor(ToolResultPostProcessorStrategy):
    @override
    async def process(
        self,
        *,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, object],
        storage: StorageProvider,
    ) -> ToolResult:
        _ = session_id
        _ = tool_result
        _ = tool_arguments
        _ = storage
        raise RuntimeError("boom")


def _make_llm_config(context_window_tokens: int | None = 100) -> LLMConfig:
    return LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model="gpt-4o-mini",
        context_window_tokens=context_window_tokens,
    )


def _make_context(*messages: str) -> SessionContext:
    context = SessionContext(
        session_id="sess_1",
        model_profile_id="openai:gpt-4o-mini",
        agent_name="agent_a",
    )
    for text in messages:
        _ = context.build_user_message([UserTextInput(text=text)])
    return context


def _make_tool_result(
    result: object, *, success: bool = True, error: str | None = None
) -> ToolResult:
    return ToolResult(
        tool_call_id="call_1",
        tool_name="exec",
        result=result,
        success=success,
        error=error,
    )


def test_strategy_slot_gets_default_and_override() -> None:
    slot = StrategySlot[str]("default")

    slot.set(" agent_a ", "override")

    assert slot.get() == "default"
    assert slot.get("agent_a") == "override"
    assert slot.get("missing") == "default"


async def test_strategy_manager_run_postprocess_uses_default_processor(tmp_path: Path) -> None:
    manager = AgentStrategyManager()
    storage = MemoryStorage(artifact_root=tmp_path / "artifacts")
    tool_result = _make_tool_result("short output")

    processed = await manager.run_postprocess(
        session_id="sess_1",
        tool_result=tool_result,
        tool_arguments={"command": "echo short"},
        storage=storage,
    )

    assert processed.summary == "short output"
    assert processed.truncated is False
    assert processed.artifact_ref is None


async def test_strategy_manager_run_postprocess_uses_agent_override(tmp_path: Path) -> None:
    manager = AgentStrategyManager()
    processor = ToolResultPostProcessor(
        config=ToolResultPostProcessConfig(
            summary_max_chars=120,
            preview_head_chars=12,
            preview_tail_chars=10,
        )
    )
    manager.set(StrategyKind.TOOL_RESULT_POSTPROCESS, "agent_a", processor)
    storage = MemoryStorage(artifact_root=tmp_path / "artifacts")
    tool_result = _make_tool_result("x" * 200)

    processed = await manager.run_postprocess(
        agent_name="agent_a",
        session_id="sess_1",
        tool_result=tool_result,
        tool_arguments={"command": "python"},
        storage=storage,
    )

    assert processed.truncated is True
    assert processed.artifact_ref is not None
    assert processed.artifact_path is not None

    artifact_ref = processed.artifact_ref
    assert artifact_ref is not None

    artifact = await storage.read_artifact(artifact_ref)

    assert artifact.success is True
    assert artifact.content == "x" * 200


async def test_strategy_manager_run_postprocess_returns_original_on_exception(
    tmp_path: Path,
) -> None:
    manager = AgentStrategyManager()
    manager.set(StrategyKind.TOOL_RESULT_POSTPROCESS, "agent_a", _ExplodingPostProcessor())
    storage = MemoryStorage(artifact_root=tmp_path / "artifacts")
    tool_result = _make_tool_result({"ok": True})

    processed = await manager.run_postprocess(
        agent_name="agent_a",
        session_id="sess_1",
        tool_result=tool_result,
        tool_arguments={},
        storage=storage,
    )

    assert processed == tool_result


async def test_strategy_manager_run_compaction_returns_noop_below_threshold() -> None:
    manager = AgentStrategyManager()
    session_context = _make_context("short")
    provider = _MockLLMProvider(
        _make_llm_config(context_window_tokens=500),
        AssertionError("complete should not be called"),
    )

    payload = await manager.run_compaction(
        session_context=session_context,
        budget=ContextBudget(),
        llm_config=_make_llm_config(context_window_tokens=500),
        llm_provider=provider,
        stage=CompactionStage.PRE_CALL,
    )

    assert payload.applied is False
    assert payload.reason == "below_threshold"
    assert session_context.last_compaction_message_id is None
    assert len(session_context.messages) == 1


async def test_strategy_manager_run_compaction_applies_and_emits_events() -> None:
    manager = AgentStrategyManager()
    manager.set(
        StrategyKind.CONTEXT_COMPACTION,
        "agent_a",
        strategy=DefaultContextCompactor(),
        compaction_options=CompactorRunOptions(
            keep_user_tokens_budget=0,
            chars_per_token=1.0,
        ),
    )
    session_context = _make_context("first message", "second message")
    provider = _MockLLMProvider(
        _make_llm_config(context_window_tokens=100),
        LLMStreamChunk(content="handoff summary"),
    )
    events: list[Event] = []

    async def emit_event(event: Event) -> None:
        events.append(event)

    payload = await manager.run_compaction(
        session_context=session_context,
        budget=ContextBudget(),
        llm_config=_make_llm_config(context_window_tokens=100),
        llm_provider=provider,
        agent_name="agent_a",
        stage=CompactionStage.OVERFLOW_ERROR,
        emit_event=emit_event,
    )

    assert payload.applied is True
    assert payload.reason == "overflow_error"
    assert session_context.last_compaction_message_id is not None
    assert len(session_context.messages) == 1
    assert session_context.messages[0].role == "compaction"
    summary_part = session_context.messages[0].parts[0]
    assert isinstance(summary_part, TextPart)
    assert summary_part.text == "COMPACTION_SUMMARY\nhandoff summary"
    assert [event.type for event in events] == [
        EventType.CONTEXT_COMPACTING,
        EventType.CONTEXT_COMPACTED,
    ]
    assert provider.complete_await_count == 1
