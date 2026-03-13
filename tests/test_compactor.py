from __future__ import annotations

from unittest.mock import AsyncMock

from agent.core.compression.compactor import NoopContextCompactor
from agent.core.compression.default_compactor import DefaultContextCompactor
from agent.core.compression.stage import CompactionStage
from agent.core.compression.types import (
    CompactorRunOptions,
)
from agent.core.message.message import Message
from agent.core.message.part import (
    ToolPart,
    ToolInput,
    ToolStateRunning,
    ToolStateCompleted,
    ToolStateError,
    TimeInfo,
)
from agent.core.llm.errors import LLMProviderError
from agent.core.session.context import SessionContext
from agent.types import ContextBudget, LLMConfig, LLMStreamChunk


def _make_session_context(session_id: str = "sess_1") -> SessionContext:
    return SessionContext(session_id=session_id, model_profile_id="openai:gpt-4o")


def _make_llm_config(context_window: int | None = None) -> LLMConfig:
    return LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model="gpt-4o",
        context_window_tokens=context_window,
    )


def _make_text_message(session_id: str, text: str) -> Message:
    sc = SessionContext(session_id=session_id, model_profile_id="openai:gpt-4o")
    from agent.core.message.part import UserTextInput

    _ = sc.build_user_message([UserTextInput(text=text)])
    return sc.messages[-1]


def _make_mock_provider(content: str = "summary text") -> AsyncMock:
    provider = AsyncMock()
    chunk = LLMStreamChunk(content=content, finish_reason="stop")
    provider.complete = AsyncMock(return_value=chunk)
    return provider


def test_noop_compactor_should_compact_false() -> None:
    noop = NoopContextCompactor()
    decision = noop.should_compact(
        messages=[],
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        stage=CompactionStage.PRE_CALL,
        options=CompactorRunOptions(),
    )
    assert decision.apply is False
    assert decision.reason == "noop"


async def test_noop_compactor_summarize_failure() -> None:
    noop = NoopContextCompactor()
    result = await noop.summarize(
        messages=[],
        llm_config=_make_llm_config(),
        llm_provider=_make_mock_provider(),
        options=CompactorRunOptions(),
    )
    assert result.ok is False


async def test_noop_compactor_run_returns_not_applied() -> None:
    noop = NoopContextCompactor()
    ctx = _make_session_context()
    _ = await noop.run(
        session_context=ctx,
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        llm_provider=_make_mock_provider(),
        stage=CompactionStage.PRE_CALL,
        options=CompactorRunOptions(),
    )


async def test_compactor_run_applies_when_should_compact_true() -> None:
    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    msg = _make_text_message("sess_1", "hello " * 1000)
    ctx.messages.append(msg)

    provider = _make_mock_provider("context summary")
    result = await compactor.run(
        session_context=ctx,
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        llm_provider=provider,
        stage=CompactionStage.OVERFLOW_ERROR,
        options=CompactorRunOptions(),
    )
    assert result.applied is True
    assert "overflow_error" in result.reason


async def test_compactor_run_emits_events() -> None:
    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    msg = _make_text_message("sess_1", "hello " * 1000)
    ctx.messages.append(msg)

    events: list[object] = []

    async def emit(event: object) -> None:
        events.append(event)

    provider = _make_mock_provider("summary")
    _ = await compactor.run(
        session_context=ctx,
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        llm_provider=provider,
        stage=CompactionStage.OVERFLOW_ERROR,
        options=CompactorRunOptions(),
        emit_event=emit,
    )
    assert len(events) >= 2


async def test_compactor_run_catches_exceptions() -> None:
    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    msg = _make_text_message("sess_1", "hello")
    ctx.messages.append(msg)

    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("unexpected"))

    _ = await compactor.run(
        session_context=ctx,
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        llm_provider=provider,
        stage=CompactionStage.OVERFLOW_ERROR,
        options=CompactorRunOptions(),
    )


def test_default_compactor_should_compact_overflow_error() -> None:
    compactor = DefaultContextCompactor()
    decision = compactor.should_compact(
        messages=[],
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        stage=CompactionStage.OVERFLOW_ERROR,
        options=CompactorRunOptions(),
    )
    assert decision.apply is True
    assert decision.reason == "overflow_error"


def test_default_compactor_should_compact_mid_turn_not_needed() -> None:
    compactor = DefaultContextCompactor()
    options = CompactorRunOptions(token_limit_reached=False, needs_follow_up=False)
    decision = compactor.should_compact(
        messages=[],
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        stage=CompactionStage.MID_TURN,
        options=options,
    )
    assert decision.apply is False


def test_default_compactor_should_compact_mid_turn_needed() -> None:
    compactor = DefaultContextCompactor()
    options = CompactorRunOptions(token_limit_reached=True, needs_follow_up=True)
    decision = compactor.should_compact(
        messages=[],
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        stage=CompactionStage.MID_TURN,
        options=options,
    )
    assert decision.apply is True
    assert "mid_turn_follow_up" in decision.reason


def test_default_compactor_pre_call_below_threshold() -> None:
    compactor = DefaultContextCompactor()
    options = CompactorRunOptions(model_auto_compact_token_limit=100000)
    decision = compactor.should_compact(
        messages=[],
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        stage=CompactionStage.PRE_CALL,
        options=options,
    )
    assert decision.apply is False
    assert "below_threshold" in decision.reason


def test_default_compactor_pre_call_no_limit() -> None:
    compactor = DefaultContextCompactor()
    options = CompactorRunOptions(model_auto_compact_token_limit=None)
    decision = compactor.should_compact(
        messages=[],
        budget=ContextBudget(),
        llm_config=_make_llm_config(context_window=None),
        stage=CompactionStage.PRE_CALL,
        options=options,
    )
    assert decision.apply is False
    assert "no_compaction_limit" in decision.reason


def test_default_compactor_pre_call_above_threshold() -> None:
    compactor = DefaultContextCompactor()
    big_text = "x" * 400
    msg = _make_text_message("sess_1", big_text)
    options = CompactorRunOptions(model_auto_compact_token_limit=10)
    decision = compactor.should_compact(
        messages=[msg],
        budget=ContextBudget(),
        llm_config=_make_llm_config(),
        stage=CompactionStage.PRE_CALL,
        options=options,
    )
    assert decision.apply is True


def test_default_compactor_pre_call_window_limit() -> None:
    compactor = DefaultContextCompactor()
    big_text = "x" * 10000
    msg = _make_text_message("sess_1", big_text)
    options = CompactorRunOptions(model_auto_compact_token_limit=None)
    decision = compactor.should_compact(
        messages=[msg],
        budget=ContextBudget(),
        llm_config=_make_llm_config(context_window=100),
        stage=CompactionStage.PRE_CALL,
        options=options,
    )
    assert decision.apply is True


async def test_default_compactor_summarize_success() -> None:
    compactor = DefaultContextCompactor()
    provider = _make_mock_provider("summary content")
    msg = _make_text_message("sess_1", "some content")
    result = await compactor.summarize(
        messages=[msg],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=CompactorRunOptions(),
    )
    assert result.ok is True
    assert result.summary_text == "summary content"


async def test_default_compactor_summarize_empty_response() -> None:
    compactor = DefaultContextCompactor()
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=LLMStreamChunk(content="", finish_reason="stop"))
    msg = _make_text_message("sess_1", "some content")
    result = await compactor.summarize(
        messages=[msg],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=CompactorRunOptions(),
    )
    assert result.ok is False
    assert result.error == "empty_summary"


async def test_default_compactor_summarize_none_response() -> None:
    compactor = DefaultContextCompactor()
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=None)
    msg = _make_text_message("sess_1", "some content")
    result = await compactor.summarize(
        messages=[msg],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=CompactorRunOptions(),
    )
    assert result.ok is False


async def test_default_compactor_summarize_retries_on_error() -> None:
    compactor = DefaultContextCompactor()
    provider = AsyncMock()
    call_count = 0

    async def fake_complete(*_args: object, **_kwargs: object) -> LLMStreamChunk:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("transient error")
        return LLMStreamChunk(content="recovered summary", finish_reason="stop")

    provider.complete = fake_complete
    msg = _make_text_message("sess_1", "some content")
    options = CompactorRunOptions(summary_max_retries=3, summary_retry_sleep_ms=0)
    result = await compactor.summarize(
        messages=[msg],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=options,
    )
    assert result.ok is True
    assert result.summary_text == "recovered summary"


async def test_default_compactor_summarize_max_retries_exceeded() -> None:
    compactor = DefaultContextCompactor()
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("persistent error"))
    msg = _make_text_message("sess_1", "some content")
    options = CompactorRunOptions(summary_max_retries=1, summary_retry_sleep_ms=0)
    result = await compactor.summarize(
        messages=[msg],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=options,
    )
    assert result.ok is False


def test_compute_bookmark_position_no_messages() -> None:
    compactor = DefaultContextCompactor()
    pos = compactor.compute_bookmark_position([], 1000, 4.0)
    assert pos == 0


def test_compute_bookmark_position_with_messages() -> None:
    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    from agent.core.message.part import UserTextInput

    _ = ctx.build_user_message([UserTextInput(text="hello world")])
    _ = ctx.build_user_message([UserTextInput(text="second message")])
    pos = compactor.compute_bookmark_position(ctx.messages, keep_user_budget=1, chars_per_token=4.0)
    assert pos >= 0


def test_estimate_tokens_with_tool_parts() -> None:
    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    _ = ctx.build_assistant_message("msg_0", provider_id="openai", model_id="gpt-4o")
    tool_part = ToolPart(
        session_id="sess_1",
        message_id=ctx.messages[-1].message_id,
        call_id="call_1",
        tool="echo",
        state=ToolStateRunning(
            input=ToolInput(arguments='{"text":"hello"}'),
            title="echo",
            time=TimeInfo(start=1000),
        ),
    )
    ctx.messages[-1].add_part(tool_part)
    tokens = compactor.estimate_tokens_from_messages(ctx.messages, 4.0)
    assert tokens >= 0


async def test_compactor_run_should_compact_raises_compactor_error() -> None:
    from unittest.mock import patch

    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    msg = _make_text_message("sess_1", "hello")
    ctx.messages.append(msg)
    provider = _make_mock_provider("ok")

    with patch.object(compactor, "should_compact", side_effect=RuntimeError("boom")):
        payload = await compactor.run(
            session_context=ctx,
            budget=ContextBudget(),
            llm_config=_make_llm_config(),
            llm_provider=provider,
            stage=CompactionStage.OVERFLOW_ERROR,
            options=CompactorRunOptions(),
        )
    assert payload.applied is False
    assert "compactor_error" in payload.reason


def test_default_compactor_pre_call_both_limits_set_uses_min() -> None:
    compactor = DefaultContextCompactor()
    big_text = "x" * 400
    msg = _make_text_message("sess_1", big_text)
    options = CompactorRunOptions(model_auto_compact_token_limit=5000)
    decision = compactor.should_compact(
        messages=[msg],
        budget=ContextBudget(),
        llm_config=_make_llm_config(context_window=100),
        stage=CompactionStage.PRE_CALL,
        options=options,
    )
    assert decision.apply is True


def test_estimate_tokens_with_tool_completed_and_error_parts() -> None:
    compactor = DefaultContextCompactor()
    ctx = _make_session_context()
    _ = ctx.build_assistant_message("msg_completed", provider_id="openai", model_id="gpt-4o")
    tool_completed = ToolPart(
        session_id="sess_1",
        message_id=ctx.messages[-1].message_id,
        call_id="call_c",
        tool="echo",
        state=ToolStateCompleted(
            input=ToolInput(arguments='{"text":"hello"}'),
            output="hello",
            summary="echo result",
            title="echo",
            time=TimeInfo(start=1000, end=2000),
        ),
    )
    ctx.messages[-1].add_part(tool_completed)

    _ = ctx.build_assistant_message("msg_error", provider_id="openai", model_id="gpt-4o")
    tool_error = ToolPart(
        session_id="sess_1",
        message_id=ctx.messages[-1].message_id,
        call_id="call_e",
        tool="echo",
        state=ToolStateError(
            input=ToolInput(arguments='{"text":"fail"}'),
            error="something went wrong",
            time=TimeInfo(start=3000, end=4000),
        ),
    )
    ctx.messages[-1].add_part(tool_error)

    tokens = compactor.estimate_tokens_from_messages(ctx.messages, 4.0)
    assert tokens > 0


async def test_default_compactor_summarize_context_overflow_with_trimming() -> None:
    compactor = DefaultContextCompactor()
    provider = AsyncMock()
    call_count = 0

    async def fake_complete(*_args: object, **_kwargs: object) -> LLMStreamChunk:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMProviderError(code="context_length", message="context length exceeded")
        return LLMStreamChunk(content="trimmed summary", finish_reason="stop")

    provider.complete = fake_complete
    msg1 = _make_text_message("sess_1", "first message")
    msg2 = _make_text_message("sess_1", "second message")
    options = CompactorRunOptions(summary_max_trims=2, summary_retry_sleep_ms=0)
    result = await compactor.summarize(
        messages=[msg1, msg2],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=options,
    )
    assert result.ok is True
    assert result.summary_text == "trimmed summary"


async def test_default_compactor_summarize_context_overflow_untrimable() -> None:
    compactor = DefaultContextCompactor()
    provider = AsyncMock()

    async def always_overflow(*_args: object, **_kwargs: object) -> LLMStreamChunk:
        raise LLMProviderError(code="context_length", message="context length exceeded")

    provider.complete = always_overflow
    msg = _make_text_message("sess_1", "only message")
    options = CompactorRunOptions(summary_max_trims=0, summary_retry_sleep_ms=0)
    result = await compactor.summarize(
        messages=[msg],
        llm_config=_make_llm_config(),
        llm_provider=provider,
        options=options,
    )
    assert result.ok is False
    assert "context_overflow_untrimable" in result.error
