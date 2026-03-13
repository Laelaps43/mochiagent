from __future__ import annotations

from typing import cast
import pytest

from agent.core.message.part import (
    PartBase,
    TextPart,
    ReasoningPart,
    ToolPart,
    ToolInput,
    ToolStatePending,
    ToolStateRunning,
    ToolStateCompleted,
    ToolStateError,
    TimeInfo,
    UserTextInput,
    UserInput,
)
from agent.types import ToolCallPayload, ToolResult


def _tool_call(call_id: str = "call_1", name: str = "echo", args: str = "{}") -> ToolCallPayload:
    from agent.types import ToolFunctionPayload

    return ToolCallPayload(
        id=call_id,
        type="function",
        function=ToolFunctionPayload(name=name, arguments=args),
    )


def _make_running_part(call_id: str = "call_1", name: str = "echo") -> ToolPart:
    return ToolPart.create_running(
        session_id="sess",
        message_id="msg",
        tool_call=_tool_call(call_id=call_id, name=name),
    )


def test_time_info_no_end() -> None:
    t = TimeInfo(start=1000)
    assert t.end is None


def test_time_info_with_end() -> None:
    t = TimeInfo(start=1000, end=2000)
    assert t.end == 2000


def test_part_base_to_llm_format_returns_none() -> None:
    base = PartBase(session_id="s", message_id="m")
    assert base.to_llm_format() is None


def test_text_part_basic() -> None:
    part = TextPart(session_id="s", message_id="m", text="hello world")
    assert part.text == "hello world"
    assert part.type == "text"
    assert part.synthetic is None
    assert part.ignored is None


def test_text_part_to_llm_format() -> None:
    part = TextPart(session_id="s", message_id="m", text="hi")
    fmt = part.to_llm_format()
    assert fmt is not None
    assert fmt["type"] == "text"
    assert fmt["content"] == "hi"


def test_text_part_with_synthetic_ignored() -> None:
    part = TextPart(session_id="s", message_id="m", text="x", synthetic=True, ignored=False)
    assert part.synthetic is True
    assert part.ignored is False


def test_user_text_input_to_part() -> None:
    inp = UserTextInput(text="test message")
    part = inp.to_part("sess_1", "msg_1")
    assert isinstance(part, TextPart)
    assert part.text == "test message"
    assert part.session_id == "sess_1"
    assert part.message_id == "msg_1"


def test_user_text_input_metadata_preserved() -> None:
    inp = UserTextInput(text="test", synthetic=True, ignored=False, metadata={"k": "v"})
    part = inp.to_part("s", "m")
    assert part.synthetic is True
    assert part.ignored is False
    assert part.metadata == {"k": "v"}


def test_user_input_alias() -> None:
    assert UserInput is UserTextInput


def test_reasoning_part_to_llm_format_none() -> None:
    part = ReasoningPart(
        session_id="s", message_id="m", text="thinking...", time=TimeInfo(start=1000)
    )
    assert part.to_llm_format() is None


def test_reasoning_part_type() -> None:
    part = ReasoningPart(session_id="s", message_id="m", text="...", time=TimeInfo(start=100))
    assert part.type == "reasoning"


def test_tool_part_create_running_basic() -> None:
    part = _make_running_part()
    assert part.tool == "echo"
    assert part.call_id == "call_1"
    assert isinstance(part.state, ToolStateRunning)
    assert part.state.status == "running"
    assert part.state.title == "echo"


def test_tool_part_create_running_stores_arguments() -> None:
    part = ToolPart.create_running(
        session_id="s",
        message_id="m",
        tool_call=_tool_call(args='{"text": "hello"}'),
    )
    assert part.state.input.arguments == '{"text": "hello"}'


def test_update_to_completed_with_summary() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result="output text",
        success=True,
        summary="summarized output",
    )
    completed = part.update_to_completed(result)
    assert isinstance(completed.state, ToolStateCompleted)
    assert completed.state.status == "completed"
    assert completed.state.output == "summarized output"


def test_update_to_completed_fallback_to_str_result() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result="raw output",
        success=True,
        summary=None,
    )
    completed = part.update_to_completed(result)
    assert isinstance(completed.state, ToolStateCompleted)
    assert completed.state.output == "raw output"


def test_update_to_completed_dict_result_json() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result={"key": "value"},
        success=True,
        summary=None,
    )
    completed = part.update_to_completed(result)
    assert isinstance(completed.state, ToolStateCompleted)
    assert "key" in completed.state.output


def test_update_to_completed_preserves_artifact() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result="big output",
        success=True,
        artifact_ref="artifact_123",
        artifact_path="/path/to/artifact",
        raw_size_chars=1000,
        truncated=True,
    )
    completed = part.update_to_completed(result)
    assert isinstance(completed.state, ToolStateCompleted)
    assert completed.state.artifact_ref == "artifact_123"
    assert completed.state.truncated is True


def test_update_to_completed_wrong_state_raises() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result="",
        success=True,
    )
    completed = part.update_to_completed(result)
    with pytest.raises(ValueError, match="expected 'running'"):
        _ = completed.update_to_completed(result)


def test_update_to_error_basic() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result=None,
        success=False,
        error="permission denied",
    )
    errored = part.update_to_error(result)
    assert isinstance(errored.state, ToolStateError)
    assert errored.state.error == "permission denied"


def test_update_to_error_none_error_fallback() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result=None,
        success=False,
        error=None,
    )
    errored = part.update_to_error(result)
    assert isinstance(errored.state, ToolStateError)
    assert errored.state.error == "Unknown error"


def test_update_to_error_wrong_state_raises() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result=None,
        success=False,
        error="err",
    )
    errored = part.update_to_error(result)
    with pytest.raises(ValueError, match="expected 'running'"):
        _ = errored.update_to_error(result)


def test_tool_part_running_to_llm_format() -> None:
    part = _make_running_part()
    fmt = part.to_llm_format()
    assert fmt is not None
    assert fmt["type"] == "tool"
    assert "tool_call" in fmt
    assert "tool_result" not in fmt


def test_tool_part_completed_to_llm_format() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result="ok",
        success=True,
        summary="ok summary",
    )
    completed = part.update_to_completed(result)
    fmt = completed.to_llm_format()
    assert fmt is not None
    assert "tool_call" in fmt
    assert "tool_result" in fmt
    tool_result = fmt["tool_result"]
    assert isinstance(tool_result, dict)
    tr = cast(dict[str, object], tool_result)
    assert tr["role"] == "tool"
    assert "ok summary" in str(tr["content"])


def test_tool_part_error_to_llm_format() -> None:
    part = _make_running_part()
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="echo",
        result=None,
        success=False,
        error="boom",
    )
    errored = part.update_to_error(result)
    fmt = errored.to_llm_format()
    assert fmt is not None
    assert "tool_result" in fmt
    tool_result = fmt["tool_result"]
    assert isinstance(tool_result, dict)
    tr = cast(dict[str, object], tool_result)
    assert "Error: boom" in str(tr["content"])


def test_tool_part_pending_to_llm_format_returns_none() -> None:
    part = ToolPart(
        session_id="s",
        message_id="m",
        call_id="call_1",
        tool="echo",
        state=ToolStatePending(input=ToolInput(), raw="{}"),
    )
    fmt = part.to_llm_format()
    assert fmt is None


def test_update_to_completed_basemodel_result() -> None:
    from agent.common.tools.results import ReadFileSuccess

    part = _make_running_part()
    model_result = ReadFileSuccess(
        path="/tmp/test.txt",
        content="hello",
        truncated=False,
        size_bytes=5,
        offset=0,
        limit=100000,
        next_offset=5,
        eof=True,
    )
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="read_file",
        result=model_result,
        success=True,
        summary=None,
    )
    completed = part.update_to_completed(result)
    assert isinstance(completed.state, ToolStateCompleted)
    assert "/tmp/test.txt" in completed.state.output
    assert "hello" in completed.state.output
