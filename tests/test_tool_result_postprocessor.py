from __future__ import annotations

from pathlib import Path

import pytest

from agent.core.storage import MemoryStorage
from agent.core.tools import ToolResultPostProcessConfig, ToolResultPostProcessor
from agent.types import ToolResult


@pytest.mark.asyncio
async def test_postprocessor_truncates_and_persists_artifact(tmp_path: Path):
    storage = MemoryStorage(artifact_root=tmp_path / "artifacts")
    processor = ToolResultPostProcessor(
        ToolResultPostProcessConfig(
            summary_max_chars=120, preview_head_chars=30, preview_tail_chars=20
        )
    )
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="web_fetch",
        result="x" * 500,
        success=True,
    )

    out = await processor.process(
        session_id="s1",
        tool_result=result,
        tool_arguments={"url": "https://example.com"},
        storage=storage,
    )

    assert out.truncated is True
    assert out.artifact_ref is not None
    assert out.artifact_path is not None
    assert "Artifact:" in (out.summary or "")
    assert Path(out.artifact_path).exists()


@pytest.mark.asyncio
async def test_postprocessor_keeps_small_output():
    storage = MemoryStorage()
    processor = ToolResultPostProcessor(
        ToolResultPostProcessConfig(
            summary_max_chars=200, preview_head_chars=50, preview_tail_chars=50
        )
    )
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="exec",
        result="hello",
        success=True,
    )

    out = await processor.process(
        session_id="s1",
        tool_result=result,
        tool_arguments={"command": "echo hello"},
        storage=storage,
    )
    assert out.truncated is False
    assert out.summary == "hello"
    assert out.artifact_ref is None
