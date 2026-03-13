from __future__ import annotations

from typing import override
import pytest

from agent.core.storage.memory import MemoryStorage
from agent.core.tools.result_postprocessor import (
    ToolResultPostProcessConfig,
    ToolResultPostProcessor,
)
from agent.types import ToolResult


def _make_result(
    *,
    success: bool = True,
    tool_name: str = "echo",
    result: object = None,
    error: str | None = None,
) -> ToolResult:
    return ToolResult(
        tool_call_id="call_1",
        tool_name=tool_name,
        result=result,
        success=success,
        error=error,
    )


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def processor() -> ToolResultPostProcessor:
    return ToolResultPostProcessor()


async def test_error_result_sets_summary(
    processor: ToolResultPostProcessor, storage: MemoryStorage
):
    result = _make_result(success=False, error="permission denied")
    out = await processor.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=storage,
    )
    assert out.success is False
    assert out.summary is not None
    assert "permission denied" in out.summary
    assert out.truncated is False


async def test_error_result_unknown_error(
    processor: ToolResultPostProcessor, storage: MemoryStorage
):
    result = _make_result(success=False, error=None)
    out = await processor.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=storage,
    )
    assert out.summary is not None
    assert "Unknown tool error" in out.summary


async def test_small_result_inline(processor: ToolResultPostProcessor, storage: MemoryStorage):
    result = _make_result(success=True, result="short answer")
    out = await processor.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=storage,
    )
    assert out.summary == "short answer"
    assert out.truncated is False
    assert out.artifact_ref is None


async def test_large_result_creates_artifact(storage: MemoryStorage):
    cfg = ToolResultPostProcessConfig(
        summary_max_chars=50,
        preview_head_chars=10,
        preview_tail_chars=10,
    )
    proc = ToolResultPostProcessor(config=cfg)
    big_text = "x" * 200
    result = _make_result(success=True, result=big_text)
    out = await proc.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=storage,
    )
    assert out.truncated is True
    assert out.artifact_ref is not None
    assert out.raw_size_chars == 200


async def test_large_result_no_artifact_support():
    from agent.core.storage.provider import (
        ArtifactMetadata,
        ArtifactReadResult,
        StorageProvider,
    )
    from agent.core.message import Message
    from agent.types import SessionMetadataData

    class _NoArtifactStorage(StorageProvider):
        @override
        async def save_session(
            self, session_id: str, session_data: SessionMetadataData
        ) -> None: ...
        @override
        async def load_session(self, session_id: str) -> SessionMetadataData | None:
            return None

        @override
        async def delete_session(self, session_id: str) -> None: ...
        @override
        async def session_exists(self, session_id: str) -> bool:
            return False

        @override
        async def list_sessions(self) -> list[str]:
            return []

        @override
        async def save_message(self, session_id: str, message: Message) -> None: ...
        @override
        async def load_messages(
            self, session_id: str, *, from_message_id: str | None = None
        ) -> list[Message]:
            return []

        @override
        async def delete_messages(self, session_id: str) -> None: ...
        @override
        async def save_artifact(
            self,
            session_id: str,
            kind: str,
            content: str,
            metadata: dict[str, object] | None = None,
        ) -> ArtifactMetadata:
            raise NotImplementedError

        @override
        async def read_artifact(
            self, artifact_ref: str, offset: int = 0, limit: int = 50000
        ) -> ArtifactReadResult:
            raise NotImplementedError

        @override
        async def delete_artifacts(self, session_id: str) -> None:
            raise NotImplementedError

    cfg = ToolResultPostProcessConfig(
        summary_max_chars=150,
        preview_head_chars=10,
        preview_tail_chars=10,
    )
    proc = ToolResultPostProcessor(config=cfg)
    big_text = "y" * 200
    result = _make_result(success=True, result=big_text)
    out = await proc.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=_NoArtifactStorage(),
    )
    assert out.truncated is True
    assert out.artifact_ref is None
    assert "Storage has no artifact support" in (out.summary or "")


async def test_dict_result_serialized(processor: ToolResultPostProcessor, storage: MemoryStorage):
    result = _make_result(success=True, result={"key": "value"})
    out = await processor.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=storage,
    )
    assert out.summary is not None
    assert "key" in out.summary


async def test_basemodel_result_serialized(
    processor: ToolResultPostProcessor, storage: MemoryStorage
):
    from agent.common.tools.results import ReadFileSuccess

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
    result = _make_result(success=True, result=model_result)
    out = await processor.process(
        session_id="sess_1",
        tool_result=result,
        tool_arguments={},
        storage=storage,
    )
    assert out.summary is not None
    assert "/tmp/test.txt" in out.summary
    assert "hello" in out.summary
