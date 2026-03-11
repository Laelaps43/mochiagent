"""
Tool Result Post Processor - 工具结果后处理

目标：
- LLM 上下文只看到摘要文本
- 超长原始输出写入 storage artifact
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict

from agent.core.storage import StorageProvider
from agent.types import ToolResult


class ToolResultPostProcessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary_max_chars: int = 4000
    preview_head_chars: int = 1500
    preview_tail_chars: int = 1000


class ToolResultPostProcessorStrategy(ABC):
    @abstractmethod
    async def process(
        self,
        *,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, Any],
        storage: StorageProvider,
    ) -> ToolResult:
        raise NotImplementedError


class ToolResultPostProcessor(ToolResultPostProcessorStrategy):
    def __init__(self, config: ToolResultPostProcessConfig | None = None):
        self.config = config or ToolResultPostProcessConfig()

    async def process(
        self,
        *,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, Any],
        storage: StorageProvider,
    ) -> ToolResult:
        result = tool_result.model_copy(deep=True)

        if not result.success:
            error_text = result.error or "Unknown tool error"
            result.summary = f"Tool `{result.tool_name}` failed: {error_text}"
            result.truncated = False
            result.raw_size_chars = len(error_text)
            return result

        raw_text = self._serialize_result(result.result)
        raw_size = len(raw_text)
        result.raw_size_chars = raw_size

        if raw_size <= self.config.summary_max_chars:
            result.summary = raw_text
            result.truncated = False
            return result

        artifact_ref: str | None = None
        artifact_path: str | None = None
        try:
            artifact = await storage.save_artifact(
                session_id=session_id,
                kind="tool_result",
                content=raw_text,
                metadata={
                    "tool_name": result.tool_name,
                    "tool_call_id": result.tool_call_id,
                    "arguments": tool_arguments,
                    "raw_size_chars": raw_size,
                },
            )
            artifact_ref = artifact.artifact_ref
            artifact_path = artifact.path
        except NotImplementedError:
            artifact_ref = None
            artifact_path = None
        head = raw_text[: self.config.preview_head_chars]
        tail = raw_text[-self.config.preview_tail_chars :]
        summary = (
            f"Tool `{result.tool_name}` produced large output ({raw_size} chars), truncated for context.\n"
            + (
                f"Artifact: {artifact_ref}\nPath: {artifact_path}\n"
                "Use `read_file` with `path`, `offset`, `limit` for chunked reading.\n\n"
                if artifact_ref and artifact_path
                else "Storage has no artifact support, only preview is available.\n\n"
            )
            + f"[Preview head]\n{head}\n\n[Preview tail]\n{tail}"
        )
        if len(summary) > self.config.summary_max_chars:
            summary = summary[: self.config.summary_max_chars]

        result.summary = summary
        result.artifact_ref = artifact_ref
        result.artifact_path = artifact_path
        result.truncated = True
        return result

    @staticmethod
    def _serialize_result(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(value)
