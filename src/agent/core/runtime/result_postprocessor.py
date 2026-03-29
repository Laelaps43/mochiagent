"""
Tool Result Post Processor - 工具结果后处理

目标：
- LLM 上下文只看到摘要文本
- 超长原始输出写入 storage artifact
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict

from agent.core.storage.provider import StorageProvider
from agent.core.tools.types import ToolResult


class ToolResultPostProcessConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    summary_max_chars: int = 50 * 1024  # 50KB, 超出写 artifact
    summary_max_lines: int = 3000
    preview_head_chars: int = 20000


class ToolResultPostProcessorStrategy(ABC):
    @abstractmethod
    async def process(
        self,
        *,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, object],
        storage: "StorageProvider",
    ) -> ToolResult:
        raise NotImplementedError


class ToolResultPostProcessor(ToolResultPostProcessorStrategy):
    def __init__(self, config: ToolResultPostProcessConfig | None = None):
        self.config: ToolResultPostProcessConfig = config or ToolResultPostProcessConfig()

    @override
    async def process(
        self,
        *,
        session_id: str,
        tool_result: ToolResult,
        tool_arguments: Mapping[str, object],
        storage: "StorageProvider",
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
        raw_lines = self._count_lines(raw_text)
        result.raw_size_chars = raw_size

        preview, is_truncated, preview_lines = self._truncate_for_context(
            raw_text,
            max_chars=self.config.preview_head_chars,
            max_lines=self.config.summary_max_lines,
        )

        if not is_truncated and raw_size <= self.config.summary_max_chars:
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
                    "raw_line_count": raw_lines,
                },
            )
            artifact_ref = artifact.artifact_ref
            artifact_path = artifact.path
        except NotImplementedError:
            artifact_ref = None
            artifact_path = None
        artifact_notice = (
            f"Complete output saved as artifact: {artifact_ref}\n"
            + "⚠️ You MUST use read_artifact with this artifact_ref to access the full data before responding.\n\n"
            if artifact_ref
            else ""
        )
        summary = (
            f"Tool `{result.tool_name}` output truncated for context "
            f"(chars: {raw_size} → {len(preview)}, lines: {raw_lines} → {preview_lines}).\n"
            + artifact_notice
            + f"[Preview]\n{preview}"
        )
        if len(summary) > self.config.summary_max_chars:
            summary = summary[: self.config.summary_max_chars]

        result.summary = summary
        result.artifact_ref = artifact_ref
        result.artifact_path = artifact_path
        result.truncated = True
        return result

    @staticmethod
    def _serialize_result(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, BaseModel):
            return json.dumps(value.model_dump(), ensure_ascii=False, indent=2)
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _count_lines(value: str) -> int:
        if not value:
            return 0
        return len(value.splitlines())

    @classmethod
    def _truncate_for_context(
        cls,
        value: str,
        *,
        max_chars: int,
        max_lines: int,
    ) -> tuple[str, bool, int]:
        if not value:
            return "", False, 0

        raw_chars = len(value)
        raw_lines = cls._count_lines(value)
        if raw_chars <= max_chars and raw_lines <= max_lines:
            return value, False, raw_lines

        safe_chars = max(1, max_chars)
        safe_lines = max(1, max_lines)

        preview_parts: list[str] = []
        used_chars = 0
        used_lines = 0

        for line in value.splitlines(keepends=True):
            if used_lines >= safe_lines or used_chars >= safe_chars:
                break
            remaining = safe_chars - used_chars
            if len(line) <= remaining:
                preview_parts.append(line)
                used_chars += len(line)
                used_lines += 1
                continue
            preview_parts.append(line[:remaining])
            used_chars += remaining
            break

        preview = "".join(preview_parts)
        if not preview:
            preview = value[:safe_chars]

        return preview, True, cls._count_lines(preview)
