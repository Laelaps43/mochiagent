"""Agent prompt loader — reads markdown files and extracts sections."""

from __future__ import annotations

from pathlib import Path


class PromptLoader:
    """从 Markdown 文件加载系统提示词，支持按 H2 段落提取和 mtime 缓存。"""

    def __init__(self) -> None:
        self._cache: str | None = None
        self._mtime: float | None = None
        self._path: Path | None = None

    def load(self, path: Path, sections: list[str] | None = None) -> str | None:
        """读取文件并返回提示词内容。

        Args:
            path: Markdown 文件路径。
            sections: 要提取的 H2 标题列表（大小写不敏感）。
                      None 表示返回整个文件内容。

        Returns:
            提示词字符串，文件不存在时返回 None。
        """
        if not path.exists():
            return None

        mtime = path.stat().st_mtime
        if self._cache is None or self._path != path or self._mtime != mtime:
            self._cache = path.read_text(encoding="utf-8")
            self._path = path
            self._mtime = mtime

        if sections is None:
            return self._cache.strip() or None

        return self._extract_sections(self._cache, sections) or None

    @staticmethod
    def _extract_sections(content: str, sections: list[str]) -> str:
        """从 Markdown 内容中提取指定 H2 段落的正文。"""
        targets = {s.lower() for s in sections}
        result: list[str] = []
        current: list[str] | None = None
        in_fence = False

        for line in content.splitlines():
            if line.startswith("```"):
                in_fence = not in_fence

            if not in_fence and line.startswith("## "):
                if current is not None:
                    result.append("\n".join(current).strip())
                heading = line[3:].strip().lower()
                current = [] if heading in targets else None
            elif current is not None:
                current.append(line)

        if current is not None:
            result.append("\n".join(current).strip())

        return "\n\n".join(r for r in result if r)
