"""Agent prompt loader — reads a markdown file with mtime-based cache."""

from __future__ import annotations

from pathlib import Path


class PromptLoader:
    """从 Markdown 文件加载系统提示词，支持 mtime 缓存。"""

    def __init__(self) -> None:
        self._cache: dict[Path, tuple[float, str]] = {}

    def load(self, path: Path) -> str | None:
        """读取文件并返回提示词内容。

        Args:
            path: Markdown 文件路径。

        Returns:
            提示词字符串，文件不存在时返回 None。
        """
        content = self._load_raw(path)
        if content is None:
            return None

        return content.strip() or None

    def _load_raw(self, path: Path) -> str | None:
        normalized = path.resolve(strict=False)
        if not normalized.exists():
            return None

        mtime = normalized.stat().st_mtime
        cached = self._cache.get(normalized)
        if cached is None or cached[0] != mtime:
            self._cache[normalized] = (mtime, normalized.read_text(encoding="utf-8"))

        return self._cache[normalized][1]
