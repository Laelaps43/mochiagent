from __future__ import annotations

import os
import time
from pathlib import Path

from agent.common.prompts import PromptLoader


def _bump_mtime(path: Path) -> None:
    current = path.stat().st_mtime
    target = max(current + 1.0, time.time() + 1.0)
    os.utime(path, (target, target))


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    loader = PromptLoader()
    missing = tmp_path / "missing.md"

    assert loader.load(missing) is None


def test_load_reads_single_file(tmp_path: Path) -> None:
    loader = PromptLoader()
    prompt = tmp_path / "AGENTS.md"
    _ = prompt.write_text("single\n", encoding="utf-8")

    assert loader.load(prompt) == "single"


def test_load_refreshes_when_mtime_changes(tmp_path: Path) -> None:
    loader = PromptLoader()
    prompt = tmp_path / "AGENTS.md"
    _ = prompt.write_text("v1\n", encoding="utf-8")

    assert loader.load(prompt) == "v1"

    _ = prompt.write_text("v2\n", encoding="utf-8")
    _bump_mtime(prompt)

    assert loader.load(prompt) == "v2"
