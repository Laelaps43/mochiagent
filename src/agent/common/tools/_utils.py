from __future__ import annotations

import os
from pathlib import Path

from agent.core.utils import truncate_text

__all__ = ["truncate_text", "validate_path_within_workspace"]

_workspace_root: Path | None = None


def set_workspace_root(root: Path) -> None:
    global _workspace_root  # noqa: PLW0603
    _workspace_root = root.resolve(strict=False)


def validate_path_within_workspace(raw_path: str) -> str | None:
    """Return an error message if *raw_path* escapes the workspace root, else ``None``."""
    if _workspace_root is None:
        return None
    resolved = Path(raw_path).resolve(strict=False)
    try:
        _ = resolved.relative_to(_workspace_root)
    except ValueError:
        return (
            f"path '{raw_path}' resolves to '{resolved}' "
            f"which is outside the workspace root '{_workspace_root}'"
        )
    # Prevent symlink escape: if the target exists, verify the real path too
    if resolved.exists():
        real = Path(os.path.realpath(resolved))
        try:
            _ = real.relative_to(_workspace_root)
        except ValueError:
            return (
                f"path '{raw_path}' is a symlink that resolves to '{real}' "
                f"which is outside the workspace root '{_workspace_root}'"
            )
    return None
