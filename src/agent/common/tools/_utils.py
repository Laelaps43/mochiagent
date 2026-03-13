from __future__ import annotations

import contextvars
import os
from pathlib import Path

from agent.core.utils import truncate_text

__all__ = ["truncate_text", "validate_path_within_workspace"]

_workspace_root_var: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_workspace_root_var", default=None
)


def set_workspace_root(root: Path) -> None:
    _ = _workspace_root_var.set(root.resolve(strict=False))


def get_workspace_root() -> Path | None:
    return _workspace_root_var.get()


def reset_workspace_root() -> None:
    _ = _workspace_root_var.set(None)


def validate_path_within_workspace(raw_path: str) -> str | None:
    """Return an error message if *raw_path* escapes the workspace root, else ``None``."""
    _workspace_root = _workspace_root_var.get()
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
    # Always verify the real path to prevent symlink escape (regardless of existence)
    real = Path(os.path.realpath(raw_path))
    try:
        _ = real.relative_to(_workspace_root)
    except ValueError:
        return (
            f"path '{raw_path}' is a symlink that resolves to '{real}' "
            f"which is outside the workspace root '{_workspace_root}'"
        )
    return None
