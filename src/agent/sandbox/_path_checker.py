"""Path checking helpers — extracted from ToolSecurityGuard.

This is a *private* module used by all sandbox backends.
"""

from __future__ import annotations

import os
from pathlib import Path


def normalize_path(raw_path: str, *, root: Path, cwd: Path | None = None) -> Path:
    """Resolve *raw_path* to an absolute ``Path``.

    - ``~`` expands to ``$HOME``
    - Relative paths resolve against *cwd* (or *root* as fallback)
    - Symlinks are **not** followed here (callers do a separate real-path check)
    """
    base = cwd or root
    text = raw_path.strip()
    if text.startswith("~/"):
        candidate = Path.home() / text[2:]
    else:
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = base / candidate
    return candidate.resolve(strict=False)


def is_inside_root(path: Path, root: Path) -> bool:
    """Return ``True`` when *path* is within *root*."""
    try:
        _ = path.relative_to(root)
    except ValueError:
        return False
    else:
        return True


def check_path_access(
    raw_path: str,
    *,
    root: Path,
    restrict: bool,
    cwd: Path | None = None,
) -> tuple[bool, str]:
    """Validate *raw_path* is accessible inside *root*.

    Returns ``(allowed, reason)``.
    """
    if not restrict:
        return True, "workspace restriction disabled"

    normalized = normalize_path(raw_path, root=root, cwd=cwd)
    if not is_inside_root(normalized, root):
        return (
            False,
            f"path '{raw_path}' resolves to '{normalized}' outside workspace root '{root}'",
        )

    # Symlink escape check (even if the file does not exist yet).
    real = Path(os.path.realpath(raw_path))
    if not is_inside_root(real, root):
        return (
            False,
            f"path '{raw_path}' is a symlink that resolves to '{real}' outside workspace root '{root}'",
        )

    return True, "path allowed"


def extract_paths_from_command(command: str) -> list[str]:
    """Best-effort extraction of path-like tokens from *command*."""
    import shlex

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return ["__INVALID_COMMAND__"]

    def _is_path_like(value: str) -> bool:
        if value.startswith("/") or value.startswith("~/"):
            return True
        if value.startswith("./") or value.startswith("../") or value in {".", ".."}:
            return True
        return "/" in value

    path_candidates: list[str] = []
    for token in tokens:
        if token == "__INVALID_COMMAND__":
            path_candidates.append(token)
            continue
        if token.startswith("-"):
            if "=" in token:
                value = token.split("=", 1)[1]
                if value and _is_path_like(value):
                    path_candidates.append(value)
            continue
        if _is_path_like(token):
            path_candidates.append(token)

    return path_candidates


def check_command_tokens(command: str, deny_tokens: set[str]) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` after scanning *command* for denied tokens."""
    for token in deny_tokens:
        if token in command:
            return False, f"command contains denied token: {token!r}"
    return True, "command allowed"
