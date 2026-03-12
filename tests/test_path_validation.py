from __future__ import annotations

from pathlib import Path

import pytest

import agent.common.tools._utils as path_utils


@pytest.fixture(autouse=True)
def reset_workspace_root():
    path_utils.reset_workspace_root()
    yield
    path_utils.reset_workspace_root()


def test_no_workspace_always_passes(tmp_path: Path):
    assert path_utils.get_workspace_root() is None
    assert path_utils.validate_path_within_workspace(str(tmp_path / "anything")) is None


def test_path_inside_workspace_passes(tmp_path: Path):
    path_utils.set_workspace_root(tmp_path)
    inside = tmp_path / "subdir" / "file.txt"
    assert path_utils.validate_path_within_workspace(str(inside)) is None


def test_path_outside_workspace_blocked(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path_utils.set_workspace_root(workspace)

    outside = tmp_path / "other" / "secret.txt"
    result = path_utils.validate_path_within_workspace(str(outside))
    assert result is not None
    assert "outside the workspace root" in result


def test_path_traversal_blocked(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path_utils.set_workspace_root(workspace)

    traversal = str(workspace / ".." / ".." / "etc" / "passwd")
    result = path_utils.validate_path_within_workspace(traversal)
    assert result is not None
    assert "outside the workspace root" in result


def test_workspace_root_itself_passes(tmp_path: Path):
    path_utils.set_workspace_root(tmp_path)
    assert path_utils.validate_path_within_workspace(str(tmp_path)) is None


def test_symlink_escape_blocked(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "secret.txt"
    _ = outside_file.write_text("secret")

    symlink = workspace / "link_to_outside"
    symlink.symlink_to(outside_file)

    path_utils.set_workspace_root(workspace)
    result = path_utils.validate_path_within_workspace(str(symlink))
    assert result is not None
    assert "outside the workspace root" in result


def test_symlink_inside_workspace_passes(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    real_file = workspace / "real.txt"
    _ = real_file.write_text("data")

    symlink = workspace / "link.txt"
    symlink.symlink_to(real_file)

    path_utils.set_workspace_root(workspace)
    assert path_utils.validate_path_within_workspace(str(symlink)) is None


def test_nonexistent_path_inside_workspace_passes(tmp_path: Path):
    path_utils.set_workspace_root(tmp_path)
    nonexistent = tmp_path / "does" / "not" / "exist.txt"
    assert path_utils.validate_path_within_workspace(str(nonexistent)) is None


def test_set_workspace_root_resolves_path(tmp_path: Path):
    path_utils.set_workspace_root(tmp_path)
    assert path_utils.get_workspace_root() == tmp_path.resolve(strict=False)
