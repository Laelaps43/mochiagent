"""Sandbox factory — auto-detect platform and create the right backend."""

from __future__ import annotations

import sys

from loguru import logger

from agent.sandbox.abc import Sandbox
from agent.sandbox.types import SandboxConfig


def create_sandbox(config: SandboxConfig | None = None) -> Sandbox:
    """Create a sandbox instance based on *config* and the current platform.

    If ``config.backend`` is ``"noop"`` (the default) the factory tries
    to auto-detect the best OS-level backend for the current platform.
    Set ``config.backend`` explicitly to skip auto-detection.
    """
    if config is None:
        config = SandboxConfig()

    backend = config.backend

    # Auto-detect when backend is "noop"
    if backend == "noop":
        detected = _detect_platform_backend()
        if detected != "noop":
            logger.info("Auto-detected platform sandbox: {}", detected)
            backend = detected

    if backend == "noop":
        from agent.sandbox.backends.noop import NoopSandbox
        return NoopSandbox(config)
    elif backend == "seatbelt":
        if sys.platform != "darwin":
            logger.warning("seatbelt backend requested but not on macOS; falling back to noop")
            from agent.sandbox.backends.noop import NoopSandbox
            return NoopSandbox(config)
        from agent.sandbox.backends.seatbelt import SeatbeltSandbox
        return SeatbeltSandbox(config)
    elif backend == "bwrap":
        if sys.platform != "linux":
            logger.warning("bwrap backend requested but not on Linux; falling back to noop")
            from agent.sandbox.backends.noop import NoopSandbox
            return NoopSandbox(config)
        from agent.sandbox.backends.bwrap import BwrapSandbox
        return BwrapSandbox(config)
    elif backend == "docker":
        raise NotImplementedError("DockerSandbox is not yet implemented")
    else:
        raise ValueError(f"Unknown sandbox backend: {backend}")


def _detect_platform_backend() -> str:
    """Return the best available OS-level backend, or ``"noop"``."""
    import shutil

    if sys.platform == "darwin":
        if shutil.which("/usr/bin/sandbox-exec"):
            return "seatbelt"
    elif sys.platform == "linux":
        if shutil.which("bwrap"):
            return "bwrap"

    return "noop"
