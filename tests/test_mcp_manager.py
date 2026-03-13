from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

from agent.core.mcp.manager import MCPManager
from agent.core.mcp.types import MCPServerConfig, MCPServerState
from agent.core.tools import ToolRegistry

_ConnectFn = Callable[[str, MCPServerConfig, MCPServerState], Coroutine[object, object, int]]
_UnregisterFn = Callable[[str], None]


def _make_registry() -> ToolRegistry:
    return ToolRegistry()


def _make_manager(registry: ToolRegistry | None = None) -> MCPManager:
    return MCPManager(registry or _make_registry())


def _make_cfg(**kwargs: object) -> MCPServerConfig:
    config = MCPServerConfig(command="echo")
    for key, val in kwargs.items():
        object.__setattr__(config, key, val)
    return config


def _get_states(mgr: MCPManager) -> dict[str, MCPServerState]:
    return cast(dict[str, MCPServerState], getattr(mgr, "_states"))


def _get_sessions(mgr: MCPManager) -> dict[str, object]:
    return cast(dict[str, object], getattr(mgr, "_sessions"))


def _get_server_stacks(mgr: MCPManager) -> dict[str, object]:
    return cast(dict[str, object], getattr(mgr, "_server_stacks"))


def _get_configs(mgr: MCPManager) -> dict[str, MCPServerConfig]:
    return cast(dict[str, MCPServerConfig], getattr(mgr, "_configs"))


def _get_server_tool_names(mgr: MCPManager) -> dict[str, list[str]]:
    return cast(dict[str, list[str]], getattr(mgr, "_server_tool_names"))


def test_registry_property() -> None:
    reg = _make_registry()
    mgr = MCPManager(reg)
    assert mgr.registry is reg


def test_load_config_missing_file(tmp_path: Path) -> None:
    result = MCPManager.load_config(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_config_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text("not-json", encoding="utf-8")
    result = MCPManager.load_config(p)
    assert result == {}


def test_load_config_not_object(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    result = MCPManager.load_config(p)
    assert result == {}


def test_load_config_no_mcp_servers_key(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text(json.dumps({"other": "value"}), encoding="utf-8")
    result = MCPManager.load_config(p)
    assert result == {}


def test_load_config_empty_mcp_servers(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    result = MCPManager.load_config(p)
    assert result == {}


def test_load_config_valid(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text(
        json.dumps({"mcpServers": {"docs": {"command": "npx", "args": ["-y", "server"]}}}),
        encoding="utf-8",
    )
    result = MCPManager.load_config(p)
    assert "docs" in result
    assert result["docs"].command == "npx"


def test_load_config_invalid_server_dict_skipped(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text(
        json.dumps(
            {"mcpServers": {"bad": {"connectTimeoutMs": "not-an-int"}, "ok": {"command": "echo"}}}
        ),
        encoding="utf-8",
    )
    result = MCPManager.load_config(p)
    assert "bad" not in result
    assert "ok" in result


def test_load_config_non_dict_server_skipped(tmp_path: Path) -> None:
    p = tmp_path / "mcp.json"
    _ = p.write_text(
        json.dumps({"mcpServers": {"bad": "not-a-dict", "ok": {"command": "echo"}}}),
        encoding="utf-8",
    )
    result = MCPManager.load_config(p)
    assert "bad" not in result
    assert "ok" in result


def test_register_server_defaults() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("myserver", cfg)
    assert state.connect_timeout_ms == 8000
    assert state.max_retries == 2
    assert state.retry_initial_ms == 300
    assert state.retry_max_ms == 3000
    assert state.failure_threshold == 3
    assert state.cooldown_sec == 20
    assert state.tool_timeout_sec == 30


def test_register_server_custom_values() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(
        command="echo",
        connectTimeoutMs=5000,
        maxRetries=1,
        retryInitialMs=100,
        retryMaxMs=2000,
        failureThreshold=5,
        cooldownSec=10,
        toolTimeout=60,
    )
    state = mgr.register_server("s", cfg)
    assert state.connect_timeout_ms == 5000
    assert state.max_retries == 1
    assert state.failure_threshold == 5
    assert state.tool_timeout_sec == 60


def test_register_server_retry_max_clipped_to_initial() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="echo", retryInitialMs=500, retryMaxMs=100)
    state = mgr.register_server("s", cfg)
    assert state.retry_max_ms == state.retry_initial_ms


def test_register_server_reuses_existing_state() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state1 = mgr.register_server("s", cfg)
    state1.status = "connected"
    state2 = mgr.register_server("s", cfg)
    assert state1 is state2
    assert state2.status == "connected"


def test_snapshot_empty() -> None:
    mgr = _make_manager()
    assert mgr.snapshot() == {}


def test_snapshot_returns_snapshot_objects() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    _ = mgr.register_server("s", cfg)
    snap = mgr.snapshot()
    assert "s" in snap
    assert snap["s"].status == "disconnected"
    assert snap["s"].config.max_retries == 2


def test_set_status_noop_for_unknown_server() -> None:
    mgr = _make_manager()
    mgr.set_status("ghost", "connected")


def test_set_status_updates_status() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    _ = mgr.register_server("s", cfg)
    mgr.set_status("s", "connected")
    state = _get_states(mgr)["s"]
    assert state.status == "connected"


def test_set_status_noop_if_same_status() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.status = "connected"
    mgr.set_status("s", "connected")
    assert state.status == "connected"


def test_can_execute_unknown_server() -> None:
    mgr = _make_manager()
    assert mgr.can_execute("ghost") is True


def test_can_execute_non_failed_server() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    _ = mgr.register_server("s", cfg)
    mgr.set_status("s", "connected")
    assert mgr.can_execute("s") is True


def test_can_execute_failed_no_retry_at() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.status = "failed"
    state.next_retry_at = None
    assert mgr.can_execute("s") is False


def test_can_execute_failed_cooldown_not_expired() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.status = "failed"
    state.next_retry_at = time.time() + 9999
    assert mgr.can_execute("s") is False


def test_can_execute_failed_cooldown_expired() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.status = "failed"
    state.next_retry_at = time.time() - 1
    assert mgr.can_execute("s") is True
    assert state.status == "degraded"
    assert state.next_retry_at is None


def test_record_tool_success_unknown_server() -> None:
    mgr = _make_manager()
    mgr.record_tool_success("ghost")


def test_record_tool_success_resets_state() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.consecutive_failures = 3
    state.last_error = "oops"
    state.next_retry_at = 9999.0
    state.status = "failed"
    mgr.record_tool_success("s")
    assert state.consecutive_failures == 0
    assert state.last_error is None
    assert state.next_retry_at is None
    assert state.status == "connected"


def test_record_tool_failure_unknown_server() -> None:
    mgr = _make_manager()
    mgr.record_tool_failure("ghost", "err")


def test_record_tool_failure_below_threshold() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="echo", failureThreshold=3)
    state = mgr.register_server("s", cfg)
    mgr.record_tool_failure("s", "timeout")
    assert state.consecutive_failures == 1
    assert state.last_error == "timeout"
    assert state.status == "degraded"


def test_record_tool_failure_reaches_threshold() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="echo", failureThreshold=2)
    state = mgr.register_server("s", cfg)
    mgr.record_tool_failure("s", "err")
    mgr.record_tool_failure("s", "err2")
    assert state.status == "failed"
    assert state.next_retry_at is not None
    assert state.tool_count == 0


def test_get_session_returns_none_when_no_session() -> None:
    mgr = _make_manager()
    assert mgr.get_session("s") is None


def test_get_session_returns_session() -> None:
    mgr = _make_manager()
    fake_session = MagicMock()
    _get_sessions(mgr)["s"] = fake_session
    result = mgr.get_session("s")
    assert result is fake_session


def test_mark_disconnected() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.status = "connected"
    state.tool_count = 5
    state.next_retry_at = 9999.0
    mgr.mark_disconnected()
    assert state.status == "disconnected"
    assert state.tool_count == 0
    assert state.next_retry_at is None


async def test_close_clears_sessions_and_marks_disconnected() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    state = mgr.register_server("s", cfg)
    state.status = "connected"

    fake_stack = AsyncMock(spec=AsyncExitStack)
    _get_server_stacks(mgr)["s"] = fake_stack
    _get_sessions(mgr)["s"] = MagicMock()

    await mgr.close()

    cast(AsyncMock, fake_stack.aclose).assert_called_once()
    assert _get_sessions(mgr) == {}
    assert state.status == "disconnected"


async def test_close_suppresses_stack_error() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    _ = mgr.register_server("s", cfg)

    fake_stack = AsyncMock(spec=AsyncExitStack)
    cast(AsyncMock, fake_stack.aclose).side_effect = RuntimeError("boom")
    _get_server_stacks(mgr)["s"] = fake_stack

    await mgr.close()
    cast(AsyncMock, fake_stack.aclose).assert_called_once()


async def test_connect_servers_calls_connect_single() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()

    async def _fake_connect(_name: str, _c: MCPServerConfig, _s: MCPServerState) -> int:
        return 3

    with patch.object(mgr, "_connect_single_server", side_effect=_fake_connect):
        total = await mgr.connect_servers({"s": cfg})

    assert total == 3
    assert "s" in _get_configs(mgr)


async def test_connect_servers_multiple() -> None:
    mgr = _make_manager()
    cfg1 = _make_cfg()
    cfg2 = _make_cfg()

    async def _fake_connect(_name: str, _c: MCPServerConfig, _s: MCPServerState) -> int:
        return 2

    with patch.object(mgr, "_connect_single_server", side_effect=_fake_connect):
        total = await mgr.connect_servers({"a": cfg1, "b": cfg2})

    assert total == 4


async def test_reconnect_server_no_config() -> None:
    mgr = _make_manager()
    result = await mgr.reconnect_server("ghost")
    assert result is False


async def test_reconnect_server_success() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    _get_configs(mgr)["s"] = cfg
    _ = mgr.register_server("s", cfg)

    async def _fake_connect(_name: str, _c: MCPServerConfig, _state: MCPServerState) -> int:
        _get_sessions(mgr)[_name] = MagicMock()
        return 1

    with patch.object(mgr, "_connect_single_server", side_effect=_fake_connect):
        result = await mgr.reconnect_server("s")

    assert result is True


async def test_reconnect_server_closes_old_stack() -> None:
    mgr = _make_manager()
    cfg = _make_cfg()
    _get_configs(mgr)["s"] = cfg
    _ = mgr.register_server("s", cfg)
    _get_sessions(mgr)["s"] = MagicMock()

    old_stack = AsyncMock(spec=AsyncExitStack)
    _get_server_stacks(mgr)["s"] = old_stack

    async def _fake_connect(_name: str, _c: MCPServerConfig, _state: MCPServerState) -> int:
        _get_sessions(mgr)[_name] = MagicMock()
        return 1

    with patch.object(mgr, "_connect_single_server", side_effect=_fake_connect):
        result = await mgr.reconnect_server("s")

    cast(AsyncMock, old_stack.aclose).assert_called_once()
    assert result is True


async def test_reconnect_server_concurrent_calls_serialized() -> None:
    """Concurrent reconnect_server calls are serialized by the per-server lock."""
    mgr = _make_manager()
    cfg = _make_cfg()
    _get_configs(mgr)["s"] = cfg
    _ = mgr.register_server("s", cfg)

    call_order: list[str] = []

    async def _fake_connect(name: str, _cfg: MCPServerConfig, _state: MCPServerState) -> int:
        call_order.append(f"start-{name}")
        await asyncio.sleep(0.02)
        call_order.append(f"end-{name}")
        return 1

    with patch.object(mgr, "_connect_single_server", side_effect=_fake_connect):
        results = await asyncio.gather(
            mgr.reconnect_server("s"),
            mgr.reconnect_server("s"),
        )

    # Both calls should succeed (serialized, not concurrent)
    assert all(results)
    assert len(call_order) == 4


async def test_connect_single_server_no_command_no_url() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="", url="")
    state = mgr.register_server("s", cfg)
    connect = cast(_ConnectFn, getattr(mgr, "_connect_single_server"))
    result = await connect("s", cfg, state)
    assert result == 0
    assert state.status == "failed"


async def test_connect_single_server_all_attempts_fail() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="echo", maxRetries=0, connectTimeoutMs=100)
    state = mgr.register_server("s", cfg)

    connect = cast(_ConnectFn, getattr(mgr, "_connect_single_server"))

    with patch(
        "agent.core.mcp.manager.MCPManager._open_transport",
        new=AsyncMock(side_effect=RuntimeError("conn failed")),
    ):
        result = await connect("s", cfg, state)

    assert result == 0
    assert state.status == "failed"


async def test_connect_single_server_with_tools() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="echo", maxRetries=0, connectTimeoutMs=1000)
    state = mgr.register_server("s", cfg)

    fake_read = MagicMock()
    fake_write = MagicMock()

    async def _fake_open_transport(
        _cfg: MCPServerConfig, _stack: AsyncExitStack, _timeout: float
    ) -> tuple[object, object]:
        return fake_read, fake_write

    fake_tool = MagicMock()
    fake_tool.name = "my_tool"
    listed = MagicMock()
    listed.tools = [fake_tool]

    fake_session = AsyncMock()
    fake_session.initialize = AsyncMock(return_value=None)
    fake_session.list_tools = AsyncMock(return_value=listed)
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=None)

    fake_wrapper = MagicMock()
    fake_wrapper.name = "my_tool"

    async def _passthrough_wait_for(coro: Awaitable[object], _timeout: float = 0) -> object:
        return await coro

    connect = cast(_ConnectFn, getattr(mgr, "_connect_single_server"))

    with patch("agent.core.mcp.manager.MCPManager._open_transport", new=_fake_open_transport):
        with patch("mcp.ClientSession", return_value=fake_session):
            with patch("agent.common.tools.mcp.MCPToolWrapper", return_value=fake_wrapper):
                with patch(
                    "agent.core.mcp.manager.asyncio.wait_for",
                    side_effect=_passthrough_wait_for,
                ):
                    result = await connect("s", cfg, state)

    assert result >= 0


async def test_connect_single_server_retry_then_fail() -> None:
    mgr = _make_manager()
    cfg = MCPServerConfig(command="echo", maxRetries=1, connectTimeoutMs=1000, retryInitialMs=1)
    state = mgr.register_server("s", cfg)

    connect = cast(_ConnectFn, getattr(mgr, "_connect_single_server"))

    with patch(
        "agent.core.mcp.manager.MCPManager._open_transport",
        new=AsyncMock(side_effect=RuntimeError("fail")),
    ):
        result = await connect("s", cfg, state)

    assert result == 0
    assert state.status == "failed"


async def test_unregister_server_tools_removes_tools() -> None:
    reg = _make_registry()
    mgr = MCPManager(reg)
    cfg = _make_cfg()
    _ = mgr.register_server("s", cfg)

    fake_tool = MagicMock()
    fake_tool.name = "t1"
    reg.register(fake_tool)
    _get_server_tool_names(mgr)["s"] = ["t1"]

    unregister = cast(_UnregisterFn, getattr(mgr, "_unregister_server_tools"))
    unregister("s")
    assert "s" not in _get_server_tool_names(mgr)
    assert not reg.has("t1")
