from __future__ import annotations

from collections.abc import Iterator

import pytest

from agent.sandbox.context import reset_sandbox
from agent.core.bus import MessageBus
from agent.core.session import SessionManager
from agent.core.storage import MemoryStorage


@pytest.fixture(autouse=True)
def reset_sandbox_after_each_test() -> Iterator[None]:
    yield
    reset_sandbox()


@pytest.fixture
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def bus() -> MessageBus:
    return MessageBus(max_concurrent=4)


@pytest.fixture
def session_manager(bus: MessageBus, storage: MemoryStorage) -> SessionManager:
    return SessionManager(bus=bus, storage=storage)
