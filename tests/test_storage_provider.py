from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast, override

import pytest

from agent.core.message import Message
from agent.core.storage.provider import ArtifactMetadata, ArtifactReadResult, StorageProvider
from agent.types import SessionMetadataData


_RawFn = Callable[..., Awaitable[object]]


class _ConcreteStorage(StorageProvider):
    @override
    async def save_session(self, session_id: str, session_data: SessionMetadataData) -> None:
        raw = cast(_RawFn, vars(StorageProvider)["save_session"])
        _ = await raw(self, session_id, session_data)

    @override
    async def load_session(self, session_id: str) -> SessionMetadataData | None:
        raw = cast(_RawFn, vars(StorageProvider)["load_session"])
        return cast("SessionMetadataData | None", await raw(self, session_id))

    @override
    async def delete_session(self, session_id: str) -> None:
        raw = cast(_RawFn, vars(StorageProvider)["delete_session"])
        _ = await raw(self, session_id)

    @override
    async def session_exists(self, session_id: str) -> bool:
        raw = cast(_RawFn, vars(StorageProvider)["session_exists"])
        return cast(bool, await raw(self, session_id))

    @override
    async def list_sessions(self) -> list[str]:
        raw = cast(_RawFn, vars(StorageProvider)["list_sessions"])
        return cast("list[str]", await raw(self))

    @override
    async def save_message(self, session_id: str, message: Message) -> None:
        raw = cast(_RawFn, vars(StorageProvider)["save_message"])
        _ = await raw(self, session_id, message)

    @override
    async def load_messages(
        self, session_id: str, *, from_message_id: str | None = None
    ) -> list[Message]:
        raw = cast(_RawFn, vars(StorageProvider)["load_messages"])
        return cast("list[Message]", await raw(self, session_id, from_message_id=from_message_id))

    @override
    async def delete_messages(self, session_id: str) -> None:
        raw = cast(_RawFn, vars(StorageProvider)["delete_messages"])
        _ = await raw(self, session_id)

    @override
    async def save_artifact(
        self,
        session_id: str,
        kind: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactMetadata:
        raw = cast(
            "Callable[..., Awaitable[ArtifactMetadata]]", vars(StorageProvider)["save_artifact"]
        )
        return await raw(self, session_id, kind, content, metadata)

    @override
    async def read_artifact(
        self,
        artifact_ref: str,
        offset: int = 0,
        limit: int = 50000,
    ) -> ArtifactReadResult:
        raw = cast(
            "Callable[..., Awaitable[ArtifactReadResult]]", vars(StorageProvider)["read_artifact"]
        )
        return await raw(self, artifact_ref, offset, limit)

    @override
    async def delete_artifacts(self, session_id: str) -> None:
        raw = cast("Callable[..., Awaitable[None]]", vars(StorageProvider)["delete_artifacts"])
        await raw(self, session_id)


@pytest.fixture
def storage() -> _ConcreteStorage:
    return _ConcreteStorage()


async def test_save_artifact_raises(storage: _ConcreteStorage):
    with pytest.raises(NotImplementedError):
        _ = await storage.save_artifact("s", "k", "c")


async def test_read_artifact_raises(storage: _ConcreteStorage):
    with pytest.raises(NotImplementedError):
        _ = await storage.read_artifact("artifact://s/a")


async def test_delete_artifacts_raises(storage: _ConcreteStorage):
    with pytest.raises(NotImplementedError):
        await storage.delete_artifacts("s")


async def test_abstract_pass_bodies_return_none(storage: _ConcreteStorage) -> None:
    from agent.types import SessionMetadataData as _SMD, ContextBudget

    session_data = _SMD(
        session_id="s1",
        state="idle",
        model_profile_id="openai:gpt-4o",
        agent_name="test",
        context_budget=ContextBudget(),
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )
    _ = await storage.save_session("s1", session_data)
    _ = await storage.load_session("s1")
    _ = await storage.delete_session("s1")
    _ = await storage.session_exists("s1")
    _ = await storage.list_sessions()
    from agent.core.session.context import SessionContext
    from agent.core.message.part import UserTextInput

    sc = SessionContext(session_id="s1", model_profile_id="openai:gpt-4o")
    _ = sc.build_user_message([UserTextInput(text="hi")])
    msg = sc.messages[-1]
    _ = await storage.save_message("s1", msg)
    _ = await storage.load_messages("s1")
    _ = await storage.delete_messages("s1")
