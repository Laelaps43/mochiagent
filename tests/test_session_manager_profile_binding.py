import pytest

from agent.core.bus import MessageBus
from agent.core.session import SessionManager
from agent.core.storage import MemoryStorage


@pytest.mark.asyncio
async def test_get_or_create_session_refreshes_profile_for_cached_session(tmp_path):
    storage = MemoryStorage(artifact_root=tmp_path / "artifacts")
    manager = SessionManager(bus=MessageBus(), storage=storage)

    await manager.create_session(session_id="sess_cached", model_profile_id="profile-a")

    context = await manager.get_or_create_session(
        session_id="sess_cached",
        model_profile_id="profile-b",
    )

    assert context.model_profile_id == "profile-b"

    persisted = await storage.load_session("sess_cached")
    assert persisted is not None
    assert persisted["model_profile_id"] == "profile-b"
    assert "llm_config" not in persisted


@pytest.mark.asyncio
async def test_get_or_create_session_requires_profile_id(tmp_path):
    storage = MemoryStorage(artifact_root=tmp_path / "artifacts")
    manager = SessionManager(bus=MessageBus(), storage=storage)

    with pytest.raises(ValueError, match="model_profile_id is required"):
        await manager.get_or_create_session(session_id="sess_new", model_profile_id="")
