from pathlib import Path

import pytest

from agent.base_agent import BaseAgent
from agent.core.storage import MemoryStorage
from agent.framework import AgentFramework
from agent.types import LLMConfig


class _ProfileAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "profile_agent"

    @property
    def description(self) -> str:
        return "agent with model profile restrictions"

    @property
    def skill_directory(self) -> Path | None:
        return None

    @property
    def allowed_model_profiles(self) -> set[str] | None:
        return {"openai:glm-4.7"}

    @property
    def default_model_profile(self) -> str | None:
        return "openai:glm-4.7"

    async def setup(self) -> None:
        return None


@pytest.mark.asyncio
async def test_take_session_uses_default_model_profile(tmp_path):
    framework = AgentFramework()
    await framework.initialize(MemoryStorage(artifact_root=tmp_path / "artifacts"))

    framework.set_llm_configs(
        [
            LLMConfig(
                adapter="openai_compatible",
                provider="openai",
                model="glm-4.7",
                api_key="profile-key-123",
                base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            )
        ]
    )
    agent = _ProfileAgent()
    await framework.register_agent(agent)

    session = await agent.take_session("sess_profile_default")
    assert session.model_profile_id == "openai:glm-4.7"

    context = await framework.session_manager.get_session("sess_profile_default")
    assert context.model_profile_id == "openai:glm-4.7"
    assert not hasattr(context.metadata, "llm_config")


@pytest.mark.asyncio
async def test_framework_resolve_rejects_disallowed_model_profile(tmp_path):
    framework = AgentFramework()
    await framework.initialize(MemoryStorage(artifact_root=tmp_path / "artifacts"))

    framework.set_llm_configs(
        [
            LLMConfig(
                adapter="openai_compatible",
                provider="openai",
                model="glm-4.7",
                api_key="profile-key-123",
            ),
            LLMConfig(
                adapter="openai_compatible",
                provider="openai",
                model="gpt-4.1",
                api_key="profile-key-456",
            ),
        ]
    )

    agent = _ProfileAgent()
    await framework.register_agent(agent)

    with pytest.raises(ValueError, match="not available"):
        agent.context.resolve_llm_config_for_agent("profile_agent", "openai:gpt-4.1")
