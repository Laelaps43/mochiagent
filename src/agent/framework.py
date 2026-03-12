"""Agent框架主类"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from loguru import logger

from .context import AgentContext
from .core.bus import MessageBus
from .core.loop import AgentEventLoop
from .core.session import SessionManager
from .core.storage import StorageProvider
from .core.llm import AdapterRegistry
from .core.runtime import AgentStrategyManager
from .core.utils import normalize_profile_id
from .types import LLMConfig

if TYPE_CHECKING:
    from .base_agent import BaseAgent


class AgentFramework:
    """Agent框架主类 - 管理所有Agent和核心组件"""

    def __init__(
        self,
        max_concurrent: int = 50,
        max_iterations: int = 100,
    ):
        """创建框架实例"""
        self.max_concurrent: int = max_concurrent
        self.max_iterations: int = max(1, max_iterations)
        self.bus: MessageBus = MessageBus(max_concurrent=max_concurrent)
        self.adapter_registry: AdapterRegistry = AdapterRegistry()

        # 延迟初始化（需要 Storage）
        self.session_manager: SessionManager | None = None
        self.event_loop: AgentEventLoop | None = None

        self._agents: dict[str, BaseAgent] = {}
        self._llm_profiles: dict[str, LLMConfig] = {}
        self._strategy_manager: AgentStrategyManager = AgentStrategyManager()
        self._initialized: bool = False
        self._started: bool = False

        logger.info(
            "AgentFramework created (max_concurrent={}, max_iterations={})",
            max_concurrent,
            self.max_iterations,
        )

    async def initialize(self, storage: StorageProvider) -> None:
        """
        初始化框架 - 配置 Storage 和核心组件

        Args:
            storage: Storage provider for session persistence
        """
        if self._initialized:
            raise RuntimeError("Framework already initialized")

        # 初始化 SessionManager
        self.session_manager = SessionManager(
            bus=self.bus,
            storage=storage,
        )

        # 初始化 EventLoop
        self.event_loop = AgentEventLoop(
            bus=self.bus,
            session_manager=self.session_manager,
            adapter_registry=self.adapter_registry,
            framework=self,
            max_iterations=self.max_iterations,
        )

        self._initialized = True
        logger.info("Framework initialized")

    async def register_agent(self, agent: BaseAgent) -> None:
        """
        注册 Agent 到框架并完成初始化。

        这个方法会：
        1. 创建 AgentContext 并绑定到 agent
        2. await agent.setup()，确保返回时 agent 已可用

        Args:
            agent: Agent instance to register

        Raises:
            RuntimeError: If framework not initialized
            ValueError: If agent is not BaseAgent instance or name conflicts
        """
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")

        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")

        normalized_allowed = self._normalize_allowed_profiles(agent.allowed_model_profiles)

        # Agent 必须明确指定允许的 profiles
        if normalized_allowed is None:
            raise ValueError(f"Agent '{agent.name}' must specify allowed_model_profiles explicitly")

        # 过滤出 agent 可用的 llm_profiles
        agent_llm_profiles = {
            pid: cfg for pid, cfg in self._llm_profiles.items() if pid in normalized_allowed
        }

        assert self.session_manager is not None
        ctx = AgentContext(
            session_manager=self.session_manager,
            message_bus=self.bus,
            strategy_manager=self._strategy_manager,
            agent_name=agent.name,
            llm_profiles=agent_llm_profiles,
        )
        agent.bind_context(ctx)
        self._agents[agent.name] = agent

        try:
            await self._setup_agent(agent)
        except Exception:
            _ = self._agents.pop(agent.name, None)
            raise

        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, agent_name: str) -> BaseAgent | None:
        return self._agents.get(agent_name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

    @property
    def strategy_manager(self) -> AgentStrategyManager:
        return self._strategy_manager

    def set_llm_configs(self, configs: list[LLMConfig]) -> None:
        """初始化设置 LLM 配置列表（profile_id = provider:model）。"""
        profiles: dict[str, LLMConfig] = {}
        for config in configs:
            profile_id = normalize_profile_id(f"{config.provider}:{config.model}")
            if config.adapter not in self.adapter_registry.list_adapters():
                available = ", ".join(sorted(self.adapter_registry.list_adapters())) or "<none>"
                raise ValueError(
                    f"Unknown adapter '{config.adapter}' for profile '{profile_id}'. Available adapters: {available}"
                )

            existing = profiles.get(profile_id)
            if existing is not None and existing.model_dump() != config.model_dump():
                raise ValueError(f"Conflicting llm config for profile_id '{profile_id}'")
            profiles[profile_id] = config

        self._llm_profiles.clear()
        self._llm_profiles.update(profiles)
        logger.info("Loaded {} llm profiles", len(self._llm_profiles))

    def _normalize_allowed_profiles(self, allowed_profiles: set[str] | None) -> set[str] | None:
        if allowed_profiles is None:
            return None
        normalized_allowed: set[str] = set()
        for item in allowed_profiles:
            normalized_allowed.add(normalize_profile_id(item))
        return normalized_allowed

    async def _setup_agent(self, agent: BaseAgent) -> None:
        """
        Setup agent.

        Args:
            agent: Agent to setup
        """
        try:
            await agent.setup()
            logger.info(f"Agent '{agent.name}' setup completed")
        except Exception as e:
            logger.error(f"Failed to setup agent '{agent.name}': {e}")
            raise

    def unregister_agent(self, agent_name: str) -> None:
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")

    async def start(self) -> None:
        """启动框架"""
        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")

        if self._started:
            logger.warning("Framework already started")
            return

        await self.bus.start()
        self._started = True
        logger.info("Framework started")

    async def stop(self) -> None:
        """停止框架"""
        if not self._started:
            return

        for agent in list(self._agents.values()):
            try:
                await agent.cleanup()
            except Exception as exc:
                logger.warning("Agent '{}' cleanup failed: {}", agent.name, exc)

        await self.bus.stop()
        self._started = False
        logger.info("Framework stopped")

    def is_running(self) -> bool:
        """检查框架是否正在运行"""
        return self._started

    def is_initialized(self) -> bool:
        """检查框架是否已初始化"""
        return self._initialized

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.stop()


DEFAULT_MAX_CONCURRENT = 50
DEFAULT_MAX_ITERATIONS = 100


class FrameworkRegistry:
    """管理框架单例及重建策略。"""

    def __init__(self) -> None:
        self.instance: AgentFramework | None = None

    @staticmethod
    def should_recreate_instance(
        current: AgentFramework,
        requested_max_concurrent: int | None,
        requested_max_iterations: int | None,
        resolved_max_concurrent: int,
        resolved_max_iterations: int,
    ) -> bool:
        if (
            requested_max_concurrent is not None
            and current.max_concurrent != resolved_max_concurrent
        ):
            return True
        if (
            requested_max_iterations is not None
            and current.max_iterations != resolved_max_iterations
        ):
            return True
        return False

    def get(
        self,
        max_concurrent: int | None = None,
        max_iterations: int | None = None,
    ) -> AgentFramework:
        resolved_max_concurrent = (
            max_concurrent if max_concurrent is not None else DEFAULT_MAX_CONCURRENT
        )
        resolved_max_iterations = max(
            1, max_iterations if max_iterations is not None else DEFAULT_MAX_ITERATIONS
        )

        if self.instance is None:
            self.instance = AgentFramework(
                max_concurrent=resolved_max_concurrent,
                max_iterations=resolved_max_iterations,
            )
            logger.info("Global framework instance created")
            return self.instance

        if not self.should_recreate_instance(
            current=self.instance,
            requested_max_concurrent=max_concurrent,
            requested_max_iterations=max_iterations,
            resolved_max_concurrent=resolved_max_concurrent,
            resolved_max_iterations=resolved_max_iterations,
        ):
            return self.instance

        if self.instance.is_initialized():
            logger.warning(
                "Framework already initialized, ignore new config (requested max_concurrent={}, max_iterations={})",
                resolved_max_concurrent,
                resolved_max_iterations,
            )
            return self.instance

        self.instance = AgentFramework(
            max_concurrent=resolved_max_concurrent,
            max_iterations=resolved_max_iterations,
        )
        logger.info("Global framework instance re-created with new config")
        return self.instance

    def reset(self) -> None:
        self.instance = None
        logger.warning("Global framework instance reset")


framework_registry = FrameworkRegistry()


def get_framework(
    max_concurrent: int | None = None,
    max_iterations: int | None = None,
) -> AgentFramework:
    """获取框架单例"""
    return framework_registry.get(
        max_concurrent=max_concurrent,
        max_iterations=max_iterations,
    )


def reset_framework() -> None:
    """重置全局框架实例（主要用于测试）"""
    framework_registry.reset()
