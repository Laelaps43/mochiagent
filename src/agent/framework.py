"""Agent框架主类"""

from __future__ import annotations

from typing import Optional, Dict, TYPE_CHECKING

from loguru import logger

from .context import AgentContext
from .core.bus import MessageBus
from .core.loop import AgentEventLoop
from .core.session import SessionManager
from .core.storage import StorageProvider
from .core.llm import ProviderRegistry

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
        self.max_concurrent = max_concurrent
        self.max_iterations = max(1, max_iterations)
        self.bus = MessageBus(max_concurrent=max_concurrent)
        self.provider_registry = ProviderRegistry()

        # 延迟初始化（需要 Storage）
        self.session_manager: Optional[SessionManager] = None
        self.event_loop: Optional[AgentEventLoop] = None

        self._agents: Dict[str, BaseAgent] = {}
        self._initialized = False
        self._started = False

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
            provider_registry=self.provider_registry,
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
        from .base_agent import BaseAgent

        if not self._initialized:
            raise RuntimeError("Framework not initialized. Call initialize() first.")

        if not isinstance(agent, BaseAgent):
            raise ValueError(f"Agent must be instance of BaseAgent, got {type(agent)}")

        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")

        ctx = AgentContext(
            session_manager=self.session_manager,
            message_bus=self.bus,
        )
        agent.bind_context(ctx)
        self._agents[agent.name] = agent

        try:
            await self._setup_agent(agent)
        except Exception:
            self._agents.pop(agent.name, None)
            raise

        logger.info(f"Registered agent: {agent.name}")

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        return self._agents.get(agent_name)

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())

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

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


DEFAULT_MAX_CONCURRENT = 50
DEFAULT_MAX_ITERATIONS = 100


class FrameworkRegistry:
    """管理框架单例及重建策略。"""

    def __init__(self) -> None:
        self.instance: Optional[AgentFramework] = None

    @staticmethod
    def resolve_requested_values(
        max_concurrent: int | None,
        max_iterations: int | None,
    ) -> tuple[int, int]:
        resolved_max_concurrent = (
            DEFAULT_MAX_CONCURRENT if max_concurrent is None else max_concurrent
        )
        resolved_max_iterations = max(
            1,
            DEFAULT_MAX_ITERATIONS if max_iterations is None else max_iterations,
        )
        return resolved_max_concurrent, resolved_max_iterations

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
        resolved_max_concurrent, resolved_max_iterations = self.resolve_requested_values(
            max_concurrent,
            max_iterations,
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
                "Framework already initialized, ignore new config "
                "(requested max_concurrent={}, max_iterations={})",
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
