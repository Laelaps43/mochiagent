"""异步Agent框架 - 基于事件驱动的多Agent系统"""

from loguru import logger

from .core.bus import MessageBus
from .core.llm import LLMProvider, AdapterRegistry
from .core.loop import AgentEventLoop
from .core.runtime import StrategyKind
from .core.session import SessionContext, SessionManager, SessionStateMachine
from .core.storage import StorageProvider, MemoryStorage
from .core.tools import (
    Tool,
    ToolExecutor,
    ToolRegistry,
    ToolResultPostProcessor,
    ToolResultPostProcessConfig,
    ToolResultPostProcessorStrategy,
)
from .core.compression import (
    ContextCompactor,
    CompactionDecision,
    SummaryBuildResult,
    CompactorRunOptions,
)
from .core.message import UserInput, UserTextInput
from .types import (
    Event,
    EventType,
    LLMConfig,
    Message,
    MessageRole,
    SessionState,
    StreamChunk,
    ToolCallPayload,
    ToolDefinition,
    ToolResult,
)
from .framework import AgentFramework, get_framework
from .framework import reset_framework as _reset_framework
from .base_agent import BaseAgent
from .session import Session
from .config import (
    ToolPolicyConfig,
    ToolRuntimeConfig,
    ToolSecurityConfig,
    WorkspaceConfig,
)

_registered_agent_classes: list[type[BaseAgent]] = []


def agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Agent 装饰器 - 自动注册 Agent 类"""
    _registered_agent_classes.append(cls)
    return cls


def get_registered_agents() -> list[type[BaseAgent]]:
    return _registered_agent_classes.copy()


def reset_framework() -> None:
    """重置全局框架实例（主要用于测试）"""
    _reset_framework()
    _registered_agent_classes.clear()


async def setup(
    storage: StorageProvider | None = None,
    agents: list[BaseAgent] | None = None,
    llm_configs: list[LLMConfig] | None = None,
    max_concurrent: int = 50,
    max_iterations: int = 100,
) -> None:
    """
    初始化 Agent 系统并注册所有 Agent

    Args:
        storage: Storage provider（默认 MemoryStorage）
        agents: Agent 实例列表
        llm_configs: 初始化时注册的 LLM 配置列表（profile_id = provider:model）
        max_concurrent: 消息总线最大并发
        max_iterations: 单次对话最大迭代轮数（LLM turn）
    """
    framework = get_framework(
        max_concurrent=max_concurrent,
        max_iterations=max_iterations,
    )

    resolved_storage = storage
    if resolved_storage is None:
        resolved_storage = MemoryStorage()
        logger.warning(
            "Using default MemoryStorage. This backend is for development/testing only and may be incomplete for production persistence. Please provide a custom StorageProvider in production."
        )

    try:
        # 初始化框架
        await framework.initialize(resolved_storage)
        await framework.start()

        if llm_configs:
            framework.set_llm_configs(llm_configs)

        # 注册所有 Agent（会自动调用 agent.setup()）
        if agents:
            for agent_instance in agents:
                await framework.register_agent(agent_instance)
    except Exception:
        try:
            await framework.stop()
        except Exception as cleanup_exc:
            logger.warning("Failed to stop framework during setup cleanup: {}", cleanup_exc)
        reset_framework()
        raise


def get_agent(agent_name: str) -> BaseAgent | None:
    """获取已注册的 Agent"""
    framework = get_framework()
    if not framework.is_initialized():
        return None
    return framework.get_agent(agent_name)


def list_agents() -> list[str]:
    """列出所有已注册的 Agent 名称"""
    framework = get_framework()
    if not framework.is_initialized():
        return []
    return framework.list_agents()


def set_agent_strategy(
    kind: StrategyKind,
    agent_name: str,
    strategy: ContextCompactor | ToolResultPostProcessorStrategy,
    *,
    compaction_options: CompactorRunOptions | None = None,
) -> None:
    """设置某个 Agent 使用的指定策略实例。"""
    framework = get_framework()
    framework.strategy_manager.set(
        kind, agent_name, strategy, compaction_options=compaction_options
    )


async def shutdown() -> None:
    """关闭 Agent 系统"""
    framework = get_framework()
    await framework.stop()
    reset_framework()


__version__ = "0.2.0"

__all__ = [
    # High-level API (用户主要使用这些)
    "setup",
    "shutdown",
    "get_agent",
    "list_agents",
    "set_agent_strategy",
    # Framework (for advanced users)
    "AgentFramework",
    "StrategyKind",
    "CompactorRunOptions",
    "get_framework",
    "reset_framework",
    "BaseAgent",
    "Session",
    # Decorators & Registry
    "agent",
    "get_registered_agents",
    # Core - Bus
    "MessageBus",
    # Core - LLM
    "LLMProvider",
    "AdapterRegistry",
    # Core - Loop
    "AgentEventLoop",
    # Core - Session
    "SessionContext",
    "SessionManager",
    "SessionStateMachine",
    # Core - Storage
    "StorageProvider",
    "MemoryStorage",
    # Core - Tools
    "Tool",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResultPostProcessor",
    "ToolResultPostProcessConfig",
    "ToolResultPostProcessorStrategy",
    # Core - Compression
    "ContextCompactor",
    "CompactionDecision",
    "SummaryBuildResult",
    # Types
    "Event",
    "EventType",
    "LLMConfig",
    "Message",
    "MessageRole",
    "SessionState",
    "StreamChunk",
    "ToolCallPayload",
    "ToolDefinition",
    "ToolResult",
    # User Input Types
    "UserInput",
    "UserTextInput",
    # Public Config
    "ToolPolicyConfig",
    "ToolRuntimeConfig",
    "ToolSecurityConfig",
    "WorkspaceConfig",
]
