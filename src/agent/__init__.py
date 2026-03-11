"""异步Agent框架 - 基于事件驱动的多Agent系统"""

from typing import Type, List, Optional, Mapping

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
from .core.compression import ContextCompactor, CompactionResult
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
from .framework import AgentFramework, get_framework, reset_framework
from .base_agent import BaseAgent
from .session import Session
from .config import (
    ToolPolicyConfig,
    ToolRuntimeConfig,
    ToolSecurityConfig,
    WorkspaceConfig,
)

_registered_agent_classes: List[Type[BaseAgent]] = []


def agent(cls: Type[BaseAgent]) -> Type[BaseAgent]:
    """Agent 装饰器 - 自动注册 Agent 类"""
    if not issubclass(cls, BaseAgent):
        raise TypeError(f"{cls.__name__} must inherit from BaseAgent")
    _registered_agent_classes.append(cls)
    return cls


def get_registered_agents() -> List[Type[BaseAgent]]:
    return _registered_agent_classes.copy()


async def setup(
    storage: Optional[StorageProvider] = None,
    agents: Optional[List[BaseAgent]] = None,
    llm_configs: Optional[List[LLMConfig]] = None,
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
            "Using default MemoryStorage. This backend is for development/testing only "
            "and may be incomplete for production persistence. "
            "Please provide a custom StorageProvider in production."
        )

    # 初始化框架
    await framework.initialize(resolved_storage)
    await framework.start()

    if llm_configs:
        framework.set_llm_configs(llm_configs)

    # 注册所有 Agent（会自动调用 agent.setup()）
    if agents:
        for agent_instance in agents:
            await framework.register_agent(agent_instance)


def get_agent(agent_name: str) -> Optional[BaseAgent]:
    """获取已注册的 Agent"""
    framework = get_framework()
    return framework.get_agent(agent_name)


def list_agents() -> List[str]:
    """列出所有已注册的 Agent 名称"""
    framework = get_framework()
    return framework.list_agents()


def register_strategy(kind: StrategyKind, name: str, factory) -> None:
    """注册某类策略工厂。"""
    framework = get_framework()
    framework.strategy_manager.register(kind, name, factory)


def list_strategies(kind: StrategyKind) -> List[str]:
    """列出某类已注册策略名。"""
    framework = get_framework()
    return framework.strategy_manager.list(kind)


def set_agent_strategy(
    kind: StrategyKind,
    agent_name: str,
    name: str,
    options: Optional[Mapping[str, object]] = None,
) -> None:
    """设置某个 Agent 使用的指定策略。"""
    framework = get_framework()
    framework.strategy_manager.set_agent(kind, agent_name, name, options)


async def shutdown() -> None:
    """关闭 Agent 系统"""
    framework = get_framework()
    await framework.stop()


__version__ = "0.2.0"

__all__ = [
    # High-level API (用户主要使用这些)
    "setup",
    "shutdown",
    "get_agent",
    "list_agents",
    "register_strategy",
    "list_strategies",
    "set_agent_strategy",
    # Framework (for advanced users)
    "AgentFramework",
    "StrategyKind",
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
    "CompactionResult",
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
