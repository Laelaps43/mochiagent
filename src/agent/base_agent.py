"""Agent基类"""

import json
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Union, List, Optional

from loguru import logger

from .config import ToolRuntimeConfig
from .core.mcp.manager import MCPManager
from .core.tools import Tool, ToolRegistry, ToolExecutor, ToolSecurityConfig
from .core.message import UserMessagePartInput
from .common.skill import Skill
from .common.tools import SkillTool
from .context import AgentContext
from .types import Event
from .session import Session

if TYPE_CHECKING:
    from .core.mcp.types import MCPServerSnapshot
    from .core.session import SessionContext


class BaseAgent(ABC):
    """
    Agent基类 - 所有Agent的基础类

    设计原则：
    - Agent 不直接依赖 Framework
    - 通过 AgentContext 访问外部依赖
    - 职责清晰：Agent 定义行为，Context 执行操作
    """

    def __init__(
        self,
        *,
        tools: ToolRuntimeConfig | None = None,
    ):
        # 运行上下文（由 Framework 注入）
        self._ctx: Optional[AgentContext] = None

        tool_runtime = tools or ToolRuntimeConfig()
        self.tool_runtime = tool_runtime

        # Agent 的内部组件
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            default_timeout=tool_runtime.timeout,
            policy_allow=tool_runtime.policy.allow,
            policy_deny=tool_runtime.policy.deny,
            workspace_root=tool_runtime.workspace.root,
            restrict_to_workspace=tool_runtime.workspace.restrict,
            security=ToolSecurityConfig(
                enforce_workspace=tool_runtime.security.enforce_workspace,
                enforce_command_guard=tool_runtime.security.enforce_command_guard,
                command_deny_tokens=tool_runtime.security.command_deny_tokens,
            ),
        )

        # Skills registered by this agent
        self._registered_skills: dict[str, Skill] = {}
        self._skill_tool: Optional[SkillTool] = None  # 缓存统一的 skill tool
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_manager: MCPManager | None = None

        logger.info(f"{self.__class__.__name__} initialized")

    def bind_context(self, ctx: AgentContext) -> None:
        """
        绑定运行上下文（由 Framework 调用）

        Args:
            ctx: Agent 运行上下文，封装了 session_manager 和 message_bus
        """
        self._ctx = ctx
        logger.debug(f"Agent {self.name} bound to context")

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent描述"""
        pass

    @abstractmethod
    async def setup(self) -> None:
        """Agent 初始化钩子（注册工具和 skills）"""
        pass

    def get_system_prompt(self, context: "SessionContext") -> str | None:
        """
        返回当前 agent 的专属 system prompt（可选覆盖）

        默认返回 None，表示当前 agent 不注入 system prompt。
        """
        return None

    @property
    def mcp_config_path(self) -> Path | None:
        """
        Agent MCP 配置文件路径（可选覆盖）。

        返回 None 表示该 agent 不启用 MCP。
        """
        return None

    def register_tool(self, tool: Tool) -> None:
        """
        注册工具到 agent

        Args:
            tool: Tool instance to register
        """
        self.tool_registry.register(tool)
        logger.info(f"Agent {self.name} registered tool: {tool.name}")

    async def register_mcp_tools(self, path: Path | None = None) -> None:
        """
        从 mcp.json 读取 mcpServers 并注册 MCP 工具。

        配置格式兼容主流 MCP 客户端:
        {
          "mcpServers": {
            "serverName": { "command": "...", "args": [...] }
          }
        }
        """
        config_path = path or self.mcp_config_path
        if config_path is None:
            return

        if not config_path.exists():
            logger.info("MCP config not found for agent '{}': {}", self.name, config_path)
            return

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read MCP config '{}': {}", config_path, exc)
            return

        mcp_servers = payload.get("mcpServers")
        if not isinstance(mcp_servers, dict) or not mcp_servers:
            logger.info("No mcpServers configured in {}", config_path)
            return

        manager = MCPManager(
            registry=self.tool_registry,
            default_timeout=self.tool_runtime.timeout,
        )
        self._mcp_manager = manager

        stack = AsyncExitStack()
        try:
            await stack.__aenter__()
            registered = await manager.connect_servers(
                mcp_servers=mcp_servers,
                stack=stack,
            )
            if registered <= 0:
                logger.warning(
                    "Agent '{}' MCP configured but no tools registered: {}",
                    self.name,
                    manager.snapshot(),
                )
                try:
                    await stack.aclose()
                except Exception as close_exc:
                    logger.warning(
                        "Agent '{}' MCP stack close after zero registration raised: {}",
                        self.name,
                        close_exc,
                    )
                return
            self._mcp_stack = stack.pop_all()
            logger.info(
                "Agent '{}' registered {} MCP tools from {}",
                self.name,
                registered,
                config_path,
            )
        except Exception as exc:
            try:
                await stack.aclose()
            except Exception as close_exc:
                logger.warning(
                    "Agent '{}' MCP stack close after error raised: {}",
                    self.name,
                    close_exc,
                )
            logger.error(
                "Failed to register MCP tools for agent '{}' from {}: {} | status={}",
                self.name,
                config_path,
                exc,
                manager.snapshot(),
            )

    def get_mcp_status(self) -> dict[str, "MCPServerSnapshot"]:
        """
        返回当前 Agent 维护的 MCP 服务状态快照。

        不暴露为框架接口，仅供 Agent 内部/调用方按需读取。
        """
        if not self._mcp_manager:
            return {}
        return self._mcp_manager.snapshot()

    @property
    @abstractmethod
    def skill_directory(self) -> Path | None:
        """
        Agent 的 skill 目录

        返回 None 表示该 agent 不使用 skills。

        Returns:
            Path to agent's skill directory, or None

        Example:
            >>> @property
            >>> def skill_directory(self) -> Path:
            ...     return Path(__file__).parent / "skills"
        """
        pass

    @property
    def allowed_model_profiles(self) -> set[str] | None:
        """
        Agent 可使用的 model profile 列表（格式：provider:model）。
        返回 None 表示不限制。
        """
        return None

    @property
    def default_model_profile(self) -> str | None:
        """
        Agent 默认使用的 model profile（格式：provider:model）。
        返回 None 表示必须显式提供 model_profile_id。
        """
        return None

    def register_skill(self, skill_name: str) -> None:
        """
        注册一个 skill 到当前 agent

        每次注册后会自动更新统一的 skill tool，无需手动 finalize。

        Args:
            skill_name: Skill 名称（对应 skills 目录下的文件夹名）

        Raises:
            ValueError: If skill cannot be loaded

        Example:
            >>> @property
            >>> def skill_directory(self) -> Path:
            ...     return Path(__file__).parent / "skills"
            ...
            >>> async def setup(self):
            ...     self.register_skill("data-analysis")
            ...     self.register_skill("sql-query")
        """
        if self.skill_directory is None:
            raise ValueError(f"Agent {self.name} has no skill_directory")

        # 使用 agent 自己的 skill 目录
        from .common.skill import SkillLoader

        loader = SkillLoader(self.skill_directory)
        skill = loader.load_skill(skill_name)

        if not skill:
            raise ValueError(f"Failed to load skill '{skill_name}' from {self.skill_directory}")

        # 存储已注册的 skill
        self._registered_skills[skill.name] = skill
        logger.info(
            f"Agent '{self.name}' registered skill '{skill.name}' from {self.skill_directory}"
        )

        # 自动更新统一的 skill tool
        self._update_skill_tool()

    def _update_skill_tool(self) -> None:
        """
        更新统一的 skill tool

        每次注册 skill 后自动调用，创建/更新包含所有 skills 的 SkillTool。
        如果之前有 skill tool，会先注销再创建新的。
        """
        # 如果之前有 skill tool，先从注册表移除
        if self._skill_tool is not None:
            self.tool_registry.unregister(self._skill_tool.name)

        # 创建新的 skill tool
        self._skill_tool = SkillTool(self._registered_skills)
        self.register_tool(self._skill_tool)

        skill_names = list(self._registered_skills.keys())
        logger.info(
            f"Agent '{self.name}' updated skill tool with "
            f"{len(self._registered_skills)} skills: {skill_names}"
        )

    async def push_message(
        self,
        session_id: str,
        message: Union[str, List[UserMessagePartInput]],
    ) -> None:
        """
        发送消息到会话

        Args:
            session_id: 会话 ID
            message: 消息内容（字符串或结构化消息）

        Raises:
            RuntimeError: If context not bound
        """
        if not self._ctx:
            raise RuntimeError(f"Agent {self.name} context not bound")

        await self._ctx.send_message(session_id, message)

    async def take_session(
        self,
        session_id: str,
        model_profile_id: Optional[str] = None,
    ) -> Session:
        """
        接管会话并返回 Session 对象

        Args:
            session_id: 会话 ID
            model_profile_id: Framework 注册的模型 profile ID（可选）

        Returns:
            Session 对象

        Raises:
            RuntimeError: If context not bound

        Example:
            >>> agent = get_agent("analytics_agent")
            >>> session = await agent.take_session(session_id, "default")
            >>> await session.push_message("用户消息")
        """
        if not self._ctx:
            raise RuntimeError(f"Agent {self.name} context not bound")

        resolved_model_profile_id = model_profile_id or self.default_model_profile
        if resolved_model_profile_id is None:
            raise ValueError("model_profile_id is required when taking a session")

        # 先获取或创建 session
        session = await self._ctx.get_session(
            self.name,
            session_id,
            model_profile_id=resolved_model_profile_id,
        )

        # 再切换 session 到当前 agent
        await self._ctx.switch_session_agent(session_id, self.name)

        return session

    async def handle_event(self, event: Event) -> None:
        """
        处理事件（可选覆盖）

        Args:
            event: Event object
        """
        pass

    async def cleanup(self) -> None:
        """
        清理资源（可选覆盖）
        """
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            finally:
                self._mcp_stack = None
        if self._mcp_manager:
            self._mcp_manager.mark_disconnected()
