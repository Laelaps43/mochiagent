"""SubAgent Runner - 直接调用 LLM 执行 SubAgent。

不注册到 framework，不走 push_message 事件驱动。
复用 ConversationRunner，直接传 agent 参数，不需要 shim。
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

from agent.context import AgentContext
from agent.core.bus.message_bus import MessageBus
from agent.core.llm.provider import AdapterRegistry
from agent.core.loop.conversation_runner import ConversationRunner
from agent.core.loop.llm_turn_handler import LLMTurnHandler
from agent.core.session.manager import SessionManager
from agent.core.utils import gen_id, normalize_profile_id
from agent.sub_agent import SubAgentBase
from agent.types import Event, EventType, LLMConfig, TokenUsage


class SubAgentResult(BaseModel):
    """SubAgent 执行结果。"""

    success: bool
    output: str
    error: str | None = None
    session_id: str
    tokens: TokenUsage = Field(default_factory=TokenUsage)


class SubAgentRunner:
    """直接调用 LLM 执行 SubAgent。

    不注册到 framework，不需要 shim。
    复用 ConversationRunner，直接传 agent 参数。
    """

    def __init__(
        self,
        *,
        adapter_registry: AdapterRegistry,
        session_manager: SessionManager,
        message_bus: MessageBus,
        llm_profiles: dict[str, LLMConfig],
        max_iterations: int = 50,
        max_depth: int = 3,
    ) -> None:
        self.adapter_registry: AdapterRegistry = adapter_registry
        self.session_manager: SessionManager = session_manager
        self.message_bus: MessageBus = message_bus
        self.llm_profiles: dict[str, LLMConfig] = llm_profiles
        self.max_iterations: int = max(1, max_iterations)
        self.max_depth: int = max(1, max_depth)

    async def run(
        self,
        subagent_cls: type[SubAgentBase],
        prompt: str,
        parent_session_id: str,
        current_depth: int = 1,
    ) -> SubAgentResult:
        if current_depth > self.max_depth:
            return SubAgentResult(
                success=False,
                output="",
                error=f"SubAgent recursion depth exceeded: {current_depth} > {self.max_depth}",
                session_id="",
            )

        subagent_name = subagent_cls.name()
        child_session_id = gen_id("sub_")

        # 解析 model profile
        raw_profile = subagent_cls.model_profile_id()
        profile_id = normalize_profile_id(raw_profile) if raw_profile else ""
        if not profile_id or profile_id not in self.llm_profiles:
            fallback_id = next(iter(self.llm_profiles), None)
            if fallback_id is None:
                return SubAgentResult(
                    success=False,
                    output="",
                    error="No LLM profiles available for subagent execution.",
                    session_id=child_session_id,
                )
            if profile_id:
                logger.warning(
                    "SubAgent '{}' requested profile '{}' not found, falling back to '{}'",
                    subagent_name,
                    profile_id,
                    fallback_id,
                )
            profile_id = fallback_id

        # 发射 SUBAGENT_INVOKED 事件
        await self.message_bus.publish(
            Event(
                type=EventType.SUBAGENT_INVOKED,
                session_id=parent_session_id,
                data={
                    "parent_session_id": parent_session_id,
                    "child_session_id": child_session_id,
                    "subagent_name": subagent_name,
                    "prompt": prompt,
                    "depth": current_depth,
                },
            )
        )

        try:
            result = await self._run_conversation(
                subagent_cls=subagent_cls,
                session_id=child_session_id,
                parent_session_id=parent_session_id,
                model_profile_id=profile_id,
                prompt=prompt,
            )
        except Exception as exc:
            logger.error("SubAgent '{}' failed: {}", subagent_name, exc)
            result = SubAgentResult(
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
                session_id=child_session_id,
            )

        # 发射 SUBAGENT_COMPLETED 事件
        await self.message_bus.publish(
            Event(
                type=EventType.SUBAGENT_COMPLETED,
                session_id=parent_session_id,
                data={
                    "parent_session_id": parent_session_id,
                    "child_session_id": child_session_id,
                    "subagent_name": subagent_name,
                    "success": result.success,
                    "output": result.output[:2000],
                    "error": result.error,
                    "tokens": result.tokens.model_dump(),
                    "depth": current_depth,
                },
            )
        )

        return result

    async def _run_conversation(
        self,
        *,
        subagent_cls: type[SubAgentBase],
        session_id: str,
        parent_session_id: str,
        model_profile_id: str,
        prompt: str,
    ) -> SubAgentResult:
        """实例化 subagent，直接调用 ConversationRunner。"""
        from agent.config import ToolRuntimeConfig
        from agent.core.message import UserTextInput
        from agent.core.runtime import AgentStrategyManager

        # 实例化 subagent（应用 tool_policy）
        policy = subagent_cls.tool_policy()
        if policy is not None:
            agent = subagent_cls(tools=ToolRuntimeConfig(policy=policy))
        else:
            agent = subagent_cls()

        await agent.setup()

        # 绑定 AgentContext（subagent 不注册到 framework，手动绑定）
        agent_profiles = {
            pid: cfg
            for pid, cfg in self.llm_profiles.items()
            if pid in {normalize_profile_id(p) for p in agent.allowed_model_profiles}
        }
        if not agent_profiles:
            agent_profiles = dict(self.llm_profiles)

        ctx = AgentContext(
            session_manager=self.session_manager,
            message_bus=self.message_bus,
            strategy_manager=AgentStrategyManager(),
            agent_name=agent.name(),
            llm_profiles=agent_profiles,
            adapter_registry=self.adapter_registry,
        )
        agent.bind_context(ctx)

        try:
            # 创建 session
            child_ctx = await self.session_manager.get_or_create_session(
                session_id=session_id,
                model_profile_id=model_profile_id,
                agent_name=agent.name(),
            )
            child_ctx.parent_session_id = parent_session_id
            await self.session_manager.save_session_metadata(session_id)

            # 添加 user message
            _ = await self.session_manager.add_user_message(
                session_id, [UserTextInput(text=prompt)]
            )

            # 创建 ConversationRunner（不需要 framework，直接传 agent）
            async def emit_event(event: Event) -> None:
                if event.session_id:
                    await self.session_manager.emit_to_session_listeners(event.session_id, event)

            turn_handler = LLMTurnHandler(emit_event=emit_event)
            runner = ConversationRunner(
                turn_handler=turn_handler,
                emit_event=emit_event,
                max_iterations=self.max_iterations,
            )

            # 直接跑对话循环，传 agent
            turn_result = await runner.conversation_loop(session_id, agent)

            if turn_result is not None:
                return SubAgentResult(
                    success=True,
                    output=turn_result.content,
                    session_id=session_id,
                    tokens=turn_result.tokens,
                )
            else:
                return SubAgentResult(
                    success=False,
                    output="",
                    error=f"Max iterations exceeded ({self.max_iterations})",
                    session_id=session_id,
                )

        finally:
            await agent.cleanup()
