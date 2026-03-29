"""Task tool - 调用 SubAgent 执行任务。

持有 subagent_classes dict 引用（name → class），动态读取。
SubAgent 通过 Framework + EventLoop 执行，复用主 agent 的完整对话循环。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from loguru import logger

from agent.sub_agent import SubAgentBase

from ...core.tools.base import Tool

if TYPE_CHECKING:
    from agent.core.bus.message_bus import MessageBus
    from agent.core.llm.provider import AdapterRegistry
    from agent.core.session.manager import SessionManager
    from agent.types import LLMConfig


class TaskTool(Tool):
    """调用 SubAgent 执行子任务的工具。

    LLM 通过 agent_name 选择要调用的 subagent，通过 prompt 传递任务指令。
    SubAgent 通过 framework 临时注册，复用 EventLoop 完整路径。
    多个 task tool call 并行执行。
    """

    def __init__(
        self,
        subagent_classes: dict[str, type[SubAgentBase]],
        adapter_registry: AdapterRegistry,
        session_manager: SessionManager,
        message_bus: MessageBus,
        llm_profiles: dict[str, LLMConfig],
        max_depth: int = 3,
        current_depth: int = 0,
    ) -> None:
        self._classes: dict[str, type[SubAgentBase]] = subagent_classes
        self._adapter_registry: AdapterRegistry = adapter_registry
        self._session_manager: SessionManager = session_manager
        self._message_bus: MessageBus = message_bus
        self._llm_profiles: dict[str, LLMConfig] = llm_profiles
        self._max_depth: int = max_depth
        self._current_depth: int = current_depth

    @property
    @override
    def timeout(self) -> int | None:
        return None  # 无超时，由 max_iterations 控制

    @property
    @override
    def name(self) -> str:
        return "task"

    @property
    @override
    def description(self) -> str:
        if not self._classes:
            return "No sub-agents available."

        desc_lines = [
            "Delegate a task to a specialized sub-agent.",
            "Each sub-agent runs in an isolated session with its own tools and context.",
            "Use this when a task matches a sub-agent's description.",
            "Multiple task calls in a single response will execute in parallel.",
            "Only the sub-agents listed here are available:",
            "<available_agents>",
        ]

        for agent_name, cls in self._classes.items():
            agent_desc: str = cls.description()
            desc_lines.extend(
                [
                    "  <agent>",
                    f"    <name>{agent_name}</name>",
                    f"    <description>{agent_desc}</description>",
                    "  </agent>",
                ]
            )

        desc_lines.append("</available_agents>")
        return "\n".join(desc_lines)

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the sub-agent to invoke",
                },
                "prompt": {
                    "type": "string",
                    "description": "The task/instruction to send to the sub-agent",
                },
            },
            "required": ["agent_name", "prompt"],
        }

    @override
    async def execute(self, agent_name: str = "", prompt: str = "", **kwargs: object) -> object:
        if agent_name not in self._classes:
            available = ", ".join(self._classes.keys())
            return f"Error: Sub-agent '{agent_name}' not found. Available: {available or 'none'}"

        cls = self._classes[agent_name]
        parent_session_id = str(kwargs.get("__session_id__", ""))

        from agent.core.loop.sub_agent_runner import SubAgentRunner
        from agent.core.message.part import (
            SubAgentPart,
            SubAgentStateCompleted,
            SubAgentStateError,
            TimeInfo,
        )
        from agent.core.utils import now_ms

        runner = SubAgentRunner(
            adapter_registry=self._adapter_registry,
            session_manager=self._session_manager,
            message_bus=self._message_bus,
            llm_profiles=self._llm_profiles,
            max_depth=self._max_depth,
        )

        start_time = now_ms()
        try:
            result = await runner.run(
                subagent_cls=cls,
                prompt=prompt,
                parent_session_id=parent_session_id,
                current_depth=self._current_depth + 1,
            )
        except Exception as exc:
            logger.error("SubAgent '{}' execution failed: {}", agent_name, exc)
            return SubAgentPart(
                session_id=parent_session_id,
                message_id="",
                call_id="",
                agent_name=agent_name,
                depth=self._current_depth + 1,
                state=SubAgentStateError(
                    prompt=prompt,
                    error=str(exc),
                    time=TimeInfo(start=start_time, end=now_ms()),
                ),
            )

        if result.success:
            state = SubAgentStateCompleted(
                prompt=prompt,
                output=result.output,
                child_session_id=result.session_id,
                tokens=result.tokens,
                time=TimeInfo(start=start_time, end=now_ms()),
            )
        else:
            state = SubAgentStateError(
                prompt=prompt,
                error=result.error or "Unknown error",
                child_session_id=result.session_id,
                time=TimeInfo(start=start_time, end=now_ms()),
            )

        return SubAgentPart(
            session_id=parent_session_id,
            message_id="",
            call_id="",
            agent_name=agent_name,
            depth=self._current_depth + 1,
            state=state,
        )
