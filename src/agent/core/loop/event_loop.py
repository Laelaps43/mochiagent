"""
Agent Event Loop - Agent事件循环
结合事件驱动和Loop循环的优势：
- 外层：事件驱动，支持多人并发
- 内层：Loop循环，自动处理多轮对话
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from loguru import logger

from agent.core.bus import MessageBus
from agent.core.llm import AdapterRegistry
from agent.core.llm.errors import LLMProviderError
from agent.core.prompt import inject_system_prompt
from agent.core.session import SessionManager
from agent.core.message import (
    ReasoningPart,
    TextPart,
    ToolPart,
)
from agent.core.tools import ToolResultPostProcessor
from agent.types import (
    Event,
    EventType,
    SessionState,
)

if TYPE_CHECKING:
    from agent.framework import AgentFramework


class AgentEventLoop:
    """
    Agent事件循环

    架构特点：
    1. 外层事件驱动 - 支持多session并发，自动排队
    2. 内层Loop循环 - 简化逻辑，自动处理tool_calls
    3. 并发控制 - 通过MessageBus的Semaphore控制最大并发数
    """

    def __init__(
        self,
        bus: MessageBus,
        session_manager: SessionManager,
        adapter_registry: AdapterRegistry,
        framework: AgentFramework,
        max_iterations: int = 100,
    ):
        self.bus = bus
        self.session_manager = session_manager
        self.adapter_registry = adapter_registry
        self.framework = framework
        self.max_iterations = max(1, max_iterations)
        self.tool_result_postprocessor = ToolResultPostProcessor()

        # 订阅事件
        self.bus.subscribe(EventType.MESSAGE_RECEIVED, self._handle_user_message)

        logger.info(
            "AgentEventLoop initialized (max_iterations={})",
            self.max_iterations,
        )

    async def _emit_event(self, event: Event) -> None:
        """
        统一的事件发射方法

        1. 先发给会话级监听者（SSE 等临时连接）
        2. 再发给系统级监听者（日志、监控等）

        Args:
            event: 事件对象
        """
        # 发给会话级监听者
        if event.session_id:
            await self.session_manager.emit_to_session_listeners(event.session_id, event)

    async def _emit_error_and_done(
        self,
        *,
        session_id: str,
        error_message: str,
        code: str | None = None,
        hint: str | None = None,
    ) -> None:
        """错误收敛：先发 LLM_ERROR，再发 MESSAGE_DONE，保证前端/终端本轮结束。"""
        error_data = {"error": error_message}
        if code:
            error_data["code"] = code
        if hint:
            error_data["hint"] = hint

        await self._emit_event(
            Event(
                type=EventType.LLM_ERROR,
                session_id=session_id,
                data=error_data,
            )
        )

        message_id = None
        try:
            context = await self.session_manager.get_session(session_id)
            if context.current_message:
                message_id = context.current_message.message_id
        except Exception:
            pass

        await self._emit_event(
            Event(
                type=EventType.MESSAGE_DONE,
                session_id=session_id,
                data={
                    "message_id": message_id,
                    "cost": 0.0,
                    "tokens": {},
                    "finish": "error",
                },
            )
        )

    async def _handle_user_message(self, event: Event) -> None:
        """
        处理用户消息 (事件驱动入口)

        注意：用户消息必须已通过 context.build_user_message() 构建并保存

        Args:
            event: MESSAGE_RECEIVED 事件
        """
        session_id = event.session_id
        logger.info(f"Handling user message for session {session_id}")

        try:
            # 获取 Session
            context = await self.session_manager.get_session(session_id)

            # 验证最后一条消息是用户消息
            if not context.messages or context.messages[-1].role != "user":
                raise ValueError(
                    f"No user message found in context for session {session_id}. "
                    f"Please call context.build_user_message() first."
                )

            # 进入对话循环（只发送 AI 生成的内容）
            await self._conversation_loop(session_id)

        except ValueError as e:
            # Session 不存在或用户消息不存在
            error_message = f"{type(e).__name__}: {e}"
            logger.error("Error handling user message (invalid state): {}", error_message)
            await self._emit_error_and_done(
                session_id=session_id,
                error_message=error_message,
                code="INVALID_STATE",
            )

        except Exception as e:
            error_message, code, hint = self._resolve_error_payload(e)
            logger.exception(
                "Error handling user message for session {}: {}",
                session_id,
                error_message,
            )
            await self._emit_error_and_done(
                session_id=session_id,
                error_message=error_message,
                code=code,
                hint=hint,
            )

    async def _conversation_loop(self, session_id: str) -> None:
        """
        对话循环 (内部Loop)

        自动处理多轮LLM调用和工具执行，直到完成

        流程:
        1. 调用LLM
        2. 如果有tool_calls，执行工具
        3. 将工具结果添加到历史
        4. 回到步骤1 (自动循环)
        5. 如果没有tool_calls，退出循环

        Args:
            session_id: 会话ID
        """
        context = await self.session_manager.get_session(session_id)

        # 标记为busy
        await self.session_manager.update_state(session_id, SessionState.PROCESSING)

        logger.info(f"Starting conversation loop for session {session_id}")

        try:
            iteration_count = 0
            while iteration_count < self.max_iterations:
                iteration_count += 1
                # 1. 调用LLM
                result = await self._call_llm_turn(session_id)

                # 2. 如果有工具调用
                if result.get("tool_calls") and result.get("finish_reason") == "tool_calls":
                    tool_calls = result["tool_calls"]
                    logger.info(f"LLM requested {len(tool_calls)} tool calls")

                    # 2.1 转换到 WAITING_TOOL 状态
                    await self.session_manager.update_state(session_id, SessionState.WAITING_TOOL)

                    # 2.2 执行工具（ToolPart 已在 _execute_tools 中添加到 current_message）
                    await self._execute_tools(session_id, tool_calls)

                    # 2.3 工具执行完成后，完成当前消息
                    await self.session_manager.finish_assistant_message(
                        session_id=session_id,
                        cost=result.get("cost", 0.0),
                        tokens=result.get("tokens", {}),
                        finish=result.get("finish_reason", "tool_calls"),
                    )

                    # 2.4 转回 PROCESSING 状态，继续循环
                    await self.session_manager.update_state(session_id, SessionState.PROCESSING)

                    # 2.5 继续循环 (自动下一轮)
                    continue

                # 3. 没有工具调用，对话完成
                logger.info(
                    f"Conversation completed for session {session_id} (finish_reason={result.get('finish_reason')})"
                )

                # 发送最终的 MESSAGE_DONE 事件
                await self._emit_event(
                    Event(
                        type=EventType.MESSAGE_DONE,
                        session_id=session_id,
                        data={
                            "message_id": result.get("message_id"),
                            "cost": result.get("cost", 0.0),
                            "tokens": result.get("tokens", {}),
                            "finish": result.get("finish_reason", "stop"),
                        },
                    )
                )

                return

            logger.error(
                "Max iterations exceeded for session {}: limit={}",
                session_id,
                self.max_iterations,
            )
            await self._emit_event(
                Event(
                    type=EventType.LLM_ERROR,
                    session_id=session_id,
                    data={
                        "error": (
                            "Conversation stopped: maximum iterations exceeded "
                            f"({self.max_iterations})"
                        ),
                        "code": "MAX_ITERATIONS_EXCEEDED",
                        "max_iterations": self.max_iterations,
                        "iterations": iteration_count,
                    },
                )
            )
            await self._emit_event(
                Event(
                    type=EventType.MESSAGE_DONE,
                    session_id=session_id,
                    data={
                        "message_id": (
                            context.current_message.message_id if context.current_message else None
                        ),
                        "cost": 0.0,
                        "tokens": {},
                        "finish": "max_iterations_exceeded",
                    },
                )
            )

        except Exception:
            await self.session_manager.update_state(session_id, SessionState.ERROR)
            raise

        finally:
            # 标记为idle
            await self.session_manager.update_state(session_id, SessionState.IDLE)

    async def _call_llm_turn(self, session_id: str) -> dict:
        """
        执行一次LLM调用 (单个turn)

        核心功能：生成 Parts 并实时发送

        Args:
            session_id: 会话ID

        Returns:
            dict包含:
            - content: str
            - tool_calls: list (可能为空)
            - finish_reason: str
        """
        context = await self.session_manager.get_session(session_id)

        if not context.model_profile_id:
            raise ValueError(
                f"Session {session_id} has no model_profile_id. "
                "Please take_session with a valid model_profile_id first."
            )

        llm_config = self.framework.resolve_llm_config_for_agent(
            context.agent_name,
            context.model_profile_id,
        )

        # 注意：不在这里设置 STREAMING 状态
        # 状态已经在 _conversation_loop 中设置为 PROCESSING
        # 流式输出期间保持 PROCESSING 状态

        # 获取LLM Provider (使用注入的 registry)
        llm = self.adapter_registry.get(llm_config)

        # 构建 AssistantMessage
        last_user_msg = context.messages[-1]
        assistant_msg = context.build_assistant_message(
            parent_id=last_user_msg.message_id,
            provider_id=llm_config.provider,
            model_id=llm_config.model,
        )
        message_id = assistant_msg.message_id

        # 累积变量
        reasoning_buffer = ""
        reasoning_start_time = None
        text_buffer = ""
        accumulated_tool_calls = []
        finish_reason = None
        total_tokens = {"input": 0, "output": 0, "reasoning": 0}
        total_cost = 0.0
        chunk_count = 0

        try:
            # 流式调用LLM（将 Part 格式转换为 LLM API 格式）
            # 从 Agent 获取工具定义
            agent = self.framework.get_agent(context.agent_name)
            tools = agent.tool_registry.get_definitions() if agent else []
            system_prompt = agent.get_system_prompt(context) if agent else None
            llm_messages = inject_system_prompt(
                context.get_llm_messages(),
                system_prompt,
            )

            logger.debug(f"Calling LLM for session {session_id}: messages={len(llm_messages)}")

            async for chunk in llm.stream_chat(
                messages=llm_messages,
                tools=tools,
            ):
                chunk_count += 1
                # 1. 处理 thinking（如果 LLM 支持）
                if "thinking" in chunk and chunk["thinking"]:
                    thinking_content = chunk["thinking"]
                    reasoning_buffer += thinking_content

                    if reasoning_start_time is None:
                        reasoning_start_time = int(time.time() * 1000)

                # 2. 处理 content
                if "content" in chunk and chunk["content"]:
                    # 如果有累积的 reasoning，先发送 ReasoningPart
                    if reasoning_buffer:
                        reasoning_part = ReasoningPart.create_fast(
                            session_id=session_id,
                            message_id=message_id,
                            text=reasoning_buffer,
                            start=reasoning_start_time or int(time.time() * 1000),
                            end=int(time.time() * 1000),
                        )
                        context.add_part_to_current(reasoning_part)

                        # 发送 PART_CREATED 事件
                        await self._emit_event(
                            Event(
                                type=EventType.PART_CREATED,
                                session_id=session_id,
                                data=reasoning_part.to_event_payload(),
                            )
                        )

                        reasoning_buffer = ""

                    # 累积文本内容（流式发送）
                    text_buffer += chunk["content"]

                    # 创建并发送 TextPart（流式）
                    text_part = TextPart.create_fast(
                        session_id=session_id,
                        message_id=message_id,
                        text=chunk["content"],
                    )
                    context.add_part_to_current(text_part)

                    # 发送 PART_CREATED 事件
                    await self._emit_event(
                        Event(
                            type=EventType.PART_CREATED,
                            session_id=session_id,
                            data=text_part.to_event_payload(),
                        )
                    )

                # 3. 处理 tool_calls
                if "tool_calls" in chunk:
                    accumulated_tool_calls = chunk["tool_calls"]

                # 4. 处理结束
                if "finish_reason" in chunk:
                    finish_reason = chunk["finish_reason"]

            # 如果没有工具调用，完成消息
            # 如果有工具调用，消息会在工具执行后完成
            if not accumulated_tool_calls:
                await self.session_manager.finish_assistant_message(
                    session_id=session_id,
                    cost=total_cost, tokens=total_tokens, finish=finish_reason or "stop"
                )

            return {
                "content": text_buffer,
                "tool_calls": accumulated_tool_calls,
                "finish_reason": finish_reason,
                "cost": total_cost,
                "tokens": total_tokens,
                "message_id": message_id,
            }

        except Exception:
            raise

    @staticmethod
    def _extract_provider_error(exc: Exception) -> LLMProviderError | None:
        current: BaseException | None = exc
        while current:
            if isinstance(current, LLMProviderError):
                return current
            current = current.__cause__
        return None

    @classmethod
    def _resolve_error_payload(
        cls,
        exc: Exception,
    ) -> tuple[str, str | None, str | None]:
        provider_error = cls._extract_provider_error(exc)
        if provider_error:
            return provider_error.message, provider_error.code, provider_error.hint
        return f"{type(exc).__name__}: {exc}", None, None

    async def _execute_tools(self, session_id: str, tool_calls: list) -> list:
        """
        执行工具调用，生成 ToolParts

        流程：
        1. 创建 running 状态的 ToolPart
        2. 并行执行工具
        3. 更新为 completed/error 状态

        Args:
            session_id: 会话ID
            tool_calls: 工具调用列表

        Returns:
            ToolResult列表
        """
        context = await self.session_manager.get_session(session_id)
        message_id = context.current_message.message_id if context.current_message else None

        if not message_id:
            logger.warning("No current message for tool calls")
            return []

        # 获取对应的 Agent 和其工具执行器
        agent = self.framework.get_agent(context.agent_name)
        if not agent:
            logger.error(f"Agent '{context.agent_name}' not found")
            return []

        tool_parts_map = {}
        for tool_call in tool_calls:
            tool_part = ToolPart.create_running(session_id, message_id, tool_call)
            tool_parts_map[tool_call["id"]] = tool_part
            context.add_part_to_current(tool_part)

            await self._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=tool_part.to_event_payload(),
                )
            )

        # 使用 Agent 的工具执行器
        results = await agent.tool_executor.execute_batch(tool_calls)
        tool_args_by_call_id: dict[str, dict] = {}
        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            call_id = tool_call.get("id", "")
            raw_args = function_info.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args)
            except Exception:
                parsed_args = {"raw": raw_args}
            tool_args_by_call_id[call_id] = parsed_args

        for result in results:
            call_id = result.tool_call_id
            if call_id not in tool_parts_map:
                continue

            original_part = tool_parts_map[call_id]

            if result.success:
                processed_result = await self.tool_result_postprocessor.process(
                    session_id=session_id,
                    tool_result=result,
                    tool_arguments=tool_args_by_call_id.get(call_id, {}),
                    storage=self.session_manager.storage,
                )
                updated_part = original_part.update_to_completed(processed_result)
            else:
                updated_part = original_part.update_to_error(result)

            for i, part in enumerate(context.current_message.parts):
                if isinstance(part, ToolPart) and part.call_id == call_id:
                    context.current_message.parts[i] = updated_part
                    break

            await self._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=updated_part.to_event_payload(),
                )
            )

        return results
