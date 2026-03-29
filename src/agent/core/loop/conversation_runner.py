"""ConversationRunner - 对话循环 + 工具执行的共享逻辑。

AgentEventLoop（事件驱动入口）和 SubAgentRunner（subagent 入口）共用此模块。
不持有 framework 引用，通过方法参数接收 agent。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import cast

from loguru import logger

from agent.base_agent import BaseAgent
from agent.core.compression import CompactionPayload
from agent.core.loop.llm_turn_handler import LLMTurnHandler
from agent.core.loop.turn_result import LLMTurnResult
from agent.core.message import SubAgentPart, ToolPart
from agent.core.tools.types import ToolCallPayload, ToolResult
from agent.types import Event, EventType, SessionState, TokenUsage


class ConversationRunner:
    """对话循环 + 工具执行。不持有 framework，agent 通过方法参数传入。"""

    def __init__(
        self,
        *,
        turn_handler: LLMTurnHandler,
        emit_event: Callable[[Event], Awaitable[None]],
        max_iterations: int = 100,
    ) -> None:
        self.turn_handler: LLMTurnHandler = turn_handler
        self.emit_event: Callable[[Event], Awaitable[None]] = emit_event
        self.max_iterations: int = max(1, max_iterations)

    async def conversation_loop(self, session_id: str, agent: BaseAgent) -> LLMTurnResult | None:
        """运行对话循环直到 LLM 返回 stop 或超过最大迭代次数。"""
        session_manager = agent.context.session_manager
        context = await session_manager.get_session(session_id)
        await session_manager.update_state(session_id, SessionState.PROCESSING)

        logger.info("Starting conversation loop for session {}", session_id)

        entered_error_path = False
        try:
            iteration_count = 0
            all_compaction_events: list[CompactionPayload] = []

            while iteration_count < self.max_iterations:
                iteration_count += 1
                result = await self.turn_handler.run(session_id, agent)

                all_compaction_events.extend(result.context_compaction_events)

                if result.tool_calls and result.finish_reason == "tool_calls":
                    tool_calls = result.tool_calls
                    logger.info("LLM requested {} tool calls", len(tool_calls))

                    await session_manager.update_state(session_id, SessionState.WAITING_TOOL)
                    _ = await self.execute_tools(session_id, agent, tool_calls)
                    await session_manager.update_state(session_id, SessionState.PROCESSING)
                    continue

                logger.info(
                    "Conversation completed for session {} (finish_reason={})",
                    session_id,
                    result.finish_reason,
                )

                await session_manager.finish_assistant_message(
                    session_id=session_id,
                    tokens=result.tokens,
                    finish=result.finish_reason or "stop",
                )

                await self.emit_event(
                    Event(
                        type=EventType.MESSAGE_DONE,
                        session_id=session_id,
                        data={
                            "message_id": result.message_id,
                            "tokens": result.tokens,
                            "context_budget": result.context_budget,
                            "context_compaction": result.context_compaction,
                            "context_compaction_events": all_compaction_events,
                            "finish": result.finish_reason or "stop",
                        },
                    )
                )
                return result

            # 超过最大迭代次数
            logger.error(
                "Max iterations exceeded for session {}: limit={}",
                session_id,
                self.max_iterations,
            )
            last_message_id = (
                context.current_message.message_id if context.current_message else None
            )
            if context.current_message:
                await session_manager.finish_assistant_message(
                    session_id=session_id,
                    tokens=TokenUsage(),
                    finish="max_iterations_exceeded",
                )
            await self.emit_event(
                Event(
                    type=EventType.LLM_ERROR,
                    session_id=session_id,
                    data={
                        "error": f"Conversation stopped: maximum iterations exceeded ({self.max_iterations})",
                        "code": "MAX_ITERATIONS_EXCEEDED",
                        "max_iterations": self.max_iterations,
                        "iterations": iteration_count,
                    },
                )
            )
            await self.emit_event(
                Event(
                    type=EventType.MESSAGE_DONE,
                    session_id=session_id,
                    data={
                        "message_id": last_message_id,
                        "tokens": TokenUsage(),
                        "context_budget": context.context_budget,
                        "finish": "max_iterations_exceeded",
                    },
                )
            )
            return None

        except Exception:
            entered_error_path = True
            await session_manager.update_state(session_id, SessionState.ERROR)
            try:
                err_ctx = await session_manager.get_session(session_id)
                if err_ctx.current_message:
                    await session_manager.finish_assistant_message(
                        session_id=session_id,
                        tokens=TokenUsage(),
                        finish="error",
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to finalize assistant message on error for session {}: {}",
                    session_id,
                    exc,
                )
            raise

        finally:
            if not entered_error_path:
                try:
                    await session_manager.update_state(session_id, SessionState.IDLE)
                except Exception as exc:
                    logger.warning("Failed to reset session {} state to IDLE: {}", session_id, exc)

    async def execute_tools(
        self, session_id: str, agent: BaseAgent, tool_calls: list[ToolCallPayload]
    ) -> list[ToolResult]:
        """执行工具调用，处理 SubAgentPart 替换。"""
        session_manager = agent.context.session_manager
        context = await session_manager.get_session(session_id)
        message_id = context.current_message.message_id if context.current_message else None

        if not message_id:
            logger.warning("No current message for tool calls")
            return []

        tool_parts_map: dict[str, ToolPart] = {}
        for tool_call in tool_calls:
            tool_part = ToolPart.create_running(session_id, message_id, tool_call)
            tool_parts_map[tool_call.id] = tool_part
            context.add_part_to_current(tool_part)

            await self.emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=tool_part.model_dump(),
                )
            )

        results = await agent.tool_executor.execute_batch(
            tool_calls, context={"__session_id__": session_id}
        )
        tool_args_by_call_id: dict[str, dict[str, object]] = {}
        for tool_call in tool_calls:
            raw_args = tool_call.function.arguments or "{}"
            try:
                loaded = cast(object, json.loads(raw_args))
                parsed: dict[str, object] = (
                    cast(dict[str, object], loaded)
                    if isinstance(loaded, dict)
                    else {"raw": raw_args}
                )
            except Exception:
                parsed = {"raw": raw_args}
            tool_args_by_call_id[tool_call.id] = parsed

        for result in results:
            call_id = result.tool_call_id
            if call_id not in tool_parts_map:
                continue

            original_part = tool_parts_map[call_id]

            # Tool 返回 SubAgentPart → 直接替换 ToolPart
            if isinstance(result.result, SubAgentPart):
                subagent_part: SubAgentPart = result.result
                subagent_part = SubAgentPart(
                    id=original_part.id,
                    session_id=session_id,
                    message_id=message_id,
                    call_id=call_id,
                    agent_name=subagent_part.agent_name,
                    depth=subagent_part.depth,
                    state=subagent_part.state,
                    metadata=subagent_part.metadata,
                )

                if context.current_message is not None:
                    for i, part in enumerate(context.current_message.parts):
                        if isinstance(part, ToolPart) and part.call_id == call_id:
                            context.current_message.parts[i] = subagent_part
                            break

                await self.emit_event(
                    Event(
                        type=EventType.PART_CREATED,
                        session_id=session_id,
                        data=subagent_part.model_dump(),
                    )
                )
                continue

            # 正常 ToolPart 更新逻辑
            if result.success:
                processed_result = await agent.context.strategy_manager.run_postprocess(
                    agent_name=context.agent_name,
                    session_id=session_id,
                    tool_result=result,
                    tool_arguments=tool_args_by_call_id.get(call_id, {}),
                    storage=session_manager.storage,
                )
                updated_part = original_part.update_to_completed(processed_result)
            else:
                updated_part = original_part.update_to_error(result)

            if context.current_message is not None:
                for i, part in enumerate(context.current_message.parts):
                    if isinstance(part, ToolPart) and part.call_id == call_id:
                        context.current_message.parts[i] = updated_part
                        break

            await self.emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=updated_part.model_dump(),
                )
            )

        return results
