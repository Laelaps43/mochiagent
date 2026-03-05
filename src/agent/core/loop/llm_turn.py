"""LLM turn execution helpers for AgentEventLoop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from loguru import logger

from agent.core.message import ReasoningPart, TextPart
from agent.core.prompt import inject_system_prompt
from agent.core.session.context_budget_utils import to_non_negative_int
from agent.types import ContextBudgetSource, Event, EventType

if TYPE_CHECKING:
    from .event_loop import AgentEventLoop


def extract_turn_tokens_from_usage(
    usage: dict | None,
) -> tuple[dict[str, int], ContextBudgetSource]:
    """Normalize provider usage payload to framework token fields."""
    if not usage:
        return {"input": 0, "output": 0, "reasoning": 0}, "estimated"

    input_tokens = to_non_negative_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tokens = to_non_negative_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    reasoning_tokens = to_non_negative_int(usage.get("reasoning_tokens", 0))
    if reasoning_tokens == 0:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning_tokens = to_non_negative_int(details.get("reasoning_tokens", 0))

    return {
        "input": input_tokens,
        "output": output_tokens,
        "reasoning": reasoning_tokens,
    }, "provider"


async def run_llm_turn(loop: AgentEventLoop, session_id: str) -> dict:
    """Execute one streaming LLM turn and emit partial parts."""
    context = await loop.session_manager.get_session(session_id)

    if not context.model_profile_id:
        raise ValueError(
            f"Session {session_id} has no model_profile_id. "
            "Please take_session with a valid model_profile_id first."
        )

    llm_config = loop.framework.resolve_llm_config_for_agent(
        context.agent_name,
        context.model_profile_id,
    )
    llm = loop.adapter_registry.get(llm_config)

    last_user_msg = context.messages[-1]
    assistant_msg = context.build_assistant_message(
        parent_id=last_user_msg.message_id,
        provider_id=llm_config.provider,
        model_id=llm_config.model,
    )
    message_id = assistant_msg.message_id

    reasoning_buffer = ""
    reasoning_start_time = None
    text_buffer = ""
    accumulated_tool_calls = []
    finish_reason = None
    total_tokens = {"input": 0, "output": 0, "reasoning": 0}
    total_cost = 0.0
    provider_usage: dict | None = None

    agent = loop.framework.get_agent(context.agent_name)
    tools = agent.tool_registry.get_definitions() if agent else []
    system_prompt = agent.get_system_prompt(context) if agent else None
    llm_messages = inject_system_prompt(context.get_llm_messages(), system_prompt)

    logger.debug("Calling LLM for session {}: messages={}", session_id, len(llm_messages))

    async for chunk in llm.stream_chat(
        messages=llm_messages,
        tools=tools,
    ):
        if "thinking" in chunk and chunk["thinking"]:
            reasoning_buffer += chunk["thinking"]
            if reasoning_start_time is None:
                reasoning_start_time = int(time.time() * 1000)

        if "content" in chunk and chunk["content"]:
            if reasoning_buffer:
                reasoning_part = ReasoningPart.create_fast(
                    session_id=session_id,
                    message_id=message_id,
                    text=reasoning_buffer,
                    start=reasoning_start_time or int(time.time() * 1000),
                    end=int(time.time() * 1000),
                )
                context.add_part_to_current(reasoning_part)
                await loop._emit_event(
                    Event(
                        type=EventType.PART_CREATED,
                        session_id=session_id,
                        data=reasoning_part.to_event_payload(),
                    )
                )
                reasoning_buffer = ""

            text_buffer += chunk["content"]
            text_part = TextPart.create_fast(
                session_id=session_id,
                message_id=message_id,
                text=chunk["content"],
            )
            context.add_part_to_current(text_part)
            await loop._emit_event(
                Event(
                    type=EventType.PART_CREATED,
                    session_id=session_id,
                    data=text_part.to_event_payload(),
                )
            )

        if "tool_calls" in chunk:
            accumulated_tool_calls = chunk["tool_calls"]

        if "finish_reason" in chunk:
            finish_reason = chunk["finish_reason"]
        if "usage" in chunk and isinstance(chunk["usage"], dict):
            provider_usage = chunk["usage"]

    total_tokens, source = extract_turn_tokens_from_usage(provider_usage)
    context_budget = context.update_context_budget(
        total_tokens=llm_config.context_window_tokens,
        input_tokens=total_tokens.get("input", 0),
        output_tokens=total_tokens.get("output", 0),
        reasoning_tokens=total_tokens.get("reasoning", 0),
        source=source,
    )
    await loop._persist_session_metadata(session_id)

    if not accumulated_tool_calls:
        await loop.session_manager.finish_assistant_message(
            session_id=session_id,
            cost=total_cost,
            tokens=total_tokens,
            finish=finish_reason or "stop",
        )

    return {
        "content": text_buffer,
        "tool_calls": accumulated_tool_calls,
        "finish_reason": finish_reason,
        "cost": total_cost,
        "tokens": total_tokens,
        "context_budget": context_budget,
        "message_id": message_id,
    }
