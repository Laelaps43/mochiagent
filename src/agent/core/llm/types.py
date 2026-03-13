"""LLM-related types."""

from pydantic import BaseModel, Field

from agent.core.tools.types import ToolCallPayload


class ProviderUsage(BaseModel):
    """LLM 供应商返回的 token 用量（统一命名，由 adapter 负责映射）"""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


class LLMStreamChunk(BaseModel):
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCallPayload] = Field(default_factory=list)
    finish_reason: str = ""
    usage: ProviderUsage | None = None
