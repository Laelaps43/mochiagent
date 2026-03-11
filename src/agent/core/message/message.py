"""
Message Container - Message = Info + Parts
"""

from dataclasses import dataclass
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from loguru import logger

from agent.types import Message as ChatMessage
from .info import MessageInfo, UserMessageInfo, AssistantMessageInfo
from .part import Part


@dataclass
class SerializedMessageData:
    """序列化的消息数据（用于存储和传输）"""
    info: UserMessageInfo | AssistantMessageInfo
    parts: list[Part]


class Message(BaseModel):
    """
    消息容器：Message = Info + Parts[]

    一个消息由元数据（Info）和内容片段（Parts）组成
    """

    info: MessageInfo
    parts: List[Part] = Field(default_factory=list)

    def add_part(self, part: Part) -> None:
        """添加 Part 到消息"""
        self.parts.append(part)

    def to_dict(self) -> SerializedMessageData:
        return SerializedMessageData(
            info=self.info,
            parts=list(self.parts),
        )

    @classmethod
    def from_dict(cls, data: SerializedMessageData) -> "Message":
        return cls(info=data.info, parts=data.parts)

    @property
    def message_id(self) -> str:
        return self.info.id

    @property
    def session_id(self) -> str:
        return self.info.session_id

    @property
    def role(self) -> str:
        return self.info.role

    def to_llm_messages(self) -> List[ChatMessage]:
        text_contents = []
        tool_calls = []
        tool_results = []
        for part in self.parts:
            llm_data = part.to_llm_format()
            if llm_data is None:
                continue
            if llm_data.get("type") == "text":
                text_contents.append(llm_data["content"])
            elif llm_data.get("type") == "tool":
                if "tool_call" in llm_data:
                    tool_calls.append(llm_data["tool_call"])
                if "tool_result" in llm_data:
                    tool_results.append(llm_data["tool_result"])

        # 如果没有任何内容（空的 assistant message），返回空列表
        if not text_contents and not tool_calls:
            return []

        messages: List[ChatMessage] = []
        main_msg: Dict[str, Any] = {
            "role": self.info.role,
            "content": "".join(text_contents),
        }
        if self.info.role == "assistant" and tool_calls:
            main_msg["tool_calls"] = tool_calls
        messages.append(ChatMessage.model_validate(main_msg))
        messages.extend(ChatMessage.model_validate(tool_result) for tool_result in tool_results)
        return messages
