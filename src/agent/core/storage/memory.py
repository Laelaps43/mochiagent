"""
Memory Storage - 内存存储实现
"""

import asyncio
import json
import re
import shutil
import time
from pathlib import Path
from typing import override
from uuid import uuid4

from loguru import logger

from agent.core.message import Message
from agent.types import SessionMetadataData

from .provider import (
    ArtifactMetadata,
    ArtifactReadResult,
    StorageProvider,
)


class MemoryStorage(StorageProvider):
    """
    纯内存存储（默认）

    特点：
    - 快速
    - 服务重启后数据丢失
    - 适合开发、测试和无需持久化的场景

    数据结构：
    - _sessions: 会话元数据
    - _messages: 会话消息列表
    """

    SESSION_COUNT_WARNING_THRESHOLD: int = 500
    MESSAGE_COUNT_WARNING_THRESHOLD: int = 2000

    def __init__(self, artifact_root: str | Path | None = None):
        self._sessions: dict[str, SessionMetadataData] = {}
        self._messages: dict[str, list[Message]] = {}
        self._artifact_root: Path = (
            Path(artifact_root) if artifact_root else (Path.cwd() / ".agent" / "artifacts")
        )
        # 延迟到首次写入 artifact 时创建目录，避免 __init__ 中阻塞调用
        logger.info("MemoryStorage initialized")
        logger.warning(
            "MemoryStorage is a lightweight in-process backend and is not a complete production-grade persistence solution. Data is process-local and may be lost after restart."
        )

    _SAFE_ID_RE: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\-]+$")

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Validate session_id contains only safe characters for filesystem paths."""
        if not MemoryStorage._SAFE_ID_RE.match(session_id):
            raise ValueError(
                f"Invalid session_id for artifact storage: {session_id!r}. "
                + "Only alphanumeric, underscore and hyphen allowed."
            )

    @staticmethod
    def _validate_artifact_id(artifact_id: str) -> None:
        """Validate artifact_id contains only safe characters for filesystem paths."""
        if not MemoryStorage._SAFE_ID_RE.match(artifact_id):
            raise ValueError(
                f"Invalid artifact_id: {artifact_id!r}. "
                + "Only alphanumeric, underscore and hyphen allowed."
            )

    @override
    async def save_session(self, session_id: str, session_data: SessionMetadataData) -> None:
        self._sessions[session_id] = session_data
        count = len(self._sessions)
        if count == self.SESSION_COUNT_WARNING_THRESHOLD:
            logger.warning(
                "MemoryStorage holds {} sessions — consider switching to a persistent StorageProvider",
                count,
            )
        logger.debug("Saved session metadata to memory: {}", session_id)

    @override
    async def load_session(self, session_id: str) -> SessionMetadataData | None:
        data = self._sessions.get(session_id)
        if data:
            logger.debug("Loaded session metadata from memory: {}", session_id)
        return data

    @override
    async def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("Deleted session from memory: {}", session_id)
        if session_id in self._messages:
            del self._messages[session_id]
            logger.debug("Deleted messages from memory: {}", session_id)
        await self.delete_artifacts(session_id)

    @override
    async def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    @override
    async def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    @override
    async def save_message(self, session_id: str, message: Message) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        msgs = self._messages[session_id]
        if msgs and msgs[-1].message_id == message.message_id:
            msgs[-1] = message
        else:
            msgs.append(message)
        msg_count = len(self._messages[session_id])
        if msg_count == self.MESSAGE_COUNT_WARNING_THRESHOLD:
            logger.warning(
                "Session {} has {} messages in MemoryStorage — consider enabling context compaction",
                session_id,
                msg_count,
            )
        logger.debug(
            "Saved message to memory: {}, total messages: {}",
            session_id,
            msg_count,
        )

    @override
    async def load_messages(
        self, session_id: str, *, from_message_id: str | None = None
    ) -> list[Message]:
        messages = self._messages.get(session_id, [])
        if from_message_id is not None:
            for idx, msg in enumerate(messages):
                if msg.message_id == from_message_id:
                    messages = messages[idx:]
                    break
        logger.debug("Loaded {} messages from memory: {}", len(messages), session_id)
        return list(messages)

    @override
    async def delete_messages(self, session_id: str) -> None:
        if session_id in self._messages:
            del self._messages[session_id]
            logger.debug("Deleted all messages from memory: {}", session_id)

    @override
    async def save_artifact(
        self,
        session_id: str,
        kind: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> ArtifactMetadata:
        self._validate_session_id(session_id)
        session_dir = self._artifact_root / session_id
        await asyncio.to_thread(lambda: session_dir.mkdir(parents=True, exist_ok=True))

        artifact_id = f"{kind}_{int(time.time())}_{uuid4().hex[:8]}"
        self._validate_artifact_id(artifact_id)
        content_path = session_dir / f"{artifact_id}.txt"
        meta_path = session_dir / f"{artifact_id}.json"

        _ = await asyncio.to_thread(content_path.write_text, content, "utf-8")
        artifact_ref = f"artifact://{session_id}/{artifact_id}"
        artifact_meta = ArtifactMetadata(
            artifact_ref=artifact_ref,
            artifact_id=artifact_id,
            session_id=session_id,
            kind=kind,
            size=len(content),
            path=str(content_path),
            created_at_ms=int(time.time() * 1000),
            metadata=metadata or {},
        )

        meta_json = json.dumps(artifact_meta.model_dump(), ensure_ascii=False, indent=2)
        _ = await asyncio.to_thread(meta_path.write_text, meta_json, "utf-8")
        return artifact_meta

    @override
    async def read_artifact(
        self,
        artifact_ref: str,
        offset: int = 0,
        limit: int = 50000,
    ) -> ArtifactReadResult:
        session_id, artifact_id = self.parse_artifact_ref(artifact_ref)
        self._validate_artifact_id(artifact_id)
        content_path = self._artifact_root / session_id / f"{artifact_id}.txt"

        if not content_path.exists():
            return ArtifactReadResult(
                error=f"Artifact not found: {artifact_ref}",
                artifact_ref=artifact_ref,
            )

        text = await asyncio.to_thread(content_path.read_text, "utf-8")
        safe_offset = max(0, offset)
        safe_limit = max(1, limit)
        chunk = text[safe_offset : safe_offset + safe_limit]
        next_offset = safe_offset + len(chunk)
        eof = next_offset >= len(text)

        return ArtifactReadResult(
            success=True,
            artifact_ref=artifact_ref,
            path=str(content_path),
            content=chunk,
            offset=safe_offset,
            limit=safe_limit,
            next_offset=next_offset,
            eof=eof,
            size=len(text),
        )

    @override
    async def delete_artifacts(self, session_id: str) -> None:
        self._validate_session_id(session_id)
        # Note: artifact_id validation is performed on individual operations
        session_dir = self._artifact_root / session_id
        if session_dir.exists():
            await asyncio.to_thread(shutil.rmtree, session_dir, True)

    @staticmethod
    def parse_artifact_ref(artifact_ref: str) -> tuple[str, str]:
        prefix = "artifact://"
        if not artifact_ref.startswith(prefix):
            raise ValueError(f"Invalid artifact_ref: {artifact_ref}")
        body = artifact_ref[len(prefix) :]
        parts = body.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid artifact_ref: {artifact_ref}")
        return parts[0], parts[1]
