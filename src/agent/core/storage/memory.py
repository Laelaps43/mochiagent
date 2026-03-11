"""
Memory Storage - 内存存储实现
"""

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from .provider import (
    ArtifactMetadata,
    ArtifactReadResult,
    SessionMetadataData,
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

    def __init__(self, artifact_root: str | Path | None = None):
        self._sessions: dict[str, SessionMetadataData] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}
        self._artifact_root = (
            Path(artifact_root) if artifact_root else (Path.cwd() / ".agent" / "artifacts")
        )
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        logger.info("MemoryStorage initialized")
        logger.warning(
            "MemoryStorage is a lightweight in-process backend and is not a complete "
            "production-grade persistence solution. Data is process-local and may be lost "
            "after restart."
        )

    async def save_session(self, session_id: str, session_data: SessionMetadataData) -> None:
        self._sessions[session_id] = session_data
        logger.debug(f"Saved session metadata to memory: {session_id}")

    async def load_session(self, session_id: str) -> SessionMetadataData | None:
        data = self._sessions.get(session_id)
        if data:
            logger.debug(f"Loaded session metadata from memory: {session_id}")
        return data

    async def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session from memory: {session_id}")
        if session_id in self._messages:
            del self._messages[session_id]
            logger.debug(f"Deleted messages from memory: {session_id}")
        await self.delete_artifacts(session_id)

    async def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    async def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())

    async def save_message(self, session_id: str, message_data: dict[str, Any]) -> None:
        if session_id not in self._messages:
            self._messages[session_id] = []
        self._messages[session_id].append(message_data)
        logger.debug(
            f"Saved message to memory: {session_id}, "
            f"total messages: {len(self._messages[session_id])}"
        )

    async def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        messages = self._messages.get(session_id, [])
        logger.debug(f"Loaded {len(messages)} messages from memory: {session_id}")
        return messages

    async def delete_messages(self, session_id: str) -> None:
        if session_id in self._messages:
            del self._messages[session_id]
            logger.debug(f"Deleted all messages from memory: {session_id}")

    async def save_artifact(
        self,
        session_id: str,
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactMetadata:
        session_dir = self._artifact_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        artifact_id = f"{kind}_{int(time.time())}_{uuid4().hex[:8]}"
        content_path = session_dir / f"{artifact_id}.txt"
        meta_path = session_dir / f"{artifact_id}.json"

        await asyncio.to_thread(content_path.write_text, content, "utf-8")
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
        await asyncio.to_thread(meta_path.write_text, meta_json, "utf-8")
        return artifact_meta

    async def read_artifact(
        self,
        artifact_ref: str,
        offset: int = 0,
        limit: int = 50000,
    ) -> ArtifactReadResult:
        session_id, artifact_id = self._parse_artifact_ref(artifact_ref)
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

    async def delete_artifacts(self, session_id: str) -> None:
        session_dir = self._artifact_root / session_id
        if session_dir.exists():
            await asyncio.to_thread(shutil.rmtree, session_dir, True)

    @staticmethod
    def _parse_artifact_ref(artifact_ref: str) -> tuple[str, str]:
        prefix = "artifact://"
        if not artifact_ref.startswith(prefix):
            raise ValueError(f"Invalid artifact_ref: {artifact_ref}")
        body = artifact_ref[len(prefix) :]
        parts = body.split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid artifact_ref: {artifact_ref}")
        return parts[0], parts[1]
