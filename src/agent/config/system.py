"""System and MessageBus runtime config exposed to package users."""

from __future__ import annotations

from typing import ClassVar

from pydantic_settings import BaseSettings, SettingsConfigDict


class MessageBusConfig(BaseSettings):
    """MessageBus 运行时配置

    环境变量示例:
        MOCHI_QUEUE_TIMEOUT=2.0
        MOCHI_MAX_CONCURRENT=100
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        env_prefix="MOCHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    queue_timeout: float = 1.0
    max_concurrent: int = 50


class SystemConfig(BaseSettings):
    """系统级框架配置

    环境变量示例:
        MOCHI_UUID_PREFIX_LENGTH=12
        MOCHI_DEFAULT_SESSION_STATE=ready
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        env_prefix="MOCHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uuid_prefix_length: int = 16
    default_session_state: str = "idle"
