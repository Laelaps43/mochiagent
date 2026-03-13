"""System and MessageBus runtime config exposed to package users."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MessageBusConfig(BaseSettings):
    """MessageBus 运行时配置

    环境变量示例:
        MOCHI_BUS_QUEUE_TIMEOUT=2.0
        MOCHI_BUS_MAX_CONCURRENT=100
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        env_prefix="MOCHI_BUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    queue_timeout: float = Field(default=1.0, gt=0)
    max_concurrent: int = Field(default=50, ge=1)


class SystemConfig(BaseSettings):
    """系统级框架配置

    环境变量示例:
        MOCHI_SYS_UUID_PREFIX_LENGTH=12
        MOCHI_SYS_DEFAULT_SESSION_STATE=ready
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        env_prefix="MOCHI_SYS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uuid_prefix_length: int = Field(default=16, ge=4, le=32)
    default_session_state: Literal["idle"] = "idle"
