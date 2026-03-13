"""Tool runtime config exposed to package users."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ToolPolicyConfig(BaseModel):
    """工具策略配置 (不支持环境变量)"""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    allow: set[str] | None = None
    deny: set[str] | None = None

    def normalized(self) -> "ToolPolicyConfig":
        return ToolPolicyConfig(
            allow={x.lower() for x in (self.allow or set())},
            deny={x.lower() for x in (self.deny or set())},
        )

    @classmethod
    def from_csv(
        cls,
        *,
        allow_csv: str | None = None,
        deny_csv: str | None = None,
    ) -> "ToolPolicyConfig":
        def _parse(raw: str | None) -> set[str]:
            if not raw:
                return set()
            return {item.strip().lower() for item in raw.split(",") if item and item.strip()}

        return cls(
            allow=_parse(allow_csv),
            deny=_parse(deny_csv),
        )


class WorkspaceConfig(BaseModel):
    """工作空间配置 (不支持环境变量)"""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    root: Path = Field(default_factory=Path.cwd)
    restrict: bool = True


class ToolSecurityConfig(BaseModel):
    """工具安全配置 (不支持环境变量)"""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    enforce_workspace: bool = True
    enforce_command_guard: bool = True
    command_deny_tokens: set[str] = Field(
        default_factory=lambda: {"`", "$(", "\n", "\r", ";", "&&", "||", "|", ">", ">>", "<"}
    )


class ToolRuntimeConfig(BaseSettings):
    """工具运行时配置

    环境变量示例:
        MOCHI_TOOL_TIMEOUT=60
        MOCHI_TOOL_MAX_BATCH_CONCURRENCY=20
        MOCHI_TOOL_EXEC_MAX_OUTPUT_CHARS=50000
        MOCHI_TOOL_WEB_FETCH_MAX_CHARS=30000
        MOCHI_TOOL_WEB_SEARCH_API_KEY=your-key-here
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        frozen=True,
        env_prefix="MOCHI_TOOL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    timeout: int = 30
    max_batch_concurrency: int = 10
    policy: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    security: ToolSecurityConfig = Field(default_factory=ToolSecurityConfig)
    exec_max_output_chars: int = 20000
    web_fetch_max_chars: int = 20000
    web_search_api_key: SecretStr = SecretStr("")
