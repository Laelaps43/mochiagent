"""
Tool Executor - 工具执行器
"""

import asyncio
import json
from pathlib import Path
from jsonschema import ValidationError, validate as jsonschema_validate
from loguru import logger

from .policy import ToolPolicyConfig, ToolPolicyEngine
from .base import Tool
from .registry import ToolRegistry
from .security_guard import ToolSecurityConfig, ToolSecurityGuard
from agent.common.tools._utils import set_workspace_root
from typing import cast
from agent.config.tools import ToolRuntimeConfig
from agent.types import ToolCallPayload, ToolResult


class ToolExecutor:
    """
    工具执行器
    负责解析和执行工具调用
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: int = 30,
        policy: ToolPolicyEngine | None = None,
        policy_allow: set[str] | None = None,
        policy_deny: set[str] | None = None,
        workspace_root: str | Path | None = None,
        restrict_to_workspace: bool = True,
        security: ToolSecurityConfig | None = None,
        max_batch_concurrency: int | None = None,
    ):
        """
        初始化工具执行器

        Args:
            registry: 工具注册表
            default_timeout: 默认超时时间(秒),默认30秒
        """
        self.registry: ToolRegistry = registry
        self.default_timeout: int = default_timeout
        self.policy: ToolPolicyEngine = policy or ToolPolicyEngine(
            config=ToolPolicyConfig(
                allow=policy_allow,
                deny=policy_deny,
            )
        )
        self.security_guard: ToolSecurityGuard = ToolSecurityGuard(
            root=Path(workspace_root) if workspace_root else Path.cwd(),
            restrict=restrict_to_workspace,
            config=security or ToolSecurityConfig(),
        )
        if restrict_to_workspace:
            set_workspace_root(Path(workspace_root) if workspace_root else Path.cwd())
        concurrency = (
            max_batch_concurrency
            if max_batch_concurrency is not None
            else ToolRuntimeConfig().max_batch_concurrency
        )
        self._batch_semaphore: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        logger.info("ToolExecutor initialized (timeout={}s)", default_timeout)

    async def execute(self, tool_call: ToolCallPayload) -> ToolResult:
        """
        执行工具调用

        Args:
            tool_call: 工具调用对象

        Returns:
            ToolResult对象
        """
        tool_call_id = tool_call.id
        tool_name = tool_call.function.name
        arguments_str = tool_call.function.arguments or "{}"

        logger.info("Executing tool: {} (call_id={})", tool_name, tool_call_id)

        try:
            # 解析参数
            try:
                parsed: object = json.loads(arguments_str)  # pyright: ignore[reportAny]
                if not isinstance(parsed, dict):
                    return ToolResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        result=None,
                        error=f"Tool arguments must be a JSON object, got {type(parsed).__name__}",
                        success=False,
                    )
                arguments = cast(dict[str, object], parsed)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse tool arguments: {}", e)
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"Invalid JSON arguments: {e}",
                    success=False,
                )

            # 获取工具
            if not self.registry.has(tool_name):
                logger.error("Tool '{}' not found", tool_name)
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"Tool '{tool_name}' not found",
                    success=False,
                )

            tool = self.registry.get(tool_name)

            # 工具策略检查（deny > allow）
            decision = self.policy.evaluate(tool_name)
            if not decision.allowed:
                logger.warning(
                    "Tool '{}' blocked by policy: {} (call_id={})",
                    tool_name,
                    decision.reason,
                    tool_call_id,
                )
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"TOOL_POLICY_DENIED: {decision.reason}",
                    success=False,
                )

            # 验证参数 (使用 JSON Schema)
            try:
                self._validate_arguments(tool, arguments)
            except ValueError as e:
                logger.error("Tool '{}' parameter validation failed: {}", tool_name, e)
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"Parameter validation failed: {e}",
                    success=False,
                )

            # 统一安全检查入口（PathChecks + CommandChecks）
            security_decision = self.security_guard.validate_tool_call(tool, arguments)
            if not security_decision.allowed:
                logger.warning(
                    "Tool '{}' blocked by security guard: {}",
                    tool_name,
                    security_decision.reason,
                )
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"TOOL_SECURITY_DENIED: {security_decision.reason}",
                    success=False,
                )

            # 执行工具 (带超时控制)
            try:
                result = await asyncio.wait_for(
                    tool.execute(**arguments), timeout=self.default_timeout
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Tool '{}' execution timeout after {}s", tool_name, self.default_timeout
                )
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"Tool execution timeout after {self.default_timeout}s",
                    success=False,
                )

            logger.info("Tool {} executed successfully", tool_name)

            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result,
                success=True,
            )

        except Exception as e:
            logger.error("Error executing tool {}: {}", tool_name, e, exc_info=True)
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=None,
                error=str(e),
                success=False,
            )

    async def execute_batch(self, tool_calls: list[ToolCallPayload]) -> list[ToolResult]:
        """
        批量执行工具调用（并行执行）

        Args:
            tool_calls: 工具调用列表

        Returns:
            ToolResult列表（顺序与输入一致）
        """
        if not tool_calls:
            return []

        logger.info("Executing {} tool calls in parallel", len(tool_calls))

        async def _limited_execute(tc: ToolCallPayload) -> ToolResult:
            async with self._batch_semaphore:
                return await self.execute(tc)

        tasks = [_limited_execute(tool_call) for tool_call in tool_calls]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ToolResult] = []
        for i, raw_result in enumerate(raw_results):
            if isinstance(raw_result, Exception):
                tool_call = tool_calls[i]
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name

                logger.error(
                    "Tool {} (call_id={}) failed with exception: {}",
                    tool_name,
                    tool_call_id,
                    raw_result,
                    exc_info=raw_result,
                )

                results.append(
                    ToolResult(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        result=None,
                        error=str(raw_result),
                        success=False,
                    )
                )
            elif isinstance(raw_result, ToolResult):
                results.append(raw_result)

        success_count = sum(1 for r in results if r.success)
        logger.info("Batch execution completed: {}/{} succeeded", success_count, len(tool_calls))

        return results

    def _validate_arguments(self, tool: Tool, arguments: dict[str, object]) -> None:
        """
        验证工具参数是否符合 schema

        Args:
            tool: 工具实例
            arguments: 参数字典

        Raises:
            ValueError: 参数验证失败
        """
        schema: dict[str, object] = tool.parameters_schema
        try:
            jsonschema_validate(instance=arguments, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Invalid parameters: {e.message}")
