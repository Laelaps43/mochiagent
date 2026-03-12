"""
Tool Executor - 工具执行器
"""

import asyncio
import json
from pathlib import Path
from typing import cast

from loguru import logger

from .policy import ToolPolicyConfig, ToolPolicyEngine
from .base import Tool
from .registry import ToolRegistry
from .security_guard import ToolSecurityConfig, ToolSecurityGuard
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
        logger.info(f"ToolExecutor initialized (timeout={default_timeout}s)")

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

        logger.info(f"Executing tool: {tool_name} (call_id={tool_call_id})")

        try:
            # 解析参数
            try:
                arguments = cast(dict[str, object], json.loads(arguments_str))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments: {e}")
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"Invalid JSON arguments: {e}",
                    success=False,
                )

            # 获取工具
            if not self.registry.has(tool_name):
                logger.error(f"Tool '{tool_name}' not found")
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
                    f"Tool '{tool_name}' blocked by policy: {decision.reason} (call_id={tool_call_id})"
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
                logger.error(f"Tool '{tool_name}' parameter validation failed: {e}")
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
                    f"Tool '{tool_name}' blocked by security guard: {security_decision.reason}"
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
                logger.error(f"Tool '{tool_name}' execution timeout after {self.default_timeout}s")
                return ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=None,
                    error=f"Tool execution timeout after {self.default_timeout}s",
                    success=False,
                )

            logger.info(f"Tool {tool_name} executed successfully")

            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
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

        logger.info(f"Executing {len(tool_calls)} tool calls in parallel")

        tasks = [self.execute(tool_call) for tool_call in tool_calls]

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[ToolResult] = []
        for i, raw_result in enumerate(raw_results):
            if isinstance(raw_result, Exception):
                tool_call = tool_calls[i]
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name

                logger.error(
                    f"Tool {tool_name} (call_id={tool_call_id}) failed with exception: {raw_result}",
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
            else:
                results.append(cast(ToolResult, raw_result))

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Batch execution completed: {success_count}/{len(tool_calls)} succeeded")

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
        try:
            from jsonschema import validate, ValidationError
        except ImportError:
            # jsonschema 未安装,跳过验证
            logger.warning("jsonschema not installed, skipping parameter validation")
            return

        schema: dict[str, object] = tool.parameters_schema
        try:
            validate(instance=arguments, schema=schema)
        except ValidationError as e:
            raise ValueError(f"Invalid parameters: {e.message}")
