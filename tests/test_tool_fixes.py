"""
Test Tool Timeout and Validation
测试工具超时和参数验证功能
"""

import asyncio
from agent.core.tools import Tool, ToolRegistry, ToolExecutor
from typing import Any, Dict


class SlowTool(Tool):
    """慢速工具 - 用于测试超时"""

    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "A tool that takes a long time to execute"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sleep_seconds": {"type": "integer", "description": "How many seconds to sleep"}
            },
            "required": ["sleep_seconds"],
        }

    async def execute(self, sleep_seconds: int) -> Any:
        """睡眠指定秒数"""
        await asyncio.sleep(sleep_seconds)
        return {"result": f"Slept for {sleep_seconds} seconds"}


class StrictTool(Tool):
    """严格参数工具 - 用于测试参数验证"""

    @property
    def name(self) -> str:
        return "strict_tool"

    @property
    def description(self) -> str:
        return "A tool with strict parameter requirements"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "description": "Name parameter"},
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                    "description": "Age parameter",
                },
            },
            "required": ["name", "age"],
        }

    async def execute(self, name: str, age: int) -> Any:
        """返回问候语"""
        return {"greeting": f"Hello {name}, you are {age} years old"}


async def test_timeout():
    """测试超时控制"""
    print("\n=== Test 1: Timeout Control ===")

    registry = ToolRegistry()
    registry.register(SlowTool())

    # 使用 5 秒超时
    executor = ToolExecutor(registry, default_timeout=5)

    # 测试 1: 正常执行 (3秒 < 5秒超时)
    print("\n1.1 Testing normal execution (3s < 5s timeout)...")
    result1 = await executor.execute(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "slow_tool", "arguments": '{"sleep_seconds": 3}'},
        }
    )
    print(f"✅ Result: success={result1.success}, result={result1.result}")

    # 测试 2: 超时 (10秒 > 5秒超时)
    print("\n1.2 Testing timeout (10s > 5s timeout)...")
    result2 = await executor.execute(
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "slow_tool", "arguments": '{"sleep_seconds": 10}'},
        }
    )
    print(f"❌ Result: success={result2.success}, error={result2.error}")


async def test_validation():
    """测试参数验证"""
    print("\n\n=== Test 2: Parameter Validation ===")

    registry = ToolRegistry()
    registry.register(StrictTool())

    executor = ToolExecutor(registry, default_timeout=5)

    # 测试 1: 有效参数
    print("\n2.1 Testing valid parameters...")
    result1 = await executor.execute(
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "strict_tool", "arguments": '{"name": "Alice", "age": 30}'},
        }
    )
    print(f"✅ Result: success={result1.success}, result={result1.result}")

    # 测试 2: 缺少必需参数
    print("\n2.2 Testing missing required parameter...")
    result2 = await executor.execute(
        {
            "id": "call_4",
            "type": "function",
            "function": {
                "name": "strict_tool",
                "arguments": '{"name": "Bob"}',  # 缺少 age
            },
        }
    )
    print(f"❌ Result: success={result2.success}, error={result2.error}")

    # 测试 3: 参数类型错误
    print("\n2.3 Testing wrong parameter type...")
    result3 = await executor.execute(
        {
            "id": "call_5",
            "type": "function",
            "function": {
                "name": "strict_tool",
                "arguments": '{"name": "Charlie", "age": "thirty"}',  # age 应该是 int
            },
        }
    )
    print(f"❌ Result: success={result3.success}, error={result3.error}")

    # 测试 4: 参数超出范围
    print("\n2.4 Testing parameter out of range...")
    result4 = await executor.execute(
        {
            "id": "call_6",
            "type": "function",
            "function": {
                "name": "strict_tool",
                "arguments": '{"name": "David", "age": 200}',  # age > 150
            },
        }
    )
    print(f"❌ Result: success={result4.success}, error={result4.error}")


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("Tool Executor Tests: Timeout & Validation")
    print("=" * 60)

    try:
        await test_timeout()
        await test_validation()

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
