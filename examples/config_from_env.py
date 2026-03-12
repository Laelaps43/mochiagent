"""
演示如何使用环境变量配置 Mochi Agent

运行方式：
1. 创建 .env 文件：
   cp .env.example .env

2. 修改 .env 中的值

3. 运行脚本：
   python examples/config_from_env.py
"""

import os
from agent.config import MessageBusConfig, SystemConfig, ToolRuntimeConfig
from agent.core.utils import gen_id


def print_config():
    """打印当前配置（用于调试）"""
    print("=" * 60)
    print("当前 Mochi Agent 配置")
    print("=" * 60)

    print("\n[MessageBus]")
    mb = MessageBusConfig()
    print(f"  queue_timeout      : {mb.queue_timeout}s")
    print(f"  max_concurrent     : {mb.max_concurrent}")

    print("\n[System]")
    sys = SystemConfig()
    print(f"  uuid_prefix_length : {sys.uuid_prefix_length}")
    print(f"  default_session_state: {sys.default_session_state}")

    print("\n[Tool]")
    tool = ToolRuntimeConfig()
    print(f"  timeout            : {tool.timeout}s")
    print(f"  max_batch_concurrency: {tool.max_batch_concurrency}")
    print(f"  exec_max_output_chars: {tool.exec_max_output_chars}")
    print(f"  web_fetch_max_chars: {tool.web_fetch_max_chars}")

    print("\n" + "=" * 60)


def demo_usage():
    """演示配置的使用"""
    print("\n演示配置使用:\n")

    # 1. gen_id 使用 SystemConfig
    session_id = gen_id("session_")
    user_id = gen_id("user_")
    print(f"生成的 session_id: {session_id}")
    print(f"生成的 user_id   : {user_id}")

    # 2. 获取配置用于业务逻辑
    tool_config = ToolRuntimeConfig()
    print(f"\n工具超时设置为: {tool_config.timeout}s")

    sys_config = SystemConfig()
    print(f"UUID 长度设置为: {sys_config.uuid_prefix_length}")


def demo_override():
    """演示通过环境变量覆盖配置"""
    print("\n\n演示环境变量覆盖:\n")

    # 临时设置环境变量
    os.environ["MOCHI_TIMEOUT"] = "999"
    os.environ["MOCHI_UUID_PREFIX_LENGTH"] = "4"

    config = ToolRuntimeConfig()
    print("设置 MOCHI_TIMEOUT=999")
    print(f"  → ToolRuntimeConfig.timeout = {config.timeout}s")

    sys_config = SystemConfig()
    print("\n设置 MOCHI_UUID_PREFIX_LENGTH=4")
    print(f"  → SystemConfig.uuid_prefix_length = {sys_config.uuid_prefix_length}")

    # 生成更短的 ID
    short_id = gen_id("short_")
    print(f"  → gen_id('short_') = {short_id} (长度: {len(short_id) - 6})")


if __name__ == "__main__":
    # 打印当前配置
    print_config()

    # 演示使用
    demo_usage()

    # 演示覆盖
    demo_override()

    print("\n\n💡 提示:")
    print("  - 修改 .env 文件来更改配置")
    print("  - 所有环境变量使用 MOCHI_ 前缀")
    print("  - 字段名自动转为大写: queue_timeout → MOCHI_QUEUE_TIMEOUT")
    print("  - 优先级: 环境变量 > .env 文件 > 代码默认值")
