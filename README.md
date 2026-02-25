# MochiAgent

MochiAgent 是一个面向生产场景的异步 LLM Agent 框架，核心能力是「事件驱动会话 + 工具编排 + 安全约束 + MCP 集成」。

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [可运行示例](#可运行示例)
- [事件系统](#事件系统)
- [配置说明](#配置说明)
- [LLM 接入](#llm-接入)
- [MCP 集成](#mcp-集成)
- [Skills（技能系统）](#skills技能系统)
- [存储扩展](#存储扩展)
- [发布流程](#发布流程)
- [核心 API](#核心-api)
- [变更日志](#变更日志)
- [许可证](#许可证)

## 环境要求

- Python `>=3.13`

## 安装

从源码安装：

```bash
git clone <your-repo-url>
cd <repo-dir>
uv sync --extra dev
```

发布后安装：

```bash
pip install mochiagent
```

## 快速开始

```python
import asyncio
from pathlib import Path

from agent import BaseAgent, Event, EventType, LLMConfig, Tool, get_agent, setup, shutdown


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "回显输入文本"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, text: str):
        return {"text": text}


class DemoAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "demo_agent"

    @property
    def description(self) -> str:
        return "示例 Agent"

    @property
    def skill_directory(self) -> Path | None:
        return None

    async def setup(self) -> None:
        self.register_tool(EchoTool())


async def main() -> None:
    llm_config = LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model="gpt-4o-mini",
        api_key="YOUR_API_KEY",
        base_url="https://api.openai.com/v1",
        stream=True,
        openai_max_retries=2,
    )
    await setup(
        agents=[DemoAgent()],
        llm_configs=[llm_config],
        max_concurrent=50,
        max_iterations=100,
    )

    agent = get_agent("demo_agent")
    if agent is None:
        raise RuntimeError("demo_agent not found")

    session = await agent.take_session(
        "demo-session-1",
        model_profile_id="openai:gpt-4o-mini",
    )
    queue: asyncio.Queue[Event] = asyncio.Queue()

    async def listener(event: Event):
        await queue.put(event)

    session.add_listener(listener)

    try:
        await agent.push_message(session.session_id, "请调用 echo 工具并回复")

        while True:
            event = await queue.get()

            if event.type == EventType.PART_CREATED:
                part = event.data or {}
                if part.get("type") == "text":
                    print(part.get("text", ""), end="", flush=True)

            if event.type == EventType.LLM_ERROR:
                print("\n[LLM_ERROR]", event.data)
                break

            if event.type == EventType.MESSAGE_DONE:
                print("\n[DONE]")
                break
    finally:
        session.remove_listener(listener)
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())
```

## 可运行示例

仓库提供可直接运行的最小示例，位于 `examples/`：

- 说明文档：`examples/README.md`
- 代码：`examples/minimal_chat.py`

快速运行：

```bash
export OPENAI_API_KEY="your_key"
uv run python examples/minimal_chat.py "请调用 echo 工具并回复 hello"
```

## 事件系统

### 1) 统一事件包络（Event）

所有事件都使用同一个结构：

```python
{
  "type": "event.type",
  "session_id": "sess_xxx",
  "data": {},
  "timestamp": "datetime对象",
  "metadata": {}
}
```

字段说明：

- `type`: 事件类型（见下方完整清单）
- `session_id`: 会话 ID
- `data`: 事件负载（不同事件不同结构）
- `timestamp`: 事件创建时间（`datetime`）
- `metadata`: 预留元数据

### 2) 对外可订阅事件

对外稳定契约以 `Session.add_listener()` 为准，当前可收到 3 个事件：

- `part.created`
- `message.done`
- `llm.error`

### 3) 事件清单与 `data` 结构（对外）

#### `part.created`

触发时机：流式文本、思考片段、工具状态更新时。

`data` 有三种形态：

1. 文本 Part（`type="text"`）

```json
{
  "id": "part_xxx",
  "session_id": "sess_xxx",
  "message_id": "msg_xxx",
  "type": "text",
  "text": "你好",
  "synthetic": null,
  "ignored": null,
  "time": { "start": 1730000000000, "end": null },
  "metadata": null
}
```

2. 思考 Part（`type="reasoning"`）

```json
{
  "id": "part_xxx",
  "session_id": "sess_xxx",
  "message_id": "msg_xxx",
  "type": "reasoning",
  "text": "...",
  "time": { "start": 1730000000000, "end": 1730000000123 },
  "metadata": null
}
```

3. 工具 Part（`type="tool"`）

```json
{
  "id": "part_xxx",
  "session_id": "sess_xxx",
  "message_id": "msg_xxx",
  "type": "tool",
  "call_id": "call_xxx",
  "tool": "read_file",
  "state": {
    "status": "running",
    "input": { "arguments": "{\"path\":\"README.md\"}" },
    "title": "read_file",
    "metadata": null,
    "time": { "start": 1730000000000 }
  },
  "metadata": null
}
```

工具 `state.status` 可能值与字段：

- `running`: `status`、`input`、`title`、`metadata`、`time.start`
- `completed`: `status`、`input`、`output`、`summary`、`artifact_ref`、`artifact_path`、`raw_size_chars`、`truncated`、`title`、`metadata`、`time.start/end`
- `error`: `status`、`input`、`error`、`metadata`、`time.start/end`

#### `message.done`

触发时机：一轮助手消息结束时。

`data` 结构：

```json
{
  "message_id": "msg_xxx",
  "cost": 0.0,
  "tokens": { "input": 0, "output": 0, "reasoning": 0 },
  "finish": "stop"
}
```

`finish` 常见值：

- `stop`：正常结束
- `error`：异常结束（通常会先收到 `llm.error`）
- `max_iterations_exceeded`：超过最大轮数
- 其他 provider 返回的结束原因（原样透传）

#### `llm.error`

触发时机：LLM 调用异常、状态非法、超出最大轮数等。

`data` 基础结构：

```json
{
  "error": "错误描述",
  "code": "可选错误码",
  "hint": "可选修复建议"
}
```

当达到最大轮数时，还会包含：

```json
{
  "error": "Conversation stopped: maximum iterations exceeded (100)",
  "code": "MAX_ITERATIONS_EXCEEDED",
  "max_iterations": 100,
  "iterations": 100
}
```

### 4) 常见事件顺序（对外）

正常路径：

```text
part.created (text/reasoning/tool)
part.created (tool completed/error，可选)
message.done
```

异常路径：

```text
llm.error
message.done
```

## 配置说明

### 1) 框架启动参数（`setup`）

| 参数             | 类型              |            默认值 | 说明                        |
| ---------------- | ----------------- | ----------------: | --------------------------- |
| `agents`         | `list[BaseAgent]` |            `None` | 要注册的 Agent 实例列表     |
| `llm_configs`    | `list[LLMConfig]` |            `None` | 初始化加载 LLM 配置（必选于实际对话） |
| `storage`        | `StorageProvider` | `MemoryStorage()` | 会话/消息/artifact 存储后端 |
| `max_concurrent` | `int`             |              `50` | 事件处理最大并发            |
| `max_iterations` | `int`             |             `100` | 单次对话最大 LLM 轮数       |

### 2) LLM 配置（`LLMConfig`）

| 字段                 | 类型             | 默认值 | 说明                       |
| -------------------- | ---------------- | -----: | -------------------------- |
| `adapter`            | `str`            |      - | 适配器标识（如 `openai_compatible`） |
| `provider`           | `str`            |      - | 提供商标识（如 `openai`）  |
| `model`              | `str`            |      - | 模型名                     |
| `api_key`            | `str \| None`    | `None` | API Key                    |
| `base_url`           | `str \| None`    | `None` | OpenAI 兼容网关地址        |
| `temperature`        | `float`          |  `0.7` | 采样温度                   |
| `max_tokens`         | `int \| None`    | `None` | 最大生成 token             |
| `stream`             | `bool`           | `True` | 是否流式                   |
| `timeout`            | `int`            |   `60` | 请求超时（秒）             |
| `openai_max_retries` | `int \| None`    | `None` | OpenAI SDK 重试次数        |
| `extra_params`       | `dict[str, Any]` |   `{}` | 透传给 provider 的额外参数 |

### 3) 工具运行配置（`ToolRuntimeConfig`）

`BaseAgent(tools=...)` 支持细粒度工具运行约束（超时、白黑名单、工作区限制、命令 token 防护、输出截断等）。

推荐生产配置：

- 默认禁用 `exec`：`policy.deny={"exec"}`
- 开启工作区约束：`workspace.restrict=True`
- 开启命令防护：`enforce_command_guard=True`
- 永远不要把原始用户输入直接拼进 shell 命令

#### Exec 风险与常见问题

`exec` 是高风险工具（基于 shell 执行命令），建议仅在受控环境启用。常见问题如下：

- 策略拦截：如果命中工具策略，会返回 `TOOL_POLICY_DENIED`。
- 安全拦截：如果命令包含禁止 token（如 `` ` ``、`$(`、换行）或路径越界，会返回 `TOOL_SECURITY_DENIED`。
- 执行超时：超过工具超时时间会返回 `Tool execution timeout after <N>s`。
- 输出被截断：`exec` 会保留完整 `stdout/stderr` 字段，但聚合输出会按 `exec_max_output_chars` 截断，`truncated=true` 表示已截断。
- shell 语义差异：命令由 shell 解析，重定向、引用和通配符行为与直接进程调用不同，容易出现“本地可跑、线上失败”。

排查建议：

- 在 `part.created` 的 tool 结果里优先看 `state.status`、`state.metadata.error`、`truncated`。
- 先最小化命令（去掉管道和重定向）确认问题边界，再逐步恢复。
- 生产环境默认禁用 `exec`，仅在容器/低权限账户中按需放开。

## LLM 接入

### 1) 默认方式：OpenAI 兼容协议

可接入 OpenAI 官方或任何兼容网关。

- `adapter`：决定走哪个适配器实现（如 `openai_compatible`）
- `provider`：业务侧供应商标识（如 `openai`、`zhipu`）
- `model_profile_id` 固定格式：`provider:model`（例如 `zhipu:glm-4.7`）

```python
from agent import LLMConfig

llm_config = LLMConfig(
    adapter="openai_compatible",
    provider="openai",
    model="gpt-4o-mini",
    api_key="YOUR_API_KEY",
    base_url="https://api.openai.com/v1",
    stream=True,
    timeout=60,
    openai_max_retries=2,
)
```

### 2) 自定义 Provider

可实现 `LLMProvider` 并注册到 `AdapterRegistry`。

```python
from typing import Any, AsyncIterator

from agent import get_framework
from agent.core.llm import LLMProvider


class MyProvider(LLMProvider):
    async def stream_chat(self, messages: list[dict[str, Any]], tools=None, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"content": "hello", "finish_reason": "stop"}

    async def complete(self, messages, tools=None, **kwargs):
        return {"content": "hello", "finish_reason": "stop", "tool_calls": []}


framework = get_framework()
framework.adapter_registry.register("my_adapter", MyProvider)
```

## MCP 集成

在 Agent 中覆盖 `mcp_config_path`，并在 `setup()` 里调用 `register_mcp_tools()`。

```python
from pathlib import Path


class DemoAgent(BaseAgent):
    @property
    def mcp_config_path(self) -> Path | None:
        return Path(__file__).with_name("mcp.json")

    async def setup(self) -> None:
        await self.register_mcp_tools()
```

`mcp.json` 示例：

```json
{
  "mcpServers": {
    "docs": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "connectTimeoutMs": 8000,
      "maxRetries": 2,
      "retryInitialMs": 300,
      "retryMaxMs": 3000,
      "failureThreshold": 3,
      "cooldownSec": 20,
      "toolTimeout": 30
    }
  }
}
```

## Skills（技能系统）

Skills 用来复用领域提示词/流程模板。典型用法：

- 在 Agent `setup()` 里调用 `register_skill("skill-name")`
- 框架统一提供 `skill` 工具
- 模型在运行时通过 `skill(name, context)` 读取技能内容

目录示例：

```text
my_agent/
├── agent.py
└── skills/
    ├── data-analysis/
    │   └── SKILL.md
    └── sql-query/
        └── SKILL.md
```

## 存储扩展

默认 `MemoryStorage` 适合开发和测试，不适合生产持久化。

生产环境建议实现并注入 `StorageProvider`：

```python
await setup(
    agents=[DemoAgent()],
    storage=MyStorage(),
)
```

## 发布流程

已配置自动发布工作流：`.github/workflows/release.yml`

- 触发方式：推送 tag（如 `v0.1.0`）
- 自动执行：
  - 构建 `sdist`/`wheel`
  - 创建 GitHub Release 并上传构建产物
  - 发布到 PyPI（基于 GitHub OIDC Trusted Publishing）

发布前请先在 PyPI 项目设置中配置 Trusted Publisher，绑定当前仓库与 workflow。

示例：

```bash
git tag v0.1.0
git push origin v0.1.0
```

## 核心 API

- `setup(agents, llm_configs, storage, max_concurrent, max_iterations)`
- `get_agent(agent_name)`
- `shutdown()`
- `BaseAgent.take_session(session_id, model_profile_id)`
- `BaseAgent.push_message(session_id, message)`
- `BaseAgent.register_skill(skill_name)`
- `Session.add_listener(listener)` / `Session.remove_listener(listener)`

## 变更日志

发布变更记录见：`CHANGELOG.md`

## 许可证

MIT
