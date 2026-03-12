# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MochiAgent 是一个异步 LLM Agent 框架，核心能力：事件驱动会话 + 工具编排 + 安全约束 + MCP 集成。Python >=3.13，包管理使用 uv。

## Common Commands

```bash
# 安装依赖
uv sync --extra dev

# Lint 检查
uv run --extra dev ruff check .

# 格式化检查
uv run --extra dev ruff format --check .

# 运行测试
uv run --extra dev pytest -q

# 运行单个测试
uv run --extra dev pytest tests/path/to/test_file.py -q

# 类型检查
uv run --extra dev basedpyright

# 构建
uv build
```

## Architecture

源码位于 `src/agent/`，wheel 打包路径为 `src/agent`，对外包名为 `agent`。

### 分层结构

```
src/agent/
├── __init__.py          # 公共 API：setup(), get_agent(), shutdown()
├── base_agent.py        # BaseAgent 抽象基类，用户继承实现自己的 Agent
├── framework.py         # AgentFramework 单例，管理所有子系统
├── session.py           # Session 包装，对外事件订阅接口
├── types.py             # 核心类型定义（Event, LLMConfig, Message 等）
├── config/              # 工具策略、安全、工作区配置
├── common/              # 内置工具（文件/Web/exec/技能）和通用 prompt
└── core/                # 框架核心模块
```

### Core 子系统

- **bus/** — `MessageBus` 异步事件队列，组件间通信
- **llm/** — LLM 适配器（当前仅 `OpenAIAdapter`，兼容 OpenAI 协议的服务均可接入）。`AdapterRegistry` 注册自定义 provider
- **loop/** — `AgentEventLoop` 处理 LLM turn 循环、工具调用编排
- **session/** — `SessionManager` + `SessionStateMachine`（状态机基于 transitions 库，状态：idle → processing → error/completed）
- **tools/** — `ToolRegistry` 注册、`ToolExecutor` 执行、`ToolSecurityGuard` 安全校验
- **mcp/** — MCP 服务器生命周期管理，自动从 MCP 服务注册工具
- **storage/** — 抽象 `StorageProvider`，默认 `MemoryStorage`（仅开发用）
- **compression/** — 上下文压缩策略（`ContextCompactor`）
- **runtime/** — `AgentStrategyManager` 运行时策略插件

### 关键设计模式

- **依赖注入**：`AgentContext` 持有 session_manager、message_bus、strategy_manager，通过框架注入
- **适配器模式**：LLM 接入通过 `LLMProvider` 抽象 + `AdapterRegistry` 注册
- **状态机**：Session 生命周期由 transitions 库驱动
- **策略模式**：上下文压缩、工具结果后处理均可插拔替换

### 事件流

用户调用 `push_message()` → MessageBus 派发 → EventLoop 调用 LLM → 流式产出 `part.created` 事件（text/reasoning/tool）→ 工具调用由 ToolExecutor 执行 → 循环直到 LLM 返回 stop → `message.done` 事件。

对外可订阅事件：`part.created`、`message.done`、`llm.error`。

## Code Conventions

- Ruff lint + format，行宽 100
- BasedPyright "recommended" 模式做类型检查
- 数据模型使用 Pydantic BaseModel
- 全异步（async/await），不使用同步阻塞调用
- LLM profile_id 格式：`provider:model`（如 `openai:gpt-4o-mini`）
- 自定义 Tool 需实现 `name`、`description`、`parameters_schema` 属性和 `execute()` 方法
