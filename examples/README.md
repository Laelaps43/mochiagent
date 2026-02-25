# 示例

这个目录包含 MochiAgent 的可运行示例。

## `minimal_chat.py`

这是一个最小端到端示例，包含：

- 工具注册（`echo`）
- 会话监听器事件消费
- `part.created`、`message.done`、`llm.error` 处理

运行方式：

```bash
export OPENAI_API_KEY="your_key"
uv run python examples/minimal_chat.py "请调用 echo 工具并回复 hello"
```

可选环境变量：

- `OPENAI_BASE_URL`（默认：`https://api.openai.com/v1`）
- `MOCHI_MODEL`（默认：`gpt-4o-mini`）
