# 环境变量配置指南

## 概述

所有配置类都支持从 `.env` 文件或系统环境变量读取。优先级：**环境变量 > .env 文件 > 代码默认值**

## 快速开始

### 1. 创建 .env 文件

在项目根目录创建 `.env` 文件：

```bash
# 复制示例文件
cp .env.example .env

# 编辑配置
vim .env
```

### 2. 使用配置

配置会**自动**从 `.env` 文件读取：

```python
from agent.config import MessageBusConfig, SystemConfig, ToolRuntimeConfig

# 自动从 .env 读取（如果文件存在）
mb_config = MessageBusConfig()
sys_config = SystemConfig()
tool_config = ToolRuntimeConfig()

print(f"工具超时: {tool_config.timeout}s")
```

## 环境变量列表

### MessageBus 配置

前缀：`MOCHI_MESSAGE_BUS_`

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `MOCHI_MESSAGE_BUS_QUEUE_TIMEOUT` | float | 1.0 | 队列等待超时（秒） |
| `MOCHI_MESSAGE_BUS_MAX_CONCURRENT` | int | 50 | 最大并发处理数 |

示例：
```bash
MOCHI_MESSAGE_BUS_QUEUE_TIMEOUT=2.0
MOCHI_MESSAGE_BUS_MAX_CONCURRENT=100
```

### System 配置

前缀：`MOCHI_SYSTEM_`

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `MOCHI_SYSTEM_UUID_PREFIX_LENGTH` | int | 16 | UUID 前缀长度（hex 字符数） |
| `MOCHI_SYSTEM_DEFAULT_SESSION_STATE` | str | "idle" | 会话默认状态 |

示例：
```bash
MOCHI_SYSTEM_UUID_PREFIX_LENGTH=12
MOCHI_SYSTEM_DEFAULT_SESSION_STATE=ready
```

### Tool 配置

前缀：`MOCHI_TOOL_`

| 环境变量 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| `MOCHI_TOOL_TIMEOUT` | int | 30 | 工具执行超时（秒） |
| `MOCHI_TOOL_MAX_BATCH_CONCURRENCY` | int | 10 | 批量工具最大并发数 |
| `MOCHI_TOOL_EXEC_MAX_OUTPUT_CHARS` | int | 20000 | 执行输出最大字符数 |
| `MOCHI_TOOL_WEB_FETCH_MAX_CHARS` | int | 20000 | Web 抓取最大字符数 |
| `MOCHI_TOOL_WEB_SEARCH_API_KEY` | str | "" | Web 搜索 API Key |

示例：
```bash
MOCHI_TOOL_TIMEOUT=60
MOCHI_TOOL_MAX_BATCH_CONCURRENCY=20
MOCHI_TOOL_EXEC_MAX_OUTPUT_CHARS=50000
MOCHI_TOOL_WEB_FETCH_MAX_CHARS=30000
MOCHI_TOOL_WEB_SEARCH_API_KEY=your-api-key-here
```

## 使用场景

### 场景 1: 开发环境

`.env.development`:
```bash
# 开发环境 - 更长的超时，更多日志
MOCHI_TOOL_TIMEOUT=120
MOCHI_MESSAGE_BUS_MAX_CONCURRENT=10
MOCHI_SYSTEM_UUID_PREFIX_LENGTH=8  # 更短的 ID 便于调试
```

### 场景 2: 生产环境

`.env.production`:
```bash
# 生产环境 - 更高性能
MOCHI_TOOL_TIMEOUT=30
MOCHI_MESSAGE_BUS_MAX_CONCURRENT=200
MOCHI_TOOL_MAX_BATCH_CONCURRENCY=50
```

### 场景 3: 测试环境

`.env.test`:
```bash
# 测试环境 - 更快超时，发现问题
MOCHI_TOOL_TIMEOUT=5
MOCHI_MESSAGE_BUS_QUEUE_TIMEOUT=0.1
```

## 高级用法

### 指定 .env 文件路径

```python
from agent.config import ToolRuntimeConfig

# 从特定文件读取
config = ToolRuntimeConfig(_env_file='config/production.env')
```

### 环境变量优先级

```python
import os

# 1. 系统环境变量（最高优先级）
os.environ['MOCHI_TOOL_TIMEOUT'] = '90'

# 2. .env 文件
# MOCHI_TOOL_TIMEOUT=60

# 3. 代码默认值（最低优先级）
from agent.config import ToolRuntimeConfig
config = ToolRuntimeConfig()
print(config.timeout)  # 输出: 90（来自环境变量）
```

### 程序化设置（覆盖所有来源）

```python
from agent.config import ToolRuntimeConfig

# 直接传参会覆盖环境变量和 .env
config = ToolRuntimeConfig(
    timeout=120,
    max_batch_concurrency=30
)
```

## Docker 部署

### Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY . .
RUN pip install -e .

# 可以在这里设置默认环境变量
ENV MOCHI_TOOL_TIMEOUT=60
ENV MOCHI_MESSAGE_BUS_MAX_CONCURRENT=100

CMD ["python", "main.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  mochi-agent:
    build: .
    environment:
      - MOCHI_TOOL_TIMEOUT=90
      - MOCHI_MESSAGE_BUS_MAX_CONCURRENT=200
      - MOCHI_TOOL_WEB_SEARCH_API_KEY=${WEB_SEARCH_API_KEY}
    env_file:
      - .env.production
```

### Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mochi-config
data:
  MOCHI_TOOL_TIMEOUT: "60"
  MOCHI_MESSAGE_BUS_MAX_CONCURRENT: "200"
  MOCHI_TOOL_MAX_BATCH_CONCURRENCY: "50"
---
apiVersion: v1
kind: Pod
metadata:
  name: mochi-agent
spec:
  containers:
  - name: agent
    image: mochi-agent:latest
    envFrom:
    - configMapRef:
        name: mochi-config
```

## 安全提示

### ⚠️ 不要提交敏感信息

```bash
# .gitignore
.env
.env.local
.env.*.local
```

### ✅ 使用示例文件

```bash
# 提交示例文件（不含敏感信息）
git add .env.example

# 忽略实际配置文件
echo ".env" >> .gitignore
```

### 🔒 API Key 管理

```bash
# ❌ 不要硬编码
MOCHI_TOOL_WEB_SEARCH_API_KEY=sk-real-key-12345

# ✅ 使用秘密管理服务
# AWS: AWS Secrets Manager
# Azure: Azure Key Vault
# GCP: Google Secret Manager
```

## 验证配置

```python
from agent.config import MessageBusConfig, SystemConfig, ToolRuntimeConfig

def print_config():
    """打印当前配置（用于调试）"""
    print("=== Current Configuration ===")

    mb = MessageBusConfig()
    print(f"\nMessageBus:")
    print(f"  queue_timeout: {mb.queue_timeout}s")
    print(f"  max_concurrent: {mb.max_concurrent}")

    sys = SystemConfig()
    print(f"\nSystem:")
    print(f"  uuid_prefix_length: {sys.uuid_prefix_length}")
    print(f"  default_session_state: {sys.default_session_state}")

    tool = ToolRuntimeConfig()
    print(f"\nTool:")
    print(f"  timeout: {tool.timeout}s")
    print(f"  max_batch_concurrency: {tool.max_batch_concurrency}")
    print(f"  exec_max_output_chars: {tool.exec_max_output_chars}")

if __name__ == "__main__":
    print_config()
```

## 常见问题

### Q: .env 文件不生效？
A: 确保：
1. 文件名正确（`.env`，不是 `env` 或 `.env.txt`）
2. 文件位置在项目根目录或工作目录
3. 环境变量名称正确（包括前缀）
4. 值的类型正确（int/float/str）

### Q: 如何禁用 .env 文件？
A:
```python
config = ToolRuntimeConfig(_env_file=None)
```

### Q: 如何读取多个 .env 文件？
A:
```python
from pydantic_settings import SettingsConfigDict

class MyConfig(ToolRuntimeConfig):
    model_config = SettingsConfigDict(
        env_file=['.env', '.env.local', '.env.production']
    )
```

### Q: 配置更改后需要重启吗？
A: 是的。配置在进程启动时加载，运行时修改 .env 文件不会生效，需要重启应用。

## 参考

- [Pydantic Settings 文档](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [12-Factor App: Config](https://12factor.net/config)
