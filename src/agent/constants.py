"""
Constants - 框架常量定义
集中管理所有魔法值和配置常量
"""

# ============ 超时配置 ============

# MessageBus 事件队列等待超时 (秒)
MESSAGE_BUS_QUEUE_TIMEOUT = 1.0

# 工具执行默认超时 (秒)
TOOL_EXECUTION_TIMEOUT = 30

# LLM 请求默认超时 (秒) - 降低超时时间以快速发现问题
LLM_REQUEST_TIMEOUT = 60

# OpenAI 客户端最大重试次数
OPENAI_MAX_RETRIES = 2


# ============ ID 生成配置 ============

# UUID 前缀截取长度（16 hex = 64-bit，碰撞概率极低）
UUID_PREFIX_LENGTH = 16


# ============ 并发控制 ============

# MessageBus 默认最大并发数
DEFAULT_MAX_CONCURRENT = 50


# ============ 状态机配置 ============

# 会话默认状态
DEFAULT_SESSION_STATE = "idle"
