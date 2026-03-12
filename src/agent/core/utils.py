from __future__ import annotations

import time
from uuid import uuid4

from agent.config.system import SystemConfig

__all__ = [
    "to_non_negative_int",
    "to_int",
    "gen_id",
    "now_ms",
    "estimate_tokens",
    "truncate_text",
    "parse_name_list",
    "normalize_profile_id",
    "format_exception",
]


# ---- numeric helpers ----


def _parse_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def to_non_negative_int(value: object, *, default: int = 0) -> int:
    parsed = _parse_int(value)
    if parsed is None:
        return default
    return max(parsed, 0)


def to_int(value: object, *, default: int = 0, minimum: int = 0) -> int:
    parsed = _parse_int(value)
    if parsed is None:
        parsed = default
    return max(minimum, parsed)


# ---- id / time helpers ----

_UUID_PREFIX_LENGTH: int = SystemConfig().uuid_prefix_length


def gen_id(prefix: str = "") -> str:
    """生成带可选前缀的短 UUID 标识符。"""
    return f"{prefix}{uuid4().hex[:_UUID_PREFIX_LENGTH]}"


def now_ms() -> int:
    """当前时间戳（毫秒）。"""
    return int(time.time() * 1000)


# ---- text / token helpers ----


def estimate_tokens(text_or_chars: str | int, chars_per_token: float) -> int:
    char_count = len(text_or_chars) if isinstance(text_or_chars, str) else text_or_chars
    return max(int(char_count / max(chars_per_token, 1.0)), 0)


def truncate_text(value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(value)
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def parse_name_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item and item.strip()}


# ---- exception helpers ----


def _collect_exception_messages(exc: BaseException, out: list[str]) -> None:
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            _collect_exception_messages(sub, out)
        return
    out.append(f"{type(exc).__name__}: {exc}")


def format_exception(exc: BaseException) -> str:
    """将异常（含 ExceptionGroup）格式化为单行摘要。"""
    messages: list[str] = []
    _collect_exception_messages(exc, messages)
    if not messages:
        return f"{type(exc).__name__}: {exc}"
    unique: list[str] = []
    for msg in messages:
        if msg not in unique:
            unique.append(msg)
    if len(unique) <= 3:
        return " | ".join(unique)
    return " | ".join(unique[:3]) + f" | ... (+{len(unique) - 3} more)"


# ---- profile helpers ----


def normalize_profile_id(profile_id: str) -> str:
    raw = profile_id.strip()
    if ":" not in raw:
        raise ValueError(
            f"Invalid model profile id '{profile_id}'. Expected format: provider:model"
        )
    provider, model = raw.split(":", 1)
    if not provider.strip() or not model.strip():
        raise ValueError("provider and model are required to build llm profile id")
    return f"{provider.strip().lower()}:{model.strip()}"
