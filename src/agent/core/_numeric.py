"""Low-level numeric conversion helpers with no cross-package imports.

Kept in a dedicated module so that ``agent.types`` can import from here
without creating a circular dependency with ``agent.core.utils``.
"""

from __future__ import annotations


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
