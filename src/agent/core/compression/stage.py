"""Compaction stage enums."""

from __future__ import annotations

from enum import Enum


class CompactionStage(str, Enum):
    PRE_CALL = "pre_call"
    OVERFLOW_ERROR = "overflow_error"
    MID_TURN = "mid_turn"
