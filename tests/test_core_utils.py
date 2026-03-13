from __future__ import annotations

import pytest

from agent.core.utils import (
    parse_int,
    estimate_tokens,
    format_exception,
    gen_id,
    normalize_profile_id,
    now_ms,
    parse_name_list,
    to_int,
    to_non_negative_int,
    truncate_text,
)


class TestParseInt:
    def test_int_passthrough(self):
        assert parse_int(42) == 42
        assert parse_int(0) == 0
        assert parse_int(-5) == -5

    def test_bool_returns_none(self):
        assert parse_int(True) is None
        assert parse_int(False) is None

    def test_float_truncated(self):
        assert parse_int(3.9) == 3
        assert parse_int(-1.1) == -1

    def test_string_parsed(self):
        assert parse_int("10") == 10
        assert parse_int("-3") == -3

    def test_invalid_string_returns_none(self):
        assert parse_int("abc") is None
        assert parse_int("") is None

    def test_none_returns_none(self):
        assert parse_int(None) is None

    def test_list_returns_none(self):
        assert parse_int([]) is None


class TestToNonNegativeInt:
    def test_positive_unchanged(self):
        assert to_non_negative_int(5) == 5

    def test_negative_clamped_to_zero(self):
        assert to_non_negative_int(-3) == 0

    def test_invalid_uses_default(self):
        assert to_non_negative_int("bad") == 0
        assert to_non_negative_int("bad", default=7) == 7

    def test_zero_stays_zero(self):
        assert to_non_negative_int(0) == 0


class TestToInt:
    def test_basic(self):
        assert to_int(5) == 5

    def test_respects_minimum(self):
        assert to_int(-10, minimum=0) == 0
        assert to_int(3, minimum=5) == 5

    def test_invalid_uses_default(self):
        assert to_int("bad", default=10) == 10

    def test_default_also_respects_minimum(self):
        assert to_int("bad", default=1, minimum=3) == 3


class TestGenId:
    def test_generates_string(self):
        assert isinstance(gen_id(), str)
        assert len(gen_id()) > 0

    def test_prefix_prepended(self):
        result = gen_id("msg_")
        assert result.startswith("msg_")

    def test_ids_are_unique(self):
        ids = {gen_id() for _ in range(100)}
        assert len(ids) == 100


class TestNowMs:
    def test_returns_int(self):
        assert isinstance(now_ms(), int)

    def test_reasonable_timestamp(self):
        ms = now_ms()
        assert ms > 1_700_000_000_000


class TestEstimateTokens:
    def test_string_input(self):
        tokens = estimate_tokens("hello world", 4.0)
        assert tokens == 2

    def test_int_input(self):
        tokens = estimate_tokens(400, 4.0)
        assert tokens == 100

    def test_zero_chars(self):
        assert estimate_tokens("", 4.0) == 0
        assert estimate_tokens(0, 4.0) == 0

    def test_zero_chars_per_token_clamped(self):
        assert estimate_tokens(100, 0.0) == 100


class TestTruncateText:
    def test_short_text_unchanged(self):
        text, truncated = truncate_text("hi", 100)
        assert text == "hi"
        assert truncated is False

    def test_exact_length_unchanged(self):
        text, truncated = truncate_text("hello", 5)
        assert text == "hello"
        assert truncated is False

    def test_long_text_truncated(self):
        text, truncated = truncate_text("hello world", 5)
        assert text == "hello"
        assert truncated is True

    def test_zero_max_chars_empty(self):
        text, truncated = truncate_text("hi", 0)
        assert text == ""
        assert truncated is True

    def test_zero_max_chars_empty_string(self):
        text, truncated = truncate_text("", 0)
        assert text == ""
        assert truncated is False


class TestParseNameList:
    def test_none_returns_empty_set(self):
        assert parse_name_list(None) == set()

    def test_empty_string_returns_empty_set(self):
        assert parse_name_list("") == set()

    def test_csv_parsed(self):
        result = parse_name_list("a,b,c")
        assert result == {"a", "b", "c"}

    def test_lowercased(self):
        result = parse_name_list("Read,WRITE,Exec")
        assert result == {"read", "write", "exec"}

    def test_whitespace_stripped(self):
        result = parse_name_list("  read , write ")
        assert result == {"read", "write"}

    def test_empty_entries_skipped(self):
        result = parse_name_list("a,,b, ,c")
        assert result == {"a", "b", "c"}


class TestFormatException:
    def test_simple_exception(self):
        e = ValueError("something failed")
        result = format_exception(e)
        assert "ValueError" in result
        assert "something failed" in result

    def test_exception_group(self):
        eg = ExceptionGroup("group", [ValueError("a"), RuntimeError("b")])
        result = format_exception(eg)
        assert "ValueError" in result
        assert "RuntimeError" in result

    def test_deduplication(self):
        eg = ExceptionGroup("group", [ValueError("dup"), ValueError("dup")])
        result = format_exception(eg)
        assert result.count("dup") == 1

    def test_truncation_with_many_errors(self):
        eg = ExceptionGroup("group", [ValueError(str(i)) for i in range(10)])
        result = format_exception(eg)
        assert "more" in result


class TestNormalizeProfileId:
    def test_valid_profile(self):
        assert normalize_profile_id("openai:gpt-4") == "openai:gpt-4"

    def test_provider_lowercased(self):
        assert normalize_profile_id("OpenAI:gpt-4") == "openai:gpt-4"

    def test_whitespace_stripped(self):
        assert normalize_profile_id("  openai : gpt-4  ") == "openai:gpt-4"

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid model profile id"):
            _ = normalize_profile_id("nocolon")

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="provider and model are required"):
            _ = normalize_profile_id(":gpt-4")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="provider and model are required"):
            _ = normalize_profile_id("openai:")
