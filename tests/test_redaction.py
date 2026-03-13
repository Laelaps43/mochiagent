from __future__ import annotations

from typing import cast

from agent.core.security.redaction import (
    REDACTED,
    mask_secret,
    redact_dict,
    redact_text,
)


class TestMaskSecret:
    def test_none_returns_none(self):
        assert mask_secret(None) is None

    def test_non_string_returns_redacted(self):
        assert mask_secret(12345) == REDACTED
        assert mask_secret(["list"]) == REDACTED

    def test_empty_string_returns_empty(self):
        assert mask_secret("") == ""

    def test_already_redacted_passthrough(self):
        assert mask_secret(REDACTED) == REDACTED

    def test_short_string_masked(self):
        assert mask_secret("abc") == "***"
        assert mask_secret("abcdefgh") == "***"

    def test_long_string_partial_mask(self):
        result = mask_secret("supersecretkey123")
        assert isinstance(result, str)
        assert result.startswith("supe")
        assert result.endswith("123")
        assert "***" in result


class TestRedactDict:
    def test_non_dict_passthrough(self):
        assert redact_dict("hello") == "hello"
        assert redact_dict(42) == 42
        assert redact_dict(None) is None

    def test_sensitive_key_masked(self):
        result = redact_dict({"api_key": "supersecretvalue123"})
        assert isinstance(result, dict)
        val: object = cast(dict[str, object], result)["api_key"]
        assert val != "supersecretvalue123"

    def test_non_sensitive_key_unchanged(self):
        result = redact_dict({"username": "alice"})
        assert result == {"username": "alice"}

    def test_nested_dict_recursive(self):
        data = {"config": {"api_key": "topsecretsssss", "host": "localhost"}}
        result = redact_dict(data)
        assert isinstance(result, dict)
        inner: dict[str, object] = cast(
            dict[str, object], cast(dict[str, object], result)["config"]
        )
        assert isinstance(inner, dict)
        assert inner["host"] == "localhost"
        assert inner["api_key"] != "topsecretsssss"

    def test_list_items_processed(self):
        data = [{"password": "p@ssw0rd1234567"}, {"name": "bob"}]
        result = redact_dict(data)
        assert isinstance(result, list)
        assert result[1] == {"name": "bob"}
        assert result[0]["password"] != "p@ssw0rd1234567"

    def test_tuple_items_processed(self):
        data = ({"token": "tok12345678abcd"}, "plain")
        result = redact_dict(data)
        assert isinstance(result, tuple)
        assert result[1] == "plain"

    def test_all_sensitive_keys_redacted(self):
        keys = [
            "api_key",
            "apikey",
            "authorization",
            "token",
            "access_token",
            "refresh_token",
            "secret",
            "client_secret",
            "password",
            "private_key",
        ]
        for key in keys:
            res = redact_dict({key: "averylongsecretvalue123"})
            assert isinstance(res, dict)
            assert res[key] != "averylongsecretvalue123", f"{key} not redacted"

    def test_hyphenated_key_redacted(self):
        result = redact_dict({"x-api-key": "mylongsecrethere1234"})
        assert isinstance(result, dict)
        assert result["x-api-key"] != "mylongsecrethere1234"

    def test_non_string_key_not_sensitive(self):
        result = redact_dict({1: "value"})
        assert isinstance(result, dict)
        assert result[1] == "value"


class TestRedactText:
    def test_none_returns_empty_string(self):
        assert redact_text(None) == ""

    def test_no_sensitive_content_unchanged(self):
        text = "Hello, world! Nothing secret here."
        assert redact_text(text) == text

    def test_bearer_token_redacted(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9"
        result = redact_text(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert REDACTED in result

    def test_api_key_value_redacted(self):
        text = 'api_key: "sk-abcdef1234567890"'
        result = redact_text(text)
        assert "sk-abcdef1234567890" not in result
        assert REDACTED in result

    def test_secret_equals_redacted(self):
        text = "secret=mysupersecrettoken1234"
        result = redact_text(text)
        assert "mysupersecrettoken1234" not in result
