from agent.core.security import REDACTED, mask_secret, redact_dict, redact_text


def test_mask_secret_keeps_edge_characters_for_long_value():
    assert mask_secret("abcd5123456789012345678.XyZaBcDeFgHiJkLm") == "abcd***kLm"


def test_redact_dict_masks_sensitive_fields_recursively():
    payload = {
        "api_key": "abc123456789xyz",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "extra_params": {
            "Authorization": "Bearer token-value-123",
            "nested": [{"access_token": "token-123456"}],
        },
    }
    result = redact_dict(payload)
    assert result["api_key"] == "abc1***xyz"
    assert result["base_url"] == payload["base_url"]
    assert result["extra_params"]["Authorization"].startswith("Bear")
    assert result["extra_params"]["nested"][0]["access_token"] == "toke***456"


def test_redact_text_masks_key_value_and_bearer_token():
    text = "api_key=abcdef123456 Authorization: Bearer xyz987654321"
    redacted = redact_text(text)
    assert "abcdef123456" not in redacted
    assert "xyz987654321" not in redacted
    assert REDACTED in redacted
