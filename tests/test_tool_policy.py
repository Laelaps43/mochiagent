from __future__ import annotations

from agent.core.tools.policy import PolicyDecision, ToolPolicyConfig, ToolPolicyEngine


class TestToolPolicyConfig:
    def test_defaults_none(self):
        cfg = ToolPolicyConfig()
        assert cfg.allow is None
        assert cfg.deny is None

    def test_normalized_lowercases(self):
        cfg = ToolPolicyConfig(allow={"Read", "WRITE"}, deny={"EXEC"})
        norm = cfg.normalized()
        assert norm.allow == {"read", "write"}
        assert norm.deny == {"exec"}

    def test_normalized_none_becomes_empty_set(self):
        cfg = ToolPolicyConfig()
        norm = cfg.normalized()
        assert norm.allow == set()
        assert norm.deny == set()

    def test_from_csv_parses(self):
        cfg = ToolPolicyConfig.from_csv(allow_csv="read,write", deny_csv="exec")
        assert cfg.allow == {"read", "write"}
        assert cfg.deny == {"exec"}

    def test_from_csv_none(self):
        cfg = ToolPolicyConfig.from_csv()
        assert cfg.allow == set()
        assert cfg.deny == set()


class TestToolPolicyEngine:
    def test_allow_all_by_default(self):
        engine = ToolPolicyEngine(ToolPolicyConfig())
        decision = engine.evaluate("any_tool")
        assert decision.allowed is True
        assert "default" in decision.reason

    def test_deny_overrides_allow(self):
        engine = ToolPolicyEngine(ToolPolicyConfig(allow={"exec"}, deny={"exec"}))
        decision = engine.evaluate("exec")
        assert decision.allowed is False
        assert "denied" in decision.reason

    def test_allow_list_restricts(self):
        engine = ToolPolicyEngine(ToolPolicyConfig(allow={"read", "write"}))
        assert engine.evaluate("read").allowed is True
        assert engine.evaluate("write").allowed is True
        assert engine.evaluate("exec").allowed is False

    def test_deny_blocks_regardless_of_allow(self):
        engine = ToolPolicyEngine(ToolPolicyConfig(deny={"exec"}))
        assert engine.evaluate("exec").allowed is False
        assert engine.evaluate("read").allowed is True

    def test_case_insensitive_evaluation(self):
        engine = ToolPolicyEngine(ToolPolicyConfig(deny={"exec"}))
        assert engine.evaluate("EXEC").allowed is False
        assert engine.evaluate("Exec").allowed is False

    def test_whitespace_trimmed(self):
        engine = ToolPolicyEngine(ToolPolicyConfig(deny={"exec"}))
        assert engine.evaluate("  exec  ").allowed is False

    def test_empty_tool_name_allowed_by_default(self):
        engine = ToolPolicyEngine(ToolPolicyConfig())
        assert engine.evaluate("").allowed is True

    def test_policy_decision_fields(self):
        engine = ToolPolicyEngine(ToolPolicyConfig(allow={"read"}))
        decision = engine.evaluate("write")
        assert isinstance(decision, PolicyDecision)
        assert decision.allowed is False
        assert "TOOLS_POLICY_ALLOW" in decision.reason
