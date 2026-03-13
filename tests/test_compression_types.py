from __future__ import annotations

from agent.core.compression.types import (
    CompactionDecision,
    CompactionPayload,
    CompactorRunOptions,
    SummaryBuildResult,
)


class TestCompactorRunOptions:
    def test_defaults(self):
        opts = CompactorRunOptions()
        assert opts.auto_compact_ratio == 0.9
        assert opts.keep_user_tokens_budget == 20000
        assert opts.chars_per_token == 4.0
        assert opts.model_auto_compact_token_limit is None
        assert opts.summarization_prompt == ""
        assert opts.summary_max_retries == 2
        assert opts.summary_max_trims == 20
        assert opts.summary_retry_sleep_ms == 300
        assert opts.token_limit_reached is False
        assert opts.needs_follow_up is False

    def test_custom_values(self):
        opts = CompactorRunOptions(
            auto_compact_ratio=0.8,
            token_limit_reached=True,
            needs_follow_up=True,
            model_auto_compact_token_limit=8000,
        )
        assert opts.auto_compact_ratio == 0.8
        assert opts.token_limit_reached is True
        assert opts.needs_follow_up is True
        assert opts.model_auto_compact_token_limit == 8000


class TestCompactionPayload:
    def test_noop_factory(self):
        p = CompactionPayload.noop(stage="pre_turn")
        assert p.applied is False
        assert p.reason == "noop"
        assert p.stage == "pre_turn"

    def test_noop_custom_reason(self):
        p = CompactionPayload.noop(stage="post_turn", reason="below threshold")
        assert p.reason == "below threshold"

    def test_applied_payload(self):
        p = CompactionPayload(applied=True, reason="compacted", name="summary", stage="pre_turn")
        assert p.applied is True
        assert p.name == "summary"


class TestSummaryBuildResult:
    def test_success_factory(self):
        r = SummaryBuildResult.success(summary_text="summary here", trimmed_count=3, retries=1)
        assert r.ok is True
        assert r.summary_text == "summary here"
        assert r.trimmed_count == 3
        assert r.retries == 1
        assert r.error == ""

    def test_success_defaults(self):
        r = SummaryBuildResult.success(summary_text="s")
        assert r.trimmed_count == 0
        assert r.retries == 0

    def test_failure_factory(self):
        r = SummaryBuildResult.failure(error="timeout", trimmed_count=1, retries=2)
        assert r.ok is False
        assert r.error == "timeout"
        assert r.summary_text == ""
        assert r.trimmed_count == 1
        assert r.retries == 2


class TestCompactionDecision:
    def test_basic(self):
        d = CompactionDecision(apply=True, reason="over budget")
        assert d.apply is True
        assert d.reason == "over budget"
        assert d.metadata == {}

    def test_metadata(self):
        d = CompactionDecision(apply=False, reason="fine", metadata={"used": 100})
        assert d.metadata == {"used": 100}
