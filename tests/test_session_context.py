from __future__ import annotations

from agent.core.message import (
    CompactionMessageInfo,
    Message,
    TextPart,
    TimeInfo,
)
from agent.core.session.context import SessionContext
from agent.types import SessionState, TokenUsage


def _make_context(session_id: str = "sess_1") -> SessionContext:
    return SessionContext(
        session_id=session_id,
        model_profile_id="openai:gpt-4",
        agent_name="test_agent",
    )


def _make_user_input():
    from agent.core.message import UserTextInput

    return UserTextInput(text="hello")


class TestSessionContextInit:
    def test_defaults(self):
        ctx = _make_context()
        assert ctx.session_id == "sess_1"
        assert ctx.state == SessionState.IDLE
        assert ctx.model_profile_id == "openai:gpt-4"
        assert ctx.agent_name == "test_agent"
        assert ctx.messages == []
        assert ctx.current_message is None
        assert ctx.last_compaction_message_id is None


class TestBuildMessages:
    def test_build_user_message(self):
        ctx = _make_context()
        msg = ctx.build_user_message([_make_user_input()])
        assert len(ctx.messages) == 1
        assert msg.session_id == "sess_1"
        assert msg.role == "user"

    def test_build_assistant_message(self):
        ctx = _make_context()
        user_msg = ctx.build_user_message([_make_user_input()])
        asst_msg = ctx.build_assistant_message(
            parent_id=user_msg.message_id,
            provider_id="openai",
            model_id="gpt-4",
        )
        assert len(ctx.messages) == 2
        assert asst_msg.role == "assistant"
        assert ctx.current_message is asst_msg


class TestGetLlmMessages:
    def test_no_compaction_returns_all(self):
        ctx = _make_context()
        _ = ctx.build_user_message([_make_user_input()])
        msgs = ctx.get_llm_messages()
        assert len(msgs) == 1

    def test_with_compaction_returns_from_bookmark(self):
        ctx = _make_context()
        _ = ctx.build_user_message([_make_user_input()])
        _ = ctx.build_user_message([_make_user_input()])
        bookmark = Message.create_compaction(session_id="sess_1", summary="summary")
        ctx.messages.append(bookmark)
        _ = ctx.build_user_message([_make_user_input()])

        msgs = ctx.get_llm_messages()
        assert any(isinstance(m.info, CompactionMessageInfo) for m in msgs)
        assert len(msgs) < 4


class TestApplyCompaction:
    def test_compaction_inserts_bookmark(self):
        ctx = _make_context()
        _ = ctx.build_user_message([_make_user_input()])
        _ = ctx.build_user_message([_make_user_input()])
        bookmark = Message.create_compaction(session_id="sess_1", summary="summary")
        ctx.apply_compaction(bookmark, insert_idx=1)

        assert ctx.messages[0] is bookmark
        assert ctx.last_compaction_message_id == bookmark.message_id


class TestAddPartToCurrent:
    def test_adds_part_when_current_exists(self):
        ctx = _make_context()
        user_msg = ctx.build_user_message([_make_user_input()])
        _ = ctx.build_assistant_message(
            parent_id=user_msg.message_id, provider_id="openai", model_id="gpt-4"
        )
        part = TextPart(session_id="sess_1", message_id="msg_x", text="hi", time=TimeInfo(start=0))
        ctx.add_part_to_current(part)
        assert ctx.current_message is not None
        assert len(ctx.current_message.parts) == 1

    def test_add_part_without_current_no_error(self):
        ctx = _make_context()
        part = TextPart(session_id="sess_1", message_id="msg_x", text="hi", time=TimeInfo(start=0))
        ctx.add_part_to_current(part)


class TestFinishCurrentMessage:
    def test_finish_clears_current(self):
        ctx = _make_context()
        user_msg = ctx.build_user_message([_make_user_input()])
        _ = ctx.build_assistant_message(
            parent_id=user_msg.message_id, provider_id="openai", model_id="gpt-4"
        )
        ctx.finish_current_message(tokens=TokenUsage(input=10, output=5), finish="stop")
        assert ctx.current_message is None

    def test_finish_without_current_no_error(self):
        ctx = _make_context()
        ctx.finish_current_message()


class TestUpdateMethods:
    def test_update_state(self):
        ctx = _make_context()
        ctx.update_state(SessionState.PROCESSING)
        assert ctx.state == SessionState.PROCESSING

    def test_switch_agent(self):
        ctx = _make_context()
        ctx.switch_agent("new_agent")
        assert ctx.agent_name == "new_agent"

    def test_update_model_profile(self):
        ctx = _make_context()
        ctx.update_model_profile("zhipu:glm-4")
        assert ctx.model_profile_id == "zhipu:glm-4"

    def test_update_context_budget(self):
        ctx = _make_context()
        budget = ctx.update_context_budget(
            total_tokens=10000,
            input_tokens=500,
            output_tokens=100,
            reasoning_tokens=0,
            source="provider",
        )
        assert budget.input_tokens == 500
        assert budget.output_tokens == 100
        assert budget.source == "provider"


class TestMetadataAndSnapshot:
    def test_metadata_fields(self):
        ctx = _make_context()
        meta = ctx.metadata
        assert meta.session_id == "sess_1"
        assert meta.agent_name == "test_agent"
        assert meta.model_profile_id == "openai:gpt-4"

    def test_snapshot_contains_messages(self):
        ctx = _make_context()
        _ = ctx.build_user_message([_make_user_input()])
        snap = ctx.snapshot
        assert snap.message_count == 1
        assert len(snap.messages) == 1

    def test_from_snapshot_roundtrip(self):
        ctx = _make_context()
        ctx.update_state(SessionState.PROCESSING)
        meta = ctx.metadata
        restored = SessionContext.from_snapshot(meta)
        assert restored.session_id == "sess_1"
        assert restored.state == SessionState.PROCESSING
        assert restored.agent_name == "test_agent"
