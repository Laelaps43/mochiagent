from agent.core.session.context import SessionContext


def test_session_context_metadata_keeps_model_profile_only():
    context = SessionContext(
        session_id="sess_1",
        model_profile_id="safe",
    )

    data = context.metadata
    assert data.model_profile_id == "safe"
    assert not hasattr(data, "llm_config")
    assert data.context_budget is not None


def test_session_context_from_snapshot_without_llm_config():
    from agent.types import SessionMetadataData, ContextBudget
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()
    context = SessionContext.from_snapshot(
        SessionMetadataData(
            session_id="sess_2",
            state="idle",
            model_profile_id="safe",
            agent_name="mochiclaw",
            context_budget=ContextBudget.zero(),
            created_at=now,
            updated_at=now,
        )
    )
    assert context.model_profile_id == "safe"
    assert context.context_budget.used_tokens == 0
