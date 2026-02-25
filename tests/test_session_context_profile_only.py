from agent.core.session.context import SessionContext


def test_session_context_metadata_keeps_model_profile_only():
    context = SessionContext(
        session_id="sess_1",
        model_profile_id="safe",
    )

    data = context.to_metadata_dict()
    assert data["model_profile_id"] == "safe"
    assert "llm_config" not in data


def test_session_context_from_dict_without_llm_config():
    context = SessionContext.from_dict(
        {
            "session_id": "sess_2",
            "state": "idle",
            "model_profile_id": "safe",
            "agent_name": "mochiclaw",
            "metadata": {},
        }
    )
    assert context.model_profile_id == "safe"
