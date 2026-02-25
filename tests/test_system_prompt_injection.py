from types import SimpleNamespace

from agent.core.prompt import inject_system_prompt


class _DummyAgent:
    def __init__(self, text: str | None):
        self._text = text

    def get_system_prompt(self, context) -> str | None:
        return self._text


def _dummy_context():
    return SimpleNamespace(
        session_id="s1",
        agent_name="analytics_agent",
        llm_config=SimpleNamespace(provider="openai", model="test-model"),
    )


def test_inject_system_prompt_replaces_existing_system_messages():
    messages = [
        {"role": "system", "content": "old-1"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "old-2"},
    ]

    out = inject_system_prompt(messages, "new-system")

    assert out[0] == {"role": "system", "content": "new-system"}
    assert [m["role"] for m in out].count("system") == 1
    assert out[1:] == [{"role": "user", "content": "hi"}]


def test_inject_system_prompt_noop_when_agent_returns_empty():
    context = _dummy_context()
    messages = [{"role": "user", "content": "hi"}]
    out = inject_system_prompt(messages, _DummyAgent(None).get_system_prompt(context))
    assert out == messages
