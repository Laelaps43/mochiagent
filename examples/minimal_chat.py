import asyncio
import os
import sys
from pathlib import Path
from typing import cast, override

from pydantic import SecretStr

from agent import (
    BaseAgent,
    Event,
    EventType,
    LLMConfig,
    Tool,
    get_agent,
    setup,
    shutdown,
)


class EchoTool(Tool):
    @property
    @override
    def name(self) -> str:
        return "echo"

    @property
    @override
    def description(self) -> str:
        return "Echo back input text."

    @property
    @override
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    @override
    async def execute(self, text: str = "", **kwargs: object) -> object:
        return {"text": text}


class DemoAgent(BaseAgent):
    @property
    @override
    def name(self) -> str:
        return "demo_agent"

    @property
    @override
    def description(self) -> str:
        return "Minimal demo agent."

    @property
    @override
    def skill_directory(self) -> Path | None:
        return None

    @property
    @override
    def allowed_model_profiles(self) -> set[str]:
        return {"openai:gpt-4o-mini"}

    @override
    async def setup(self) -> None:
        self.register_tool(EchoTool())


async def run_once(prompt: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MOCHI_MODEL", "gpt-4o-mini")

    llm_config = LLMConfig(
        adapter="openai_compatible",
        provider="openai",
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        stream=True,
        openai_max_retries=2,
    )

    await setup(
        agents=[DemoAgent()],
        llm_configs=[llm_config],
        max_concurrent=50,
        max_iterations=100,
    )
    agent = get_agent("demo_agent")
    if agent is None:
        raise RuntimeError("demo_agent not found")

    session = await agent.take_session(
        "examples-minimal-chat",
        model_profile_id=f"openai:{model}",
    )
    queue: asyncio.Queue[Event] = asyncio.Queue()

    async def listener(event: Event):
        await queue.put(event)

    session.add_listener(listener)

    try:
        await agent.push_message(session.session_id, prompt)

        while True:
            event = await queue.get()
            if event.type == EventType.PART_CREATED:
                data = event.data or {}
                part_type = data.get("type")
                if part_type == "text":
                    print(data.get("text", ""), end="", flush=True)
                elif part_type == "tool":
                    state_obj = data.get("state")
                    state = (
                        cast(dict[str, object], state_obj).get("status")
                        if isinstance(state_obj, dict)
                        else None
                    )
                    tool_name = data.get("tool", "unknown")
                    print(f"\n[tool] {tool_name} -> {state}")

            if event.type == EventType.LLM_ERROR:
                print(f"\n[LLM_ERROR] {event.data}")
                break

            if event.type == EventType.MESSAGE_DONE:
                print(f"\n[DONE] {event.data}")
                break
    finally:
        session.remove_listener(listener)
        await shutdown()


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "请调用 echo 工具并回复 hello"
    asyncio.run(run_once(prompt))


if __name__ == "__main__":
    main()
