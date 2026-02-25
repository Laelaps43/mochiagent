import asyncio
import os
import sys
from pathlib import Path

from agent import BaseAgent, Event, EventType, LLMConfig, Tool, get_agent, setup, shutdown


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back input text."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, text: str):
        return {"text": text}


class DemoAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "demo_agent"

    @property
    def description(self) -> str:
        return "Minimal demo agent."

    @property
    def skill_directory(self) -> Path | None:
        return None

    async def setup(self) -> None:
        self.register_tool(EchoTool())


async def run_once(prompt: str) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("MOCHI_MODEL", "gpt-4o-mini")

    await setup(agents=[DemoAgent()], max_concurrent=50, max_iterations=100)
    agent = get_agent("demo_agent")
    if agent is None:
        raise RuntimeError("demo_agent not found")

    llm_config = LLMConfig(
        provider="openai",
        model=model,
        api_key=api_key,
        base_url=base_url,
        stream=True,
        openai_max_retries=2,
    )

    session = await agent.take_session("examples-minimal-chat", llm_config)
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
                    state = (data.get("state") or {}).get("status")
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
