import asyncio
import logging
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, BaseContextProvider, Message, tool
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from mem0 import AsyncMemory
from pydantic import Field
from rich import print
from rich.logging import RichHandler


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.rstrip("/")
    if stripped.endswith("/openai/v1"):
        return f"{stripped}/"
    if "openai.azure.com" in stripped:
        return f"{stripped}/openai/v1/"
    return stripped


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


# Setup logging
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure OpenAI client based on environment
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
BASE_URL = normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL"))
API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
CHAT_MODEL = (
    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("LLM_MODEL")
    or "gpt-4.1-mini"
)
EMBEDDING_MODEL = (
    os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    or os.getenv("OPENAI_EMBEDDING_MODEL")
    or os.getenv("MEM0_EMBEDDING_MODEL")
    or "text-embedding-3-small"
)
EMBEDDING_DIMS = int(os.getenv("MEM0_EMBEDDING_DIMS") or ("3072" if "large" in EMBEDDING_MODEL else "1536"))

client = OpenAIChatClient(
    base_url=BASE_URL,
    api_key=API_KEY,
    model_id=CHAT_MODEL,
)


class CompatibleMem0ContextProvider(BaseContextProvider):
    """Mem0 provider tuned for the installed OSS client behavior."""

    def __init__(self, *, source_id: str, mem0_client: AsyncMemory, user_id: str, agent_id: str):
        super().__init__(source_id)
        self.mem0_client = mem0_client
        self.user_id = user_id
        self.agent_id = agent_id

    async def before_run(self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]) -> None:
        input_text = "\n".join(msg.text for msg in context.input_messages if msg and msg.text and msg.text.strip())
        if not input_text:
            return

        search_response = await self.mem0_client.search(
            query=input_text,
            user_id=self.user_id,
            agent_id=self.agent_id,
        )
        memories = search_response.get("results", []) if isinstance(search_response, dict) else search_response
        lines = [memory.get("memory", "") for memory in memories if memory.get("memory")]
        if lines:
            context.extend_messages(
                self.source_id,
                [Message(role="user", text="## Memories\n" + "\n".join(lines))],
            )

    async def after_run(self, *, agent: Any, session: Any, context: Any, state: dict[str, Any]) -> None:
        # Storing the user's message alone produces stable fact extraction with this Mem0 OSS version.
        user_messages = [
            {"role": "user", "content": message.text}
            for message in context.input_messages
            if message.role == "user" and message.text and message.text.strip()
        ]
        if user_messages:
            await self.mem0_client.add(
                messages=user_messages,
                user_id=self.user_id,
                agent_id=self.agent_id,
            )


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Returns weather data for a given city."""
    logger.info("Getting weather for %s", city)
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {city} is {random.choice(conditions)} with a high of {random.randint(10, 30)} C."


def build_mem0_config() -> dict:
    if not API_KEY or not BASE_URL:
        raise ValueError(
            "Mem0 requires an OpenAI-compatible endpoint and API key. "
            "Set LLM_BASE_URL/LLM_API_KEY or AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_API_KEY."
        )

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": API_KEY,
                "model": CHAT_MODEL,
                "openai_base_url": BASE_URL,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": API_KEY,
                "model": EMBEDDING_MODEL,
                "embedding_dims": EMBEDDING_DIMS,
                "openai_base_url": BASE_URL,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "embedding_model_dims": EMBEDDING_DIMS,
            },
        },
    }


async def main() -> None:
    """Demonstrate an agent with Mem0 OSS for long-term memory."""
    safe_print("\n=== Agent with Mem0 OSS Memory ===")

    user_id = str(uuid.uuid4())
    mem0_config = build_mem0_config()

    # Data flow in Mem0 dynamic memory
    #
    #   User message / Agent response
    #               |
    #               v
    #   Mem0 extracts durable facts from the conversation
    #               |
    #               v
    #   Facts are embedded and stored in a vector index
    #               |
    #               v
    #   On later prompts, Mem0 retrieves relevant memories
    #   and injects them back into the agent as context.
    mem0_client = await AsyncMemory.from_config(mem0_config)

    provider = CompatibleMem0ContextProvider(
        source_id="mem0_memory",
        agent_id="weather_agent",
        user_id=user_id,
        mem0_client=mem0_client,
    )

    agent = Agent(
        client=client,
        instructions=(
            "You are a helpful weather assistant. Always use any provided memory context when it is relevant. "
            "If stored memories mention user preferences or favorite places, answer from those memories directly."
        ),
        tools=[get_weather],
        context_providers=[provider],
    )

    safe_print("\n--- Step 1: Teaching preferences ---")
    safe_print("User: Remember that my favorite city is Tokyo and I prefer Celsius.")
    response = await agent.run("Remember that my favorite city is Tokyo and I prefer Celsius.")
    safe_print(f"Agent: {response.text}")

    safe_print("\n--- Step 2: Inspecting stored memories ---")
    memories = await mem0_client.get_all(user_id=user_id, agent_id="weather_agent")
    for mem in memories.get("results", []):
        safe_print(f"Stored memory: {mem.get('memory', '')}")

    safe_print("\n--- Step 3: Recalling preferences ---")
    safe_print("User: Based on your stored memories, what's my favorite city and preferred temperature unit?")
    response = await agent.run("Based on your stored memories, what's my favorite city and preferred temperature unit?")
    safe_print(f"Agent: {response.text}")

    safe_print("\n--- Step 4: Tool use with memory ---")
    safe_print("User: What's the weather in my favorite city?")
    response = await agent.run("What's the weather in my favorite city?")
    safe_print(f"Agent: {response.text}")

    safe_print("\n--- Extracted memories ---")
    memories = await mem0_client.get_all(user_id=user_id, agent_id="weather_agent")
    for mem in memories.get("results", []):
        safe_print(f"  - {mem.get('memory', '')}")


if __name__ == "__main__":
    asyncio.run(main())
