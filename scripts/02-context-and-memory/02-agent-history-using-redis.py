# Ensure redis is running "redis://localhost:6380"
"""
02-agent-history-using-redis.py takes the same idea of a session, but moves the chat history into Redis through RedisHistoryProvider. That changes the role of the session from “memory stored in this Python process” to “pointer to memory stored in an external system.” Because the history is outside the app, a new agent instance can reconnect to the same session_id later and continue the conversation. That makes it much better for real applications where you want durability across restarts, multiple app instances, background workers, containers, or horizontally scaled services. The tradeoff is extra setup and operations: Redis must be running, reachable, and configured correctly.
"""
# Data flow in Redis-backed history
#
#   User / Agent messages
#            |
#            v
#   RedisHistoryProvider.save_messages(...)
#            |
#            v
#   +----------------------------------+
#   | Redis                            |
#   |----------------------------------|
#   | session:abc-123 -> [msg1, msg2]  |
#   | session:def-456 -> [msg1, msg2]  |
#   +----------------------------------+
#            |
#            v
#   create_session(session_id="abc-123")
#            |
#            v
#   RedisHistoryProvider.get_messages(...)
#            |
#            v
#   Restore Message objects and continue chat

import asyncio
import logging
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.redis import RedisHistoryProvider
from dotenv import load_dotenv
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
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

client = OpenAIChatClient(
    base_url=normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")),
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
    model_id=(
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4.1-mini"
    ),
)


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Returns weather data for a given city."""
    logger.info("Getting weather for %s", city)
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {city} is {random.choice(conditions)} with a high of {random.randint(10, 30)} C."


async def example_persistent_session() -> None:
    """A Redis-backed session persists conversation history across application restarts."""
    safe_print("\n=== Persistent Redis Session ===")

    session_id = str(uuid.uuid4())

    # Phase 1: Start a conversation with a Redis-backed history provider
    safe_print("--- Phase 1: Starting conversation ---")
    redis_provider = RedisHistoryProvider(source_id="redis_chat", redis_url=REDIS_URL)

    agent = Agent(
        client=client,
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
        context_providers=[redis_provider],
    )

    session = agent.create_session(session_id=session_id)

    safe_print("User: What's the weather like in Tokyo?")
    response = await agent.run("What's the weather like in Tokyo?", session=session)
    safe_print(f"Agent: {response.text}")

    safe_print("\nUser: How about Paris?")
    response = await agent.run("How about Paris?", session=session)
    safe_print(f"Agent: {response.text}")

    # Phase 2: Simulate an application restart and reconnect using the same session ID in Redis.
    safe_print("\n--- Phase 2: Resuming after 'restart' ---")
    redis_provider2 = RedisHistoryProvider(source_id="redis_chat", redis_url=REDIS_URL)

    agent2 = Agent(
        client=client,
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
        context_providers=[redis_provider2],
    )

    session2 = agent2.create_session(session_id=session_id)

    safe_print("User: Which of the cities I asked about had better weather?")
    response = await agent2.run("Which of the cities I asked about had better weather?", session=session2)
    safe_print(f"Agent: {response.text}")


async def main() -> None:
    """Run all Redis session examples to demonstrate persistent storage patterns."""
    # Verify Redis connectivity
    import redis as redis_client

    r = redis_client.from_url(REDIS_URL)
    try:
        r.ping()
    except Exception as e:
        logger.error(f"Cannot connect to Redis at {REDIS_URL}: {e}")
        logger.error(
            "Ensure Redis is running (e.g. via the dev container"
            " or 'docker run -p 6379:6379 redis:7-alpine')."
        )
        return
    finally:
        r.close()

    await example_persistent_session()

if __name__ == "__main__":
    asyncio.run(main())
