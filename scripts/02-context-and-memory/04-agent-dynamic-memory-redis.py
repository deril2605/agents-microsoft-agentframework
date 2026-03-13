"""
For redis stack commands below
docker stop redis-agentframework
docker rm redis-agentframework
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
"""

"""
Every message is stored as memeory in redis and later searched on when user asks something
"""

import asyncio
import logging
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Annotated

import redis as redis_client
from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.redis import RedisContextProvider
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


def verify_redis_stack() -> bool:
    """Verify that Redis is reachable and has the RediSearch module loaded."""
    redis_connection = redis_client.from_url(REDIS_URL)
    try:
        redis_connection.ping()
        modules = redis_connection.execute_command("MODULE", "LIST")
        has_search = any(
            any(str(item).lower() in {"search", "ft"} for item in module if not isinstance(item, bytes))
            or any(part.decode("utf-8", errors="ignore").lower() in {"search", "ft"} for part in module if isinstance(part, bytes))
            for module in modules
        )
        if not has_search:
            logger.error("Redis is running at %s, but the RediSearch module is not loaded.", REDIS_URL)
            logger.error(
                "This demo needs Redis Stack. Example: docker run -d --name redis-stack "
                "-p 6379:6379 redis/redis-stack:latest"
            )
            return False
        return True
    except Exception as exc:
        logger.error("Cannot connect to Redis at %s: %s", REDIS_URL, exc)
        logger.error(
            "Start Redis Stack first. Example: docker run -d --name redis-stack "
            "-p 6379:6379 redis/redis-stack:latest"
        )
        return False
    finally:
        redis_connection.close()


async def example_agent_with_memory() -> None:
    """Demonstrate an agent with Redis-backed long-term memory via RedisContextProvider."""
    safe_print("\n=== Agent with Redis Memory (RedisContextProvider) ===")

    user_id = str(uuid.uuid4())

    # Data flow in Redis dynamic memory
    #
    #   User message / Agent response
    #               |
    #               v
    #   RedisContextProvider stores each message in Redis
    #               |
    #               v
    #   RediSearch index enables lookup by meaning / keyword match
    #               |
    #               v
    #   On a later prompt, matching memories are retrieved
    #   and injected back into the agent as extra context.
    memory_provider = RedisContextProvider(
        source_id="redis_memory",
        redis_url=REDIS_URL,
        index_name="agent_memory_demo",
        prefix="memory_demo",
        application_id="weather_app",
        agent_id="weather_agent",
        user_id=user_id,
        overwrite_index=True,
    )

    agent = Agent(
        client=client,
        instructions=(
            "You are a helpful weather assistant. Personalize replies using provided context. "
            "Before answering, always check for stored context."
        ),
        tools=[get_weather],
        context_providers=[memory_provider],
    )

    safe_print("\n--- Step 1: Teaching a preference ---")
    safe_print("User: Remember that my favorite city is Tokyo.")
    response = await agent.run("Remember that my favorite city is Tokyo.")
    safe_print(f"Agent: {response.text}")

    safe_print("\n--- Step 2: Recalling a preference ---")
    safe_print("User: What's my favorite city?")
    response = await agent.run("What's my favorite city?")
    safe_print(f"Agent: {response.text}")

    safe_print("\n--- Step 3: Tool use with memory ---")
    safe_print("User: What's the weather in Paris?")
    response = await agent.run("What's the weather in Paris?")
    safe_print(f"Agent: {response.text}")

    safe_print("\nUser: What city did I just ask about and what was the weather?")
    response = await agent.run("What city did I just ask about and what was the weather?")
    safe_print(f"Agent: {response.text}")


async def main() -> None:
    if not verify_redis_stack():
        return
    await example_agent_with_memory()


if __name__ == "__main__":
    asyncio.run(main())
