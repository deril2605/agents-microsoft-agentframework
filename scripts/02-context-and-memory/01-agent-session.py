"""
01-agent-session.py is the lightweight, in-memory approach. The conversation state lives inside the Python app itself, attached to the session object you create with agent.create_session(). That means it’s great for demos, local experiments, short-lived chats, and cases where one process handles the whole interaction. It’s simple because there’s no extra infrastructure, no Redis container, and no external storage layer. The tradeoff is that the memory only exists while that script is running. If you stop the script, restart your terminal, redeploy the app, or scale to another process, the session history is gone unless you save it somewhere else yourself.
"""

import asyncio
import logging
import os
import random
import sys
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from pydantic import Field


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


logging.basicConfig(level=logging.WARNING, force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
client = OpenAIChatClient(
    base_url=normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")),
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
    model_id=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("LLM_MODEL") or "gpt-4.1-mini",
)


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Returns weather data for a given city."""
    logger.info("Getting weather for %s", city)
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {city} is {random.choice(conditions)} with a high of {random.randint(10, 30)} C."


agent = Agent(
    client=client,
    instructions="You are a helpful weather agent.",
    tools=[get_weather],
)


async def example_without_session() -> None:
    """Without a session, each call is independent and has no memory of prior messages."""
    safe_print("\n=== Without Session (No Memory) ===")

    safe_print("User: What's the weather like in Seattle?")
    response = await agent.run("What's the weather like in Seattle?")
    safe_print(f"Agent: {response.text}")

    safe_print("\nUser: What was the last city I asked about?")
    response = await agent.run("What was the last city I asked about?")
    safe_print(f"Agent: {response.text}")


async def example_with_session() -> None:
    """With a session, the agent maintains context across multiple messages."""
    safe_print("\n=== With Session (Persistent Memory) ===")

    session = agent.create_session()

    safe_print("User: What's the weather like in Tokyo?")
    response = await agent.run("What's the weather like in Tokyo?", session=session)
    safe_print(f"Agent: {response.text}")

    safe_print("\nUser: How about London?")
    response = await agent.run("How about London?", session=session)
    safe_print(f"Agent: {response.text}")

    safe_print("\nUser: Which of those cities has better weather?")
    response = await agent.run("Which of those cities has better weather?", session=session)
    safe_print(f"Agent: {response.text}")


async def example_session_across_agents() -> None:
    """A session can be shared across different agent instances."""
    safe_print("\n=== Session Across Agent Instances ===")

    session = agent.create_session()

    safe_print("User: What's the weather in Paris?")
    response = await agent.run("What's the weather in Paris?", session=session)
    safe_print(f"Agent 1: {response.text}")

    agent2 = Agent(
        client=client,
        instructions="You are a helpful weather agent.",
        tools=[get_weather],
    )

    safe_print("\nUser: What was the last city I asked about?")
    response = await agent2.run("What was the last city I asked about?", session=session)
    safe_print(f"Agent 2: {response.text}")


async def main() -> None:
    """Run all session examples to demonstrate different persistence patterns."""
    await example_without_session()
    await example_with_session()
    await example_session_across_agents()


if __name__ == "__main__":
    asyncio.run(main())
