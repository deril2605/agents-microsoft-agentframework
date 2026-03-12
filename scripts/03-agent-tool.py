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


# Setup logging
logging.basicConfig(level=logging.WARNING, force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure Azure OpenAI client based on environment
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
client = OpenAIChatClient(
    base_url=normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")),
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
    model_id=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("LLM_MODEL") or "gpt-4.1-mini",
)
STDOUT_ENCODING = sys.stdout.encoding or "utf-8"


@tool
def get_weather(
    city: Annotated[str, Field(description="City name, spelled out fully")],
) -> dict:
    """Returns weather data for a given city, a dictionary with temperature and description."""
    logger.info(f"Getting weather for {city}")
    if random.random() < 0.05:
        return {
            "temperature": 72,
            "description": "Sunny",
        }
    else:
        return {
            "temperature": 60,
            "description": "Rainy",
        }


agent = Agent(
    client=client,
    instructions="You're an informational agent. Answer questions cheerfully.",
    tools=[get_weather],
)


async def main():
    response = await agent.run("how's weather today in sf?")
    print(response.text.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))


if __name__ == "__main__":
    asyncio.run(main())
