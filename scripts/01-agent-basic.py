import asyncio
import os
import sys
from pathlib import Path

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.rstrip("/")
    if stripped.endswith("/openai/v1"):
        return f"{stripped}/"
    if "openai.azure.com" in stripped:
        return f"{stripped}/openai/v1/"
    return stripped


# Configure Azure OpenAI client based on environment
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")
azure_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("LLM_MODEL")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")

client = OpenAIChatClient(
    base_url=normalize_base_url(azure_endpoint),
    api_key=azure_api_key,
    model_id=azure_model,
)

agent = Agent(client=client, instructions="You're an informational agent. Answer questions cheerfully.")


async def main():
    response = await agent.run("Whats weather today in San Francisco?")
    stdout_encoding = sys.stdout.encoding or "utf-8"
    print(response.text.encode(stdout_encoding, errors="replace").decode(stdout_encoding))


if __name__ == "__main__":
    asyncio.run(main())
