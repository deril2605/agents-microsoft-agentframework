import asyncio
import logging
import os
import sys
from pathlib import Path

from agent_framework import Agent, MCPStreamableHTTPTool
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


async def main() -> None:
    """Run an agent that uses a remote MCP server (Microsoft Learn) to answer documentation questions."""
    async with (
        MCPStreamableHTTPTool(
            name="Microsoft Learn MCP",
            url="https://learn.microsoft.com/api/mcp",
        ) as mcp_server,
        Agent(
            client=client,
            instructions=(
                "You help with Microsoft documentation questions. "
                "Use the available tools to search for relevant docs."
            ),
            tools=[mcp_server],
        ) as agent,
    ):
        response = await agent.run("How do I create an Azure storage account using az cli?")
        print(response.text.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))


if __name__ == "__main__":
    asyncio.run(main())
