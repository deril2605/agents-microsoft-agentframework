"""
Context bloat without sub-agents.

This example shows a single agent reading and searching source files
directly. All raw tool output flows into the same agent context window.
"""

import asyncio
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
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


handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
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

PROJECT_DIR = os.path.dirname(__file__)


@tool
def list_project_files(
    directory: Annotated[str, Field(description="Relative directory path within this examples folder.")],
) -> str:
    logger.info("[Tool] list_project_files('%s')", directory)
    target = os.path.join(PROJECT_DIR, directory)
    if not os.path.isdir(target):
        return f"Error: directory '{directory}' not found."
    return "\n".join(sorted(os.listdir(target)))


@tool
def read_project_file(
    filepath: Annotated[str, Field(description="Relative file path within this examples folder.")],
) -> str:
    logger.info("[Tool] read_project_file('%s')", filepath)
    target = os.path.join(PROJECT_DIR, filepath)
    if not os.path.isfile(target):
        return f"Error: file '{filepath}' not found."
    with open(target, encoding="utf-8") as file_handle:
        return file_handle.read()


@tool
def search_project_files(
    query: Annotated[str, Field(description="Case-insensitive text to search for across all .py files.")],
) -> str:
    logger.info("[Tool] search_project_files('%s')", query)
    query_lower = query.lower()
    results: list[str] = []
    for file_path in sorted(glob.glob(os.path.join(PROJECT_DIR, "*.py"))):
        relpath = os.path.relpath(file_path, PROJECT_DIR)
        with open(file_path, encoding="utf-8") as file_handle:
            for lineno, line in enumerate(file_handle, 1):
                if query_lower in line.lower():
                    results.append(f"{relpath}:{lineno}: {line.rstrip()}")
    if not results:
        return f"No matches found for '{query}'."
    if len(results) > 50:
        return "\n".join(results[:50]) + f"\n... ({len(results) - 50} more matches truncated)"
    return "\n".join(results)


agent = Agent(
    name="coding-assistant",
    client=client,
    instructions=(
        "You are a helpful coding assistant. Answer questions about codebases. "
        "Use the available tools to list, read, and search Python source files, "
        "then provide a clear, well-organized answer."
    ),
    tools=[list_project_files, read_project_file, search_project_files],
)

USER_QUERY = "What different middleware patterns are used across this project? Read the relevant files to find out."


async def main() -> None:
    safe_print("\n=== Code Research WITHOUT Sub-Agents ===")
    safe_print("All file contents flow directly into the agent context window.\n")

    safe_print(f"User: {USER_QUERY}")
    response = await agent.run(USER_QUERY)
    safe_print(f"Assistant: {response.text}\n")

    usage = response.usage_details or {}
    input_tokens = usage.get("input_token_count", 0) or 0
    output_tokens = usage.get("output_token_count", 0) or 0
    total_tokens = usage.get("total_token_count", 0) or 0

    safe_print("-- Token Usage --")
    safe_print(f"Assistant tokens: input={input_tokens:,} output={output_tokens:,} total={total_tokens:,}\n")
    safe_print("All raw file contents were in the same agent context window.\n")


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
