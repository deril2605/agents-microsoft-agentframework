"""
Context isolation with sub-agents.

When an agent delegates tool-heavy work to a sub-agent, the sub-agent's
context window absorbs all the raw tool output. The main agent only sees
the sub-agent's concise summary, keeping its own context window smaller.
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
subagent_usage_log: list[dict] = []


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


research_agent = Agent(
    name="research-agent",
    client=client,
    instructions=(
        "You are a code research assistant. Use the available tools to inspect "
        "Python source files in the project and answer the question. Return a concise "
        "summary of your findings in under 200 words. Do not include raw file contents."
    ),
    tools=[list_project_files, read_project_file, search_project_files],
)


@tool
async def research_codebase(
    question: Annotated[str, Field(description="A research question about the codebase to investigate.")],
) -> str:
    logger.info("[Sub-Agent] Delegating: %s", question[:80])
    response = await research_agent.run(question)

    usage = response.usage_details or {}
    subagent_usage_log.append(usage)
    logger.info(
        "[Sub-Agent] Done. Sub-agent used input=%d output=%d total=%d tokens",
        usage.get("input_token_count", 0) or 0,
        usage.get("output_token_count", 0) or 0,
        usage.get("total_token_count", 0) or 0,
    )
    return response.text or "No findings."


coordinator = Agent(
    name="coordinator",
    client=client,
    instructions=(
        "You are a helpful coding assistant. Answer questions about the codebase. "
        "Use the research_codebase tool to investigate before answering and provide "
        "a clear, well-organized response."
    ),
    tools=[research_codebase],
)

USER_QUERY = "What different middleware patterns are used across this project? Read the relevant files to find out."


async def main() -> None:
    safe_print("\n=== Code Research WITH Sub-Agents (Context Isolation) ===")
    safe_print("The coordinator delegates file reading to a research sub-agent.")
    safe_print("Raw file contents stay in the sub-agent context, not the coordinator.\n")

    subagent_usage_log.clear()

    safe_print(f"User: {USER_QUERY}")
    response = await coordinator.run(USER_QUERY)
    safe_print(f"Coordinator: {response.text}\n")

    coord_usage = response.usage_details or {}
    coord_input = coord_usage.get("input_token_count", 0) or 0
    coord_output = coord_usage.get("output_token_count", 0) or 0
    coord_total = coord_usage.get("total_token_count", 0) or 0

    sub_input = sum((usage.get("input_token_count", 0) or 0) for usage in subagent_usage_log)
    sub_output = sum((usage.get("output_token_count", 0) or 0) for usage in subagent_usage_log)
    sub_total = sum((usage.get("total_token_count", 0) or 0) for usage in subagent_usage_log)

    safe_print("-- Token Usage --")
    safe_print(f"Coordinator tokens: input={coord_input:,} output={coord_output:,} total={coord_total:,}")
    safe_print(f"Sub-agent tokens: input={sub_input:,} output={sub_output:,} total={sub_total:,}\n")
    safe_print("The coordinator's input tokens stay lower because it only sees the sub-agent summary.\n")


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[coordinator], auto_open=True)
    else:
        asyncio.run(main())
