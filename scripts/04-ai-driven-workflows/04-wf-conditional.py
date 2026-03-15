r"""Writer -> Reviewer workflow with conditional edges based on a sentinel token.

Demonstrates:
- conditional edges
- condition functions that inspect `AgentExecutorResponse`
- a terminal `@executor` node

The reviewer is instructed to begin its response with exactly `APPROVED`
or `REVISION NEEDED`. Two outgoing edges route the flow accordingly:
- `APPROVED` -> publisher
- `REVISION NEEDED` -> editor

This is the minimal branching version focused on conditional edges.

Run:
    uv run .\scripts\04-ai-driven-workflows\04-wf-conditional.py
    uv run .\scripts\04-ai-driven-workflows\04-wf-conditional.py --devui
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from agent_framework import Agent, AgentExecutorResponse, WorkflowBuilder, WorkflowContext, executor
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from rich.logging import RichHandler
from typing_extensions import Never


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.rstrip("/")
    if stripped.endswith("/openai/v1"):
        return f"{stripped}/"
    if "openai.azure.com" in stripped:
        return f"{stripped}/openai/v1/"
    return stripped


log_handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[log_handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

base_url = normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL"))
api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
model_id = (
    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("LLM_MODEL")
    or "gpt-4.1-mini"
)

client = OpenAIChatClient(
    base_url=base_url,
    api_key=api_key,
    model_id=model_id,
)


def is_approved(message: Any) -> bool:
    """Route to publisher if the reviewer starts with APPROVED."""
    if not isinstance(message, AgentExecutorResponse):
        return False
    return message.agent_response.text.upper().startswith("APPROVED")


def needs_revision(message: Any) -> bool:
    """Route to editor if the reviewer requests changes."""
    if not isinstance(message, AgentExecutorResponse):
        return False
    return message.agent_response.text.upper().startswith("REVISION NEEDED")


writer = Agent(
    client=client,
    name="Writer",
    instructions=(
        "You are a concise content writer. "
        "Write a clear, engaging short LinkedIn post based on the user's topic. "
        "Keep it grounded, readable, and not overly dramatic."
    ),
)

reviewer = Agent(
    client=client,
    name="Reviewer",
    instructions=(
        "You are a strict content reviewer. Evaluate the writer's draft.\n"
        "Check whether the post is engaging, credible, and a good fit for LinkedIn.\n"
        "Make sure it does not sound overly AI-generated or overconfident.\n"
        "Do not use em dashes or fancy Unicode characters.\n"
        "Your response must begin with exactly one of these tokens:\n"
        "APPROVED\n"
        "REVISION NEEDED\n"
        "If you choose APPROVED, include the final publishable post immediately after the token.\n"
        "If you choose REVISION NEEDED, briefly explain what should be fixed."
    ),
)

editor = Agent(
    client=client,
    name="Editor",
    instructions=(
        "You are a skilled editor. "
        "You receive the writer's draft followed by the reviewer's feedback. "
        "Rewrite the draft so it addresses the reviewer's concerns. "
        "Output only the improved LinkedIn post."
    ),
)


@executor(id="publisher")
async def publisher(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """Strip the APPROVED prefix and yield the publishable content."""
    text = response.agent_response.text
    content = text[len("APPROVED") :].lstrip(":").strip()
    await ctx.yield_output(f"Published:\n\n{content}")


workflow = (
    WorkflowBuilder(start_executor=writer)
    .add_edge(writer, reviewer)
    .add_edge(reviewer, publisher, condition=is_approved)
    .add_edge(reviewer, editor, condition=needs_revision)
    .build()
)


async def main() -> None:
    if not api_key:
        logger.error("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = "Write a LinkedIn post predicting the 5 jobs AI agents will replace by December 2026."
    print(f"Prompt: {prompt}\n")

    events = await workflow.run(prompt)
    for output in events.get_outputs():
        print(output)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8094, auto_open=True)
    else:
        asyncio.run(main())
