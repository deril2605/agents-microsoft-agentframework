r"""Writer -> Reviewer workflow using SequentialBuilder orchestration.

Demonstrates:
- `SequentialBuilder(participants=[...])` with AI agents
- collecting the final conversation from `workflow.run()` and `get_outputs()`

Each participant receives the full conversation history generated so far.

Reference:
    https://learn.microsoft.com/en-us/agent-framework/workflows/orchestrations/sequential?pivots=programming-language-python

Run:
    uv run .\scripts\04-ai-driven-workflows\03-wf-agents-sequential.py
    uv run .\scripts\04-ai-driven-workflows\03-wf-agents-sequential.py --devui
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatClient
from agent_framework.orchestrations import SequentialBuilder
from dotenv import load_dotenv
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

writer = Agent(
    client=client,
    name="Writer",
    instructions=(
        "You are a concise content writer. "
        "Write a clear, engaging one-paragraph LinkedIn post based on the user's topic. "
        "Focus on accuracy, readability, and a punchy professional tone."
    ),
)

reviewer = Agent(
    client=client,
    name="Reviewer",
    instructions=(
        "You are a thoughtful content reviewer, not a rewriter. "
        "Read the writer's draft and return feedback only. "
        "Do not rewrite, paraphrase, or improve the draft directly. "
        "Give concise, specific feedback on clarity, accuracy, tone, and structure. "
        "Use exactly this format:\n"
        "Strengths:\n- ...\n"
        "Issues:\n- ...\n"
        "Suggestion:\n- ..."
    ),
)

workflow = SequentialBuilder(participants=[writer, reviewer]).build()


async def main() -> None:
    if not api_key:
        logger.error("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = 'Write a one-paragraph LinkedIn post: "The AI workflow mistake almost every team makes."'
    logger.info("Prompt: %s", prompt)

    events = await workflow.run(prompt)
    outputs = events.get_outputs()

    for conversation in outputs:
        logger.info("===== Final Conversation =====")
        messages: list[Message] = conversation
        for index, message in enumerate(messages, start=1):
            author = message.author_name or ("assistant" if message.role == "assistant" else "user")
            logger.info("%02d [%s]\n%s", index, author, message.text)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8096, auto_open=True)
    else:
        asyncio.run(main())
