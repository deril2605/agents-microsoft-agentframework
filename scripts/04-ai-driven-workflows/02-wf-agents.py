r"""Writer -> Reviewer workflow using AI agents as executors.

Demonstrates:
- `Agent` as a `WorkflowBuilder` executor
- direct workflow edges
- collecting output with `workflow.run()` and `get_outputs()`

This example uses the same `WorkflowBuilder + add_edge` pattern as the
RAG-ingest workflow, but with AI agents instead of Python executors.

Run:
    uv run .\scripts\04-ai-driven-workflows\02-wf-agents.py
    uv run .\scripts\04-ai-driven-workflows\02-wf-agents.py --devui
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from agent_framework import Agent, WorkflowBuilder
from agent_framework.openai import OpenAIChatClient
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
        "Write a clear, engaging short article based on the user's topic. "
        "Focus on accuracy and readability."
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

workflow = WorkflowBuilder(start_executor=writer).add_edge(writer, reviewer).build()


async def main() -> None:
    if not api_key:
        logger.error("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = 'Write a 2-sentence LinkedIn post: "Why your AI pilot looks good but fails in production."'
    print(f"Prompt: {prompt}\n")

    events = await workflow.run(prompt)
    for output in events.get_outputs():
        print("===== Output =====")
        print(output)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8092, auto_open=True)
    else:
        asyncio.run(main())
