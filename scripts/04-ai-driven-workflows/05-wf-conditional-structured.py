r"""Writer -> Reviewer workflow using structured outputs with conditional edges.

Demonstrates:
- `response_format` for typed reviewer decisions
- conditional edges
- a terminal `@executor` publisher node

This is a direct contrast with the string-sentinel conditional workflow:
- same branching shape via `add_edge(..., condition=...)`
- different decision mechanism using typed JSON

Routing:
- `decision == APPROVED` -> publisher
- `decision == REVISION_NEEDED` -> editor

Run:
    uv run .\scripts\04-ai-driven-workflows\05-wf-conditional-structured.py
    uv run .\scripts\04-ai-driven-workflows\05-wf-conditional-structured.py --devui
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

from agent_framework import Agent, AgentExecutorResponse, WorkflowBuilder, WorkflowContext, executor
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
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


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


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


class ReviewDecision(BaseModel):
    """Structured reviewer decision used for conditional routing."""

    decision: Literal["APPROVED", "REVISION_NEEDED"]
    feedback: str
    post_text: str | None = None


def parse_review_decision(message: Any) -> ReviewDecision | None:
    """Parse structured reviewer output from an AgentExecutorResponse."""
    if not isinstance(message, AgentExecutorResponse):
        return None

    try:
        return ReviewDecision.model_validate_json(message.agent_response.text)
    except ValidationError:
        return None


def is_approved(message: Any) -> bool:
    """Route to publisher when structured decision is APPROVED."""
    result = parse_review_decision(message)
    return result is not None and result.decision == "APPROVED"


def needs_revision(message: Any) -> bool:
    """Route to editor when structured decision is REVISION_NEEDED."""
    result = parse_review_decision(message)
    return result is not None and result.decision == "REVISION_NEEDED"


writer = Agent(
    client=client,
    name="Writer",
    instructions=(
        "You are a concise content writer. "
        "Write a clear, engaging short LinkedIn post based on the user's topic."
    ),
)

reviewer = Agent(
    client=client,
    name="Reviewer",
    instructions=(
        "You are a strict content reviewer. Evaluate the writer's draft. "
        "If the draft is ready, set decision=APPROVED and include the publishable post in post_text. "
        "If it needs changes, set decision=REVISION_NEEDED and provide actionable feedback in feedback. "
        "Always return valid JSON matching the ReviewDecision schema."
    ),
    default_options={"response_format": ReviewDecision},
)

editor = Agent(
    client=client,
    name="Editor",
    instructions=(
        "You are a skilled editor. "
        "You receive a writer's draft followed by the reviewer's feedback. "
        "Rewrite the draft to address all issues raised in the feedback. "
        "Output only the improved post."
    ),
)


@executor(id="publisher")
async def publisher(response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
    """Publish content from structured reviewer output."""
    result = parse_review_decision(response)
    if result is None:
        await ctx.yield_output("Published:\n\n(Unable to parse structured reviewer output.)")
        return

    content = (result.post_text or "").strip()
    if not content:
        content = "(Reviewer approved but did not provide post_text.)"

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
        safe_print("Output:")
        safe_print(str(output))


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8096, auto_open=True)
    else:
        asyncio.run(main())
