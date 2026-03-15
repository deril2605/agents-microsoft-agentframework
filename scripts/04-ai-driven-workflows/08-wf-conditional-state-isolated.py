r"""Writer -> Reviewer workflow with structured outputs, state, and state isolation.

Demonstrates:
- structured reviewer decisions
- conditional edges
- iterative review loops
- explicit state management with `WorkflowContext.set_state()` / `get_state()`
- per-request state isolation via a workflow factory helper

Routing:
- `decision == APPROVED` -> publisher
- `decision == REVISION_NEEDED` -> editor -> reviewer

State isolation best practice:
- build a fresh workflow and fresh agents per request by calling `create_workflow(...)`
- keep agent thread state and workflow state from leaking across independent runs

Run:
    uv run .\scripts\04-ai-driven-workflows\08-wf-conditional-state-isolated.py
    uv run .\scripts\04-ai-driven-workflows\08-wf-conditional-state-isolated.py --devui
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


def parse_review_decision(message: Any) -> ReviewDecision | None:
    """Parse structured reviewer output from AgentExecutorResponse."""
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


def create_workflow(model_client: OpenAIChatClient):
    """Create a fresh workflow instance with isolated agent and workflow state."""
    writer = Agent(
        client=model_client,
        name="Writer",
        instructions=(
            "You are a concise content writer. "
            "Write a clear, engaging short LinkedIn post based on the user's topic."
        ),
    )

    reviewer = Agent(
        client=model_client,
        name="Reviewer",
        instructions=(
            "You are a strict content reviewer. Evaluate the writer's draft.\n"
            "Check that the post is engaging and a good fit for LinkedIn.\n"
            "Make sure that it does not sound overly AI-generated or overconfident.\n"
            "Do not use em dashes or fancy Unicode characters.\n"
            "Return a structured decision with decision and feedback.\n"
            "Set decision=APPROVED if the draft is clear, accurate, and well-structured.\n"
            "Set decision=REVISION_NEEDED if it requires improvement.\n"
            "In feedback, explain your reasoning briefly and provide actionable edits when needed."
        ),
        default_options={"response_format": ReviewDecision},
    )

    editor = Agent(
        client=model_client,
        name="Editor",
        instructions=(
            "You are a skilled editor. "
            "You receive a writer's draft followed by the reviewer's feedback. "
            "Rewrite the draft to address all issues raised in the feedback. "
            "Output only the improved post."
        ),
    )

    @executor(id="store_post_text")
    async def store_post_text(response: AgentExecutorResponse, ctx: WorkflowContext[AgentExecutorResponse]) -> None:
        """Persist the latest post text in workflow state and pass it downstream."""
        ctx.set_state("post_text", response.agent_response.text.strip())
        await ctx.send_message(response)

    @executor(id="publisher")
    async def publisher(_response: AgentExecutorResponse, ctx: WorkflowContext[Never, str]) -> None:
        """Publish the latest approved post text from workflow state."""
        content = str(ctx.get_state("post_text", "")).strip()
        await ctx.yield_output(f"Published:\n\n{content}")

    return (
        WorkflowBuilder(start_executor=writer, max_iterations=8)
        .add_edge(writer, store_post_text)
        .add_edge(store_post_text, reviewer)
        .add_edge(reviewer, publisher, condition=is_approved)
        .add_edge(reviewer, editor, condition=needs_revision)
        .add_edge(editor, store_post_text)
        .build()
    )


async def main() -> None:
    if not api_key:
        logger.error("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = "Write a LinkedIn post predicting the 5 jobs AI agents will replace by December 2026."
    safe_print(f"Prompt: {prompt}\n")

    workflow = create_workflow(client)
    events = await workflow.run(prompt)
    for output in events.get_outputs():
        safe_print(str(output))


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[create_workflow(client)], port=8097, auto_open=True)
    else:
        asyncio.run(main())
