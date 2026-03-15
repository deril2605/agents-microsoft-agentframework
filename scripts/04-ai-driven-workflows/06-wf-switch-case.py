r"""Customer message router using structured outputs and switch-case edges.

Demonstrates:
- `response_format=` for reliable structured output
- `@executor` for a converter node
- `add_switch_case_edge_group` for multi-way routing

A classifier agent uses a Pydantic model as its `response_format`, so the
category is always a valid typed value instead of relying on fragile string
matching. A converter executor extracts the structured result, then switch-case
edges route to a specialized handler for each category.

Pipeline:
    Classifier -> extract_category -> [Question  -> handle_question]
                                   -> [Complaint -> handle_complaint]
                                   -> [Default   -> handle_feedback]

Run:
    uv run .\scripts\04-ai-driven-workflows\06-wf-switch-case.py
    uv run .\scripts\04-ai-driven-workflows\06-wf-switch-case.py --devui
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

from agent_framework import Agent, AgentExecutorResponse, Case, Default, WorkflowBuilder, WorkflowContext, executor
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


class ClassifyResult(BaseModel):
    """Structured classification result from the classifier agent."""

    category: Literal["Question", "Complaint", "Feedback"]
    original_message: str
    reasoning: str


classifier = Agent(
    client=client,
    name="Classifier",
    instructions=(
        "You are a customer message classifier. "
        "Classify the incoming customer message into exactly one category: "
        "Question, Complaint, or Feedback. "
        "Return JSON with category, original_message, and reasoning."
    ),
    default_options={"response_format": ClassifyResult},
)


@executor(id="extract_category")
async def extract_category(response: AgentExecutorResponse, ctx: WorkflowContext[ClassifyResult]) -> None:
    """Parse the classifier's structured JSON output and send it downstream."""
    try:
        result = ClassifyResult.model_validate_json(response.agent_response.text)
    except ValidationError as exc:
        raise ValueError(f"Unable to parse classifier output: {response.agent_response.text}") from exc

    logger.info("-> Classified as %s: %s", result.category, result.reasoning)
    await ctx.send_message(result)


def is_question(msg: Any) -> bool:
    return isinstance(msg, ClassifyResult) and msg.category == "Question"


def is_complaint(msg: Any) -> bool:
    return isinstance(msg, ClassifyResult) and msg.category == "Complaint"


@executor(id="handle_question")
async def handle_question(result: ClassifyResult, ctx: WorkflowContext[Never, str]) -> None:
    """Route a question to the Q&A team."""
    await ctx.yield_output(
        "Question routed to Q&A team\n\n"
        f"Message: {result.original_message}\n"
        f"Reason: {result.reasoning}"
    )


@executor(id="handle_complaint")
async def handle_complaint(result: ClassifyResult, ctx: WorkflowContext[Never, str]) -> None:
    """Escalate a complaint to the support team."""
    await ctx.yield_output(
        "Complaint escalated to support team\n\n"
        f"Message: {result.original_message}\n"
        f"Reason: {result.reasoning}"
    )


@executor(id="handle_feedback")
async def handle_feedback(result: ClassifyResult, ctx: WorkflowContext[Never, str]) -> None:
    """Forward feedback to the product team."""
    await ctx.yield_output(
        "Feedback forwarded to product team\n\n"
        f"Message: {result.original_message}\n"
        f"Reason: {result.reasoning}"
    )


workflow = (
    WorkflowBuilder(start_executor=classifier)
    .add_edge(classifier, extract_category)
    .add_switch_case_edge_group(
        extract_category,
        [
            Case(condition=is_question, target=handle_question),
            Case(condition=is_complaint, target=handle_complaint),
            Default(target=handle_feedback),
        ],
    )
    .build()
)


async def main() -> None:
    if not api_key:
        logger.error("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    message = "How do I reset my password?"
    safe_print(f"Customer message: {message}\n")

    events = await workflow.run(message)
    for output in events.get_outputs():
        safe_print(str(output))


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8095, auto_open=True)
    else:
        asyncio.run(main())
