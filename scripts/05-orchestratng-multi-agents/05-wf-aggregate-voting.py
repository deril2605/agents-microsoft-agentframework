r"""Fan-out/fan-in with majority-vote aggregation.

Three classifier agents use different reasoning strategies to independently
categorize a support ticket. The fan-in aggregator tallies votes and picks the
majority label.

Aggregation technique:
- majority vote with pure logic in the aggregator

Note:
- in production, using different models per branch would strengthen the
  ensemble
- here we simulate diversity with different prompting strategies on the same
  model

Run:
    uv run .\scripts\05-orchestratng-multi-agents\05-wf-aggregate-voting.py
    uv run .\scripts\05-orchestratng-multi-agents\05-wf-aggregate-voting.py --devui
"""

import asyncio
import os
import sys
from collections import Counter
from enum import Enum
from pathlib import Path

from agent_framework import Agent, AgentExecutorResponse, Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from pydantic import BaseModel
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


class Category(str, Enum):
    BUG = "bug"
    BILLING = "billing"
    FEATURE_REQUEST = "feature_request"
    GENERAL = "general"


class Classification(BaseModel):
    """Structured output for each classifier agent."""

    category: Category


class DispatchPrompt(Executor):
    """Emit the ticket text downstream for fan-out broadcast."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(prompt)


class TallyVotes(Executor):
    """Fan-in aggregator that counts votes and picks the majority label."""

    @handler
    async def tally(
        self,
        results: list[AgentExecutorResponse],
        ctx: WorkflowContext[Never, str],
    ) -> None:
        """Count classifier votes and yield the winning category."""
        votes: list[tuple[str, str]] = []
        for result in results:
            classification = Classification.model_validate_json(result.agent_response.text)
            votes.append((result.executor_id, classification.category.value))

        labels = [label for _, label in votes]
        counter = Counter(labels)
        winner, count = counter.most_common(1)[0]

        report = f"Result: {winner} ({count}/{len(votes)} votes)\n"
        for agent_id, label in votes:
            report += f"  {agent_id}: {label}\n"
        await ctx.yield_output(report)


dispatcher = DispatchPrompt(id="dispatcher")

keyword_classifier = Agent(
    client=client,
    name="KeywordClassifier",
    instructions=(
        "Classify the support ticket into exactly one category: bug, billing, feature_request, or general.\n"
        "Rules:\n"
        "- If the message mentions error, crash, bug, broken, or fail -> bug\n"
        "- If the message mentions invoice, charge, payment, refund, or subscription -> billing\n"
        "- If the message mentions add, wish, suggest, request, or would be nice -> feature_request\n"
        "- Otherwise -> general"
    ),
    default_options={"response_format": Classification},
)

sentiment_classifier = Agent(
    client=client,
    name="SentimentClassifier",
    instructions=(
        "Classify the support ticket into exactly one category: bug, billing, feature_request, or general.\n"
        "Analyze the emotional tone:\n"
        "- Frustrated or angry about something not working -> bug\n"
        "- Confused or upset about money or charges -> billing\n"
        "- Enthusiastic or hopeful about new capabilities -> feature_request\n"
        "- Neutral informational inquiry -> general"
    ),
    default_options={"response_format": Classification},
)

intent_classifier = Agent(
    client=client,
    name="IntentClassifier",
    instructions=(
        "Classify the support ticket into exactly one category: bug, billing, feature_request, or general.\n"
        "Focus on what the user wants to accomplish:\n"
        "- Wants something fixed or repaired -> bug\n"
        "- Wants a refund, explanation of charges, or account adjustment -> billing\n"
        "- Wants a new feature or improvement -> feature_request\n"
        "- Wants general information or has a question -> general"
    ),
    default_options={"response_format": Classification},
)

tally = TallyVotes(id="tally")

workflow = (
    WorkflowBuilder(
        name="FanOutFanInVoting",
        description="Ensemble classification with majority-vote aggregation.",
        start_executor=dispatcher,
        output_executors=[tally],
    )
    .add_fan_out_edges(dispatcher, [keyword_classifier, sentiment_classifier, intent_classifier])
    .add_fan_in_edges([keyword_classifier, sentiment_classifier, intent_classifier], tally)
    .build()
)


async def main() -> None:
    if not api_key:
        safe_print("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    samples = [
        "The app crashes every time I try to upload a photo. Error code 500.",
        "I wish the export button actually worked. Please add a fix - I'm losing data daily!",
        "The current search fails on long queries - it would be amazing if you could add fuzzy matching.",
    ]

    for sample in samples:
        safe_print(f"Ticket: {sample}")
        events = await workflow.run(sample)
        for output in events.get_outputs():
            safe_print(str(output))


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8103, auto_open=True)
    else:
        asyncio.run(main())
