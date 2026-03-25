r"""Fan-out/fan-in with LLM summarization aggregation.

This uses the same three expert branches as the explicit fan-in sample, but
instead of a hand-coded template, a summarizer agent synthesizes all branch
outputs into a concise executive brief.

Aggregation technique:
- LLM synthesis as a post-processing step

Run:
    uv run .\scripts\05-orchestratng-multi-agents\02-wf-aggregate-summary.py
    uv run .\scripts\05-orchestratng-multi-agents\02-wf-aggregate-summary.py --devui
"""

import asyncio
import os
import sys
from pathlib import Path

from agent_framework import Agent, AgentExecutorResponse, Executor, WorkflowBuilder, WorkflowContext, handler
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
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


class DispatchPrompt(Executor):
    """Emit the same prompt downstream so fan-out edges can broadcast it."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(prompt)


class SummarizerExecutor(Executor):
    """Fan-in aggregator that synthesizes expert outputs via a wrapped agent."""

    agent: Agent

    def __init__(self, client: OpenAIChatClient, id: str = "Summarizer"):
        super().__init__(id=id)
        self.agent = Agent(
            client=client,
            name=id,
            instructions=(
                "You receive analysis from three domain experts: researcher, marketer, and legal. "
                "Synthesize their combined insights into a concise 3-sentence executive brief "
                "that a CEO could read in 30 seconds. Do not repeat the raw analysis."
            ),
        )

    @handler
    async def run(self, results: list[AgentExecutorResponse], ctx: WorkflowContext[Never, str]) -> None:
        """Format branch outputs and feed them to the summarizer agent."""
        sections = []
        for result in results:
            sections.append(f"[{result.executor_id}]\n{result.agent_response.text}")
        combined = "\n\n---\n\n".join(sections)
        response = await self.agent.run(combined)
        await ctx.yield_output(response.text)


dispatcher = DispatchPrompt(id="dispatcher")

researcher = Agent(
    client=client,
    name="Researcher",
    instructions=(
        "You are an expert market researcher. "
        "Given the prompt, provide concise factual insights, opportunities, and risks. "
        "Use short bullet points."
    ),
)

marketer = Agent(
    client=client,
    name="Marketer",
    instructions=(
        "You are a marketing strategist. "
        "Given the prompt, propose a clear value proposition and audience messaging. "
        "Use short bullet points."
    ),
)

legal = Agent(
    client=client,
    name="Legal",
    instructions=(
        "You are a legal and compliance reviewer. "
        "Given the prompt, list constraints, disclaimers, and policy concerns. "
        "Use short bullet points."
    ),
)

summarizer = SummarizerExecutor(client=client)

workflow = (
    WorkflowBuilder(
        name="FanOutFanInLLMSummary",
        description="Fan-out/fan-in with LLM summarization aggregation.",
        start_executor=dispatcher,
        output_executors=[summarizer],
    )
    .add_fan_out_edges(dispatcher, [researcher, marketer, legal])
    .add_fan_in_edges([researcher, marketer, legal], summarizer)
    .build()
)


async def main() -> None:
    if not api_key:
        safe_print("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = "We are launching a budget-friendly electric bike for urban commuters."
    safe_print(f"Prompt: {prompt}\n")

    events = await workflow.run(prompt)
    for output in events.get_outputs():
        safe_print("=== Executive Brief (LLM-synthesized) ===")
        safe_print(str(output))


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8101, auto_open=True)
    else:
        asyncio.run(main())
