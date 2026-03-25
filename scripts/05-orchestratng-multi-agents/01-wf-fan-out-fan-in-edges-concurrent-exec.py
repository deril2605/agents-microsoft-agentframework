r"""Fan-out/fan-in workflow with explicit edge groups.

Demonstrates:
- `WorkflowBuilder.add_fan_out_edges(...)`
- `WorkflowBuilder.add_fan_in_edges(...)`

A dispatcher sends one prompt to three expert agents in parallel, then an
aggregator receives all branch results as a list and consolidates them into one
structured report.

Run:
    uv run .\scripts\05-orchestratng-multi-agents\01-wf-fan-out-fan-in-edges-concurrent-exec.py
    uv run .\scripts\05-orchestratng-multi-agents\01-wf-fan-out-fan-in-edges-concurrent-exec.py --devui
"""

import asyncio
import os
import sys
from dataclasses import dataclass
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
        """Send one prompt message to all downstream expert branches."""
        await ctx.send_message(prompt)


@dataclass
class AggregatedInsights:
    """Typed container for consolidated expert perspectives."""

    research: str
    marketing: str
    legal: str


class AggregateInsights(Executor):
    """Join fan-in branch outputs and emit one consolidated report."""

    @handler
    async def aggregate(
        self,
        results: list[AgentExecutorResponse],
        ctx: WorkflowContext[Never, str],
    ) -> None:
        """Reduce a list of expert responses to one structured summary."""
        expert_outputs: dict[str, str] = {"research": "", "marketing": "", "legal": ""}

        for result in results:
            executor_id = result.executor_id.lower()
            text = result.agent_response.text
            if "research" in executor_id:
                expert_outputs["research"] = text
            elif "market" in executor_id:
                expert_outputs["marketing"] = text
            elif "legal" in executor_id:
                expert_outputs["legal"] = text

        aggregated = AggregatedInsights(
            research=expert_outputs["research"],
            marketing=expert_outputs["marketing"],
            legal=expert_outputs["legal"],
        )

        consolidated = (
            "=== Consolidated Launch Brief ===\n\n"
            f"Research Findings:\n{aggregated.research}\n\n"
            f"Marketing Angle:\n{aggregated.marketing}\n\n"
            f"Legal/Compliance Notes:\n{aggregated.legal}\n"
        )
        await ctx.yield_output(consolidated)


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

aggregator = AggregateInsights(id="aggregator")

workflow = (
    WorkflowBuilder(
        name="FanOutFanInEdges",
        description="Explicit fan-out/fan-in using edge groups.",
        start_executor=dispatcher,
        output_executors=[aggregator],
    )
    .add_fan_out_edges(dispatcher, [researcher, marketer, legal])
    .add_fan_in_edges([researcher, marketer, legal], aggregator)
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
        safe_print(str(output))


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8097, auto_open=True)
    else:
        asyncio.run(main())
