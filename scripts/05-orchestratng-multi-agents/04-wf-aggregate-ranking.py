r"""Fan-out/fan-in with LLM-as-judge ranking aggregation.

Three creative agents with different personas produce marketing slogans. A
ranker executor collects the candidates, formats them, and uses the LLM as a
judge to score and rank them by creativity, memorability, clarity, and brand
fit.

Aggregation technique:
- LLM-as-judge over parallel candidates

Run:
    uv run .\scripts\05-orchestratng-multi-agents\04-wf-aggregate-ranking.py
    uv run .\scripts\05-orchestratng-multi-agents\04-wf-aggregate-ranking.py --devui
"""

import asyncio
import os
import sys
from pathlib import Path

from agent_framework import Agent, AgentExecutorResponse, Executor, Message, WorkflowBuilder, WorkflowContext, handler
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
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


class RankedSlogan(BaseModel):
    """A single ranked slogan entry."""

    rank: int = Field(description="Rank position, 1 = best.")
    agent_name: str = Field(description="Name of the agent that produced the slogan.")
    slogan: str = Field(description="The marketing slogan text.")
    score: int = Field(description="Score from 1 to 10.")
    justification: str = Field(description="One-sentence justification for the score.")


class RankedSlogans(BaseModel):
    """Typed output: a ranked list of slogans."""

    rankings: list[RankedSlogan] = Field(description="Slogans ranked from best to worst.")


class DispatchPrompt(Executor):
    """Emit the product brief downstream for fan-out broadcast."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(prompt)


class RankerExecutor(Executor):
    """Fan-in aggregator that ranks candidate slogans with the LLM."""

    def __init__(self, *, client: OpenAIChatClient, id: str = "Ranker") -> None:
        super().__init__(id=id)
        self._client = client

    @handler
    async def run(
        self,
        results: list[AgentExecutorResponse],
        ctx: WorkflowContext[Never, RankedSlogans],
    ) -> None:
        """Collect slogans, format them, and ask the LLM to rank them."""
        lines = []
        for result in results:
            slogan = result.agent_response.text.strip().strip("\"'").split("\n")[0].strip().strip("\"'")
            lines.append(f'- [{result.executor_id}]: "{slogan}"')

        messages = [
            Message(
                role="system",
                text=(
                    "You are a senior creative director judging marketing slogans. "
                    "Given a list of candidate slogans, rank them from best to worst. "
                    "For each slogan, give a 1-10 score and a one-sentence justification "
                    "evaluating creativity, memorability, clarity, and brand fit."
                ),
            ),
            Message(role="user", text="Candidate slogans:\n" + "\n".join(lines)),
        ]
        response = await self._client.get_response(messages, options={"response_format": RankedSlogans})
        await ctx.yield_output(response.value)


dispatcher = DispatchPrompt(id="dispatcher")

bold_writer = Agent(
    client=client,
    name="BoldWriter",
    instructions=(
        "You are a bold, dramatic copywriter. "
        "Given the product brief, propose one punchy marketing slogan with at most 10 words. "
        "Make it attention-grabbing and confident. Reply with only the slogan."
    ),
)

minimalist_writer = Agent(
    client=client,
    name="MinimalistWriter",
    instructions=(
        "You are a minimalist copywriter who values brevity above all. "
        "Given the product brief, propose one ultra-short marketing slogan with at most 6 words. "
        "Less is more. Reply with only the slogan."
    ),
)

emotional_writer = Agent(
    client=client,
    name="EmotionalWriter",
    instructions=(
        "You are an empathy-driven copywriter. "
        "Given the product brief, propose one marketing slogan with at most 10 words "
        "that connects emotionally with the audience. Reply with only the slogan."
    ),
)

ranker = RankerExecutor(client=client)

workflow = (
    WorkflowBuilder(
        name="FanOutFanInRanked",
        description="Generate slogans in parallel, then rank them with an LLM judge.",
        start_executor=dispatcher,
        output_executors=[ranker],
    )
    .add_fan_out_edges(dispatcher, [bold_writer, minimalist_writer, emotional_writer])
    .add_fan_in_edges([bold_writer, minimalist_writer, emotional_writer], ranker)
    .build()
)


async def main() -> None:
    if not api_key:
        safe_print("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = "Budget-friendly electric bike for urban commuters. Reliable, affordable, green."
    safe_print(f"Product brief: {prompt}\n")

    events = await workflow.run(prompt)
    for output in events.get_outputs():
        for entry in output.rankings:
            safe_print(f'#{entry.rank} (score {entry.score}) [{entry.agent_name}]: "{entry.slogan}"')
            safe_print(f"   {entry.justification}\n")


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8104, auto_open=True)
    else:
        asyncio.run(main())
