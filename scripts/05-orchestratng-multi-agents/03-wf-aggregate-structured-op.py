r"""Fan-out/fan-in with structured extraction aggregation.

Three interviewer agents (technical, behavioral, culture-fit) each assess a
job candidate. The fan-in executor collects their assessments, calls the LLM
with `response_format=CandidateReview`, and yields a typed Pydantic model that
is ready for downstream code, not just prose.

Aggregation technique:
- LLM structured extraction into a typed model

Run:
    uv run .\scripts\05-orchestratng-multi-agents\03-wf-aggregate-structured-op.py
    uv run .\scripts\05-orchestratng-multi-agents\03-wf-aggregate-structured-op.py --devui
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

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


class CandidateReview(BaseModel):
    """Typed output produced by the reviewer."""

    technical_score: int = Field(description="Technical skills score from 1 to 10.")
    technical_reason: str = Field(description="Justification for the technical score.")
    behavioral_score: int = Field(description="Behavioral skills score from 1 to 10.")
    behavioral_reason: str = Field(description="Justification for the behavioral score.")
    recommendation: Literal["strong hire", "hire with reservations", "no hire"] = Field(
        description="Final hiring recommendation."
    )


class DispatchPrompt(Executor):
    """Emit the candidate description downstream for fan-out broadcast."""

    @handler
    async def dispatch(self, prompt: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(prompt)


class ExtractReview(Executor):
    """Fan-in aggregator that produces a typed CandidateReview."""

    def __init__(self, *, client: OpenAIChatClient, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._client = client

    @handler
    async def extract(
        self,
        results: list[AgentExecutorResponse],
        ctx: WorkflowContext[Never, CandidateReview],
    ) -> None:
        """Collect interviewer assessments and ask the LLM for a structured review."""
        sections = []
        for result in results:
            label = result.executor_id.replace("_", " ").title()
            sections.append(f"[{label}]\n{result.agent_response.text}")
        combined = "\n\n".join(sections)

        messages = [
            Message(
                role="system",
                text=(
                    "You are a hiring committee reviewer. "
                    "Based on the following interviewer assessments, produce a structured candidate review."
                ),
            ),
            Message(role="user", text=combined),
        ]
        response = await self._client.get_response(messages, options={"response_format": CandidateReview})
        review: CandidateReview = response.value
        await ctx.yield_output(review)


dispatcher = DispatchPrompt(id="dispatcher")

technical_interviewer = Agent(
    client=client,
    name="TechnicalInterviewer",
    instructions=(
        "You are a senior engineer conducting a technical interview. "
        "Assess the candidate's technical skills, architecture knowledge, and coding ability. "
        "Be specific about strengths and gaps. Use short bullet points."
    ),
)

behavioral_interviewer = Agent(
    client=client,
    name="BehavioralInterviewer",
    instructions=(
        "You are an HR specialist conducting a behavioral interview. "
        "Assess the candidate's communication, teamwork, conflict resolution, and leadership. "
        "Be specific about strengths and gaps. Use short bullet points."
    ),
)

cultural_interviewer = Agent(
    client=client,
    name="CulturalInterviewer",
    instructions=(
        "You are a team lead assessing culture fit. "
        "Evaluate whether the candidate aligns with a collaborative, fast-paced startup culture. "
        "Be specific about strengths and gaps. Use short bullet points."
    ),
)

extractor = ExtractReview(client=client, id="extractor")

workflow = (
    WorkflowBuilder(
        name="FanOutFanInStructured",
        description="Fan-out/fan-in with Pydantic structured extraction.",
        start_executor=dispatcher,
        output_executors=[extractor],
    )
    .add_fan_out_edges(dispatcher, [technical_interviewer, behavioral_interviewer, cultural_interviewer])
    .add_fan_in_edges([technical_interviewer, behavioral_interviewer, cultural_interviewer], extractor)
    .build()
)


async def main() -> None:
    if not api_key:
        safe_print("No model API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY in .env.")
        return

    prompt = (
        "Candidate applying for Senior Software Engineer. "
        "5 years experience in Python and distributed systems. "
        "Strong communicator but limited cloud experience."
    )
    safe_print(f"Candidate brief: {prompt}\n")

    events = await workflow.run(prompt)
    for output in events.get_outputs():
        safe_print(f"Recommendation: {output.recommendation}\n")
        safe_print(f"Technical: {output.technical_score}/10 - {output.technical_reason}\n")
        safe_print(f"Behavioral: {output.behavioral_score}/10 - {output.behavioral_reason}")


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8102, auto_open=True)
    else:
        asyncio.run(main())
