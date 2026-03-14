"""
Context compaction via summarization middleware.

When a conversation grows long, the accumulated messages can exceed the
model's context window or become expensive. This middleware monitors
cumulative token usage and, once a threshold is crossed, asks the LLM to
summarize the conversation so far. The summary replaces the old messages,
freeing up context space for future turns.
"""

import asyncio
import logging
import os
import random
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from agent_framework import (
    Agent,
    AgentContext,
    AgentMiddleware,
    AgentResponse,
    InMemoryHistoryProvider,
    Message,
    tool,
)
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from pydantic import Field
from rich import print
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


def safe_print(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
client = OpenAIChatClient(
    base_url=normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")),
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
    model_id=(
        os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or "gpt-4.1-mini"
    ),
)


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Return weather data for a given city."""
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    temp = random.randint(30, 90)
    return f"The weather in {city} is {random.choice(conditions)} with a high of {temp} F."


@tool
def get_activities(
    city: Annotated[str, Field(description="The city to find activities in.")],
) -> str:
    """Return popular weekend activities for a given city."""
    all_activities = [
        "Visit the farmer's market",
        "Hike at the local state park",
        "Check out a food truck festival",
        "Go to the art museum",
        "Take a walking tour of downtown",
        "Visit the botanical garden",
        "Catch a live music show",
        "Try a new brunch spot",
    ]
    picked = random.sample(all_activities, k=3)
    return f"Popular activities in {city}: {', '.join(picked)}."


SUMMARIZE_PROMPT = (
    "You are a summarization assistant. Condense the following conversation "
    "into a concise summary that preserves all key facts, decisions, and context "
    "needed to continue the conversation. Write the summary in third person. "
    "Be concise but do not lose important details like specific cities, "
    "weather conditions, or recommendations that were discussed."
)


class SummarizationMiddleware(AgentMiddleware):
    """Summarize conversation history once cumulative token usage crosses a threshold."""

    def __init__(self, client: OpenAIChatClient, token_threshold: int = 1000) -> None:
        self.client = client
        self.token_threshold = token_threshold
        self.context_tokens = 0

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        lines: list[str] = []
        for msg in messages:
            if msg.text:
                lines.append(f"{msg.role}: {msg.text}")
        return "\n".join(lines)

    async def _summarize(self, messages: list[Message]) -> str:
        conversation_text = self._format_messages_for_summary(messages)
        summary_messages = [
            Message(role="system", text=SUMMARIZE_PROMPT),
            Message(role="user", text=f"Summarize this conversation:\n\n{conversation_text}"),
        ]
        response = await self.client.get_response(summary_messages)
        return response.text or "No summary available."

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        session = context.session

        if session and self.context_tokens > self.token_threshold:
            history_state = session.state.get(InMemoryHistoryProvider.DEFAULT_SOURCE_ID, {})
            history = list(history_state.get("messages", []))
            if len(history) > 2:
                logger.info(
                    "[Summarization] Token usage (%d) exceeds threshold (%d). Summarizing %d messages.",
                    self.context_tokens,
                    self.token_threshold,
                    len(history),
                )
                summary_text = await self._summarize(history)
                history_state["messages"] = [
                    Message(role="assistant", text=f"[Summary of earlier conversation]\n{summary_text}"),
                ]
                session.state[InMemoryHistoryProvider.DEFAULT_SOURCE_ID] = history_state
                self.context_tokens = 0
                logger.info("[Summarization] History compacted to one summary message.")
        else:
            logger.info(
                "[Summarization] Token usage: %d / %d threshold. No summarization needed.",
                self.context_tokens,
                self.token_threshold,
            )

        await call_next()

        if context.result and isinstance(context.result, AgentResponse) and context.result.usage_details:
            new_tokens = context.result.usage_details.get("total_token_count", 0) or 0
            self.context_tokens += new_tokens
            logger.info(
                "[Summarization] This turn used %d tokens. Context total: %d",
                new_tokens,
                self.context_tokens,
            )


summarization_middleware = SummarizationMiddleware(client=client, token_threshold=500)

agent = Agent(
    name="weekend-planner",
    client=client,
    instructions=(
        "You are a helpful weekend-planning assistant. Help users plan "
        "their weekends by checking weather and suggesting activities. "
        "Be friendly and provide detailed recommendations."
    ),
    tools=[get_weather, get_activities],
    middleware=[summarization_middleware],
)


async def main() -> None:
    """Run a multi-turn conversation that triggers summarization."""
    safe_print("\n=== Context Compaction with Summarization ===")
    safe_print(f"Token threshold: {summarization_middleware.token_threshold}")
    safe_print("The middleware will summarize the conversation once token usage exceeds the threshold.\n")

    session = agent.create_session()

    user_msg = "What's the weather like in San Francisco this weekend?"
    safe_print(f"User: {user_msg}")
    response = await agent.run(user_msg, session=session)
    safe_print(f"Agent: {response.text}\n")

    user_msg = "How about Portland? What's the weather and what activities can I do there?"
    safe_print(f"User: {user_msg}")
    response = await agent.run(user_msg, session=session)
    safe_print(f"Agent: {response.text}\n")

    user_msg = "What about Seattle? Give me the full picture about weather and things to do."
    safe_print(f"User: {user_msg}")
    response = await agent.run(user_msg, session=session)
    safe_print(f"Agent: {response.text}\n")

    user_msg = "Of all the cities we discussed, which one has the best combination of weather and activities?"
    safe_print(f"User: {user_msg}")
    response = await agent.run(user_msg, session=session)
    safe_print(f"Agent: {response.text}\n")

    user_msg = "Great, let's go with that city. What should I pack?"
    safe_print(f"User: {user_msg}")
    response = await agent.run(user_msg, session=session)
    safe_print(f"Agent: {response.text}\n")

    safe_print(f"Final context token count: {summarization_middleware.context_tokens}")


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
