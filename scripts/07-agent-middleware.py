"""
Middleware flow diagram:

 agent.run("user message")
 │
 ▼
 ┌─────────────────────────────────────────────┐
 │         Agent Middleware                    │
 │  (timing, blocking, logging)                │
 │                                             │
 │  ┌───────────────────────────────────────┐  │
 │  │       Chat Middleware                 │  │
 │  │  (logging, message counting)          │  │
 │  │                                       │  │
 │  │        ┌──────────────┐               │  │
 │  │        │   AI Model   │               │  │
 │  │        └──────┬───────┘               │  │
 │  │               │ tool calls            │  │
 │  │               ▼                       │  │
 │  │  ┌──────────────────────────────────┐ │  │
 │  │  │     Function Middleware          │ │  │
 │  │  │  (logging, timing)               │ │  │
 │  │  │                                  │ │  │
 │  │  │  get_weather(), get_date(), ...  │ │  │
 │  │  └──────────────────────────────────┘ │  │
 │  │               │                       │  │
 │  │               ▼                       │  │
 │  │        ┌──────────────┐               │  │
 │  │        │   AI Model   │               │  │
 │  │        │  (final ans) │               │  │
 │  │        └──────────────┘               │  │
 │  └───────────────────────────────────────┘  │
 └─────────────────────────────────────────────┘
 │
 ▼
 response
"""

import asyncio
import logging
import os
import random
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

from agent_framework import (
    AgentMiddleware,
    AgentContext,
    AgentResponse,
    Agent,
    ChatContext,
    Message,
    ChatMiddleware,
    FunctionInvocationContext,
    FunctionMiddleware,
    tool,
)
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
from pydantic import Field


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.rstrip("/")
    if stripped.endswith("/openai/v1"):
        return f"{stripped}/"
    if "openai.azure.com" in stripped:
        return f"{stripped}/openai/v1/"
    return stripped

# Setup logging
logging.basicConfig(level=logging.WARNING, force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure Azure OpenAI client based on environment
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
client = OpenAIChatClient(
    base_url=normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")),
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
    model_id=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("LLM_MODEL") or "gpt-4.1-mini",
)
STDOUT_ENCODING = sys.stdout.encoding or "utf-8"


# ---- Tools ----


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> dict:
    """Return weather data for a given city, a dictionary with temperature and description."""
    logger.info(f"Getting weather for {city}")
    if random.random() < 0.05:
        return {"temperature": 72, "description": "Sunny"}
    else:
        return {"temperature": 60, "description": "Rainy"}


@tool
def get_current_date() -> str:
    """Get the current date from the system and return as a string in format YYYY-MM-DD."""
    logger.info("Getting current date")
    return datetime.now().strftime("%Y-%m-%d")


# ---- Function-based middleware ----


async def timing_agent_middleware(
    context: AgentContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """Agent middleware that logs execution timing."""
    start = time.perf_counter()
    logger.info("[Timing][Agent Middleware] Starting agent execution")

    await call_next()

    elapsed = time.perf_counter() - start
    logger.info(f"[Timing][Agent Middleware] Execution completed in {elapsed:.2f}s")


async def logging_function_middleware(
    context: FunctionInvocationContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """Function middleware that logs function calls and results."""
    logger.info(f"[Logging][Function Middleware] Calling {context.function.name} with args: {context.arguments}")

    await call_next()

    logger.info(f"[Logging][Function Middleware] {context.function.name} returned: {context.result}")


async def logging_chat_middleware(
    context: ChatContext,
    call_next: Callable[[], Awaitable[None]],
) -> None:
    """Chat middleware that logs AI interactions."""
    logger.info(f"[Logging][Chat Middleware] Sending {len(context.messages)} messages to AI")

    await call_next()

    logger.info("[Logging][Chat Middleware] AI response received")


# ---- Class-based middleware ----


class BlockingAgentMiddleware(AgentMiddleware):
    """Agent middleware that blocks requests containing forbidden words."""

    def __init__(self, blocked_words: list[str]) -> None:
        """Initialize with a list of words that should be blocked."""
        self.blocked_words = blocked_words

    async def process(
        self,
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Check messages for blocked content and terminate if found."""
        last_message = context.messages[-1] if context.messages else None
        if last_message and last_message.text:
            for word in self.blocked_words:
                if word.lower() in last_message.text.lower():
                    logger.warning(f"[Blocking][Agent Middleware] Request blocked: contains '{word}'")
                    context.terminate = True
                    context.result = AgentResponse(
                        messages=[
                            Message(role="assistant", text=f"Sorry, I can't process requests about '{word}'.")
                        ]
                    )
                    return

        await call_next()


class TimingFunctionMiddleware(FunctionMiddleware):
    """Function middleware that tracks execution time of each function call."""

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Time the function execution and log the duration."""
        start = time.perf_counter()
        logger.info(f"[Timing][Function Middleware] Starting {context.function.name}")

        await call_next()

        elapsed = time.perf_counter() - start
        logger.info(f"[Timing][Function Middleware] {context.function.name} took {elapsed:.4f}s")


class MessageCountChatMiddleware(ChatMiddleware):
    """Chat middleware that tracks the total number of messages sent to the AI."""

    def __init__(self) -> None:
        """Initialize the message counter."""
        self.total_messages = 0

    async def process(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Count messages and log the running total."""
        self.total_messages += len(context.messages)
        logger.info(
            "[Message Count][Chat Middleware] Messages in this request: %s, total so far: %s",
            len(context.messages),
            self.total_messages,
        )

        await call_next()

        logger.info("[Message Count][Chat Middleware] Chat response received")


# ---- Agent setup ----

# Instantiate class-based middleware
blocking_middleware = BlockingAgentMiddleware(blocked_words=["nuclear", "classified"])
timing_function_middleware = TimingFunctionMiddleware()
message_count_middleware = MessageCountChatMiddleware()

agent = Agent(
    name="middleware-demo",
    client=client,
    instructions="You help users plan their weekends. Use the available tools to check the weather and date.",
    tools=[get_weather, get_current_date],
    middleware=[
        # Agent-level middleware applied to ALL runs
        timing_agent_middleware,
        blocking_middleware,
        logging_function_middleware,
        timing_function_middleware,
        logging_chat_middleware,
        message_count_middleware,
    ],
)


async def main() -> None:
    """Run the agent with different inputs to demonstrate middleware behavior."""
    # Normal request - all middleware fires
    logger.info("=== Normal Request ===")
    response = await agent.run("What's the weather like this weekend in San Francisco?")
    print(response.text.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))

    # Blocked request - blocking middleware terminates early
    logger.info("\n=== Blocked Request ===")
    response = await agent.run("Tell me about nuclear physics.")
    print(response.text.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))

    # Another normal request with run-level middleware
    logger.info("\n=== Request with Run-Level Middleware ===")

    async def extra_agent_middleware(
        context: AgentContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        """Run-level middleware that only applies to this specific run."""
        logger.info("[Run-Level Middleware] This middleware only applies to this run")
        await call_next()
        logger.info("[Run-Level Middleware] Run completed")

    response = await agent.run(
        "What's the weather like in Portland?",
        middleware=[extra_agent_middleware],
    )
    print(response.text.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))

if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
