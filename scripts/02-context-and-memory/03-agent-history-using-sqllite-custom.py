# Data flow in SQLite history storage
#
#   User message / Agent response
#               |
#               v
#       save_messages(...)
#               |
#               v
#   +--------------------------------------+
#   | SQLite file: chat_history.sqlite3    |
#   |                                      |
#   | messages table                       |
#   | -----------------------------------  |
#   | id | session_id | message_json      |
#   | 1  | abc-123    | {...user msg...}  |
#   | 2  | abc-123    | {...agent msg...} |
#   | 3  | abc-123    | {...user msg...}  |
#   +--------------------------------------+
#               |
#               v
#       get_messages(session_id)
#               |
#               v
#   Rebuild Message objects in original order
#   and inject them back into the agent session


import asyncio
import logging
import os
import random
import sqlite3
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from agent_framework import Agent, BaseHistoryProvider, Message, tool
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


# Setup logging
handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure OpenAI client based on environment
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


class SQLiteHistoryProvider(BaseHistoryProvider):
    """A custom history provider backed by SQLite."""

    def __init__(self, db_path: str):
        super().__init__("sqlite-history")
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    async def get_messages(self, session_id: str | None, **kwargs: Any) -> list[Message]:
        """Retrieve all messages for this session from SQLite."""
        if session_id is None:
            return []
        cursor = self._conn.execute(
            "SELECT message_json FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [Message.from_json(row[0]) for row in cursor.fetchall()]

    async def save_messages(self, session_id: str | None, messages: Sequence[Message], **kwargs: Any) -> None:
        """Save messages to the SQLite database."""
        if session_id is None:
            return
        self._conn.executemany(
            "INSERT INTO messages (session_id, message_json) VALUES (?, ?)",
            [(session_id, message.to_json()) for message in messages],
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather for.")],
) -> str:
    """Returns weather data for a given city."""
    logger.info("Getting weather for %s", city)
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {city} is {random.choice(conditions)} with a high of {random.randint(10, 30)} C."


async def main() -> None:
    """Demonstrate a SQLite-backed session that persists conversation history to a local file."""
    db_path = str(Path(__file__).with_name("chat_history.sqlite3"))
    session_id = str(uuid.uuid4())

    safe_print("\n=== Persistent SQLite Session ===")
    safe_print("--- Phase 1: Starting conversation ---")

    sqlite_provider = SQLiteHistoryProvider(db_path=db_path)

    try:
        agent = Agent(
            client=client,
            instructions="You are a helpful weather agent.",
            tools=[get_weather],
            context_providers=[sqlite_provider],
        )

        session = agent.create_session(session_id=session_id)

        safe_print("User: What's the weather like in Tokyo?")
        response = await agent.run("What's the weather like in Tokyo?", session=session)
        safe_print(f"Agent: {response.text}")

        safe_print("\nUser: How about Paris?")
        response = await agent.run("How about Paris?", session=session)
        safe_print(f"Agent: {response.text}")

        safe_print("\n--- Phase 2: Resuming after 'restart' ---")
        sqlite_provider2 = SQLiteHistoryProvider(db_path=db_path)

        try:
            agent2 = Agent(
                client=client,
                instructions="You are a helpful weather agent.",
                tools=[get_weather],
                context_providers=[sqlite_provider2],
            )

            session2 = agent2.create_session(session_id=session_id)

            safe_print("User: Which of the cities I asked about had better weather?")
            response = await agent2.run("Which of the cities I asked about had better weather?", session=session2)
            safe_print(f"Agent: {response.text}")
        finally:
            sqlite_provider2.close()
    finally:
        sqlite_provider.close()


if __name__ == "__main__":
    asyncio.run(main())
