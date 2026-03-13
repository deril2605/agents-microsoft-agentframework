"""
Knowledge retrieval (RAG) via a custom context provider.

Diagram:

  Input -> Agent ---------------------> LLM -> Response
            |                           ^
            | search with input         | relevant knowledge
            v                           |
        +----------------+              |
        | Knowledge      |--------------+
        | store          |
        | (SQLite FTS5)  |
        +----------------+
"""

import asyncio
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from agent_framework import Agent, AgentSession, BaseContextProvider, Message, SessionContext, SupportsAgentRun
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv
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


PRODUCTS = [
    {
        "name": "TrailBlaze Hiking Boots",
        "category": "Footwear",
        "price": 149.99,
        "description": (
            "Waterproof hiking boots with Vibram soles, ankle support, "
            "and breathable Gore-Tex lining. Ideal for rocky trails and wet conditions."
        ),
    },
    {
        "name": "SummitPack 40L Backpack",
        "category": "Bags",
        "price": 89.95,
        "description": (
            "Lightweight 40-liter backpack with hydration sleeve, rain cover, "
            "and ergonomic hip belt. Great for day hikes and overnight trips."
        ),
    },
    {
        "name": "ArcticShield Down Jacket",
        "category": "Clothing",
        "price": 199.00,
        "description": (
            "800-fill goose down jacket rated to -20 F. "
            "Features a water-resistant shell, packable design, and adjustable hood."
        ),
    },
    {
        "name": "RiverRun Kayak Paddle",
        "category": "Water Sports",
        "price": 74.50,
        "description": (
            "Fiberglass kayak paddle with adjustable ferrule and drip rings. "
            "Lightweight at 28 oz, suitable for touring and recreational kayaking."
        ),
    },
    {
        "name": "TerraFirm Trekking Poles",
        "category": "Accessories",
        "price": 59.99,
        "description": (
            "Collapsible carbon-fiber trekking poles with cork grips and tungsten tips. "
            "Adjustable from 24 to 54 inches, with anti-shock springs."
        ),
    },
    {
        "name": "ClearView Binoculars 10x42",
        "category": "Optics",
        "price": 129.00,
        "description": (
            "Roof-prism binoculars with 10x magnification and 42mm objective lenses. "
            "Nitrogen-purged and waterproof. Ideal for birding and wildlife observation."
        ),
    },
    {
        "name": "NightGlow LED Headlamp",
        "category": "Lighting",
        "price": 34.99,
        "description": (
            "Rechargeable 350-lumen headlamp with red-light mode and adjustable beam. "
            "IPX6 waterproof rating, runs up to 40 hours on low."
        ),
    },
    {
        "name": "CozyNest Sleeping Bag",
        "category": "Camping",
        "price": 109.00,
        "description": (
            "Three-season mummy sleeping bag rated to 20 F. "
            "Synthetic insulation, compression sack included. Weighs 2.5 lbs."
        ),
    },
]


def create_knowledge_db(db_path: str) -> sqlite3.Connection:
    """Create and seed a product catalog in SQLite with an FTS5 index."""
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS products_fts")
    conn.execute("DROP TABLE IF EXISTS products")

    conn.execute(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            description TEXT NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO products (name, category, price, description) VALUES (?, ?, ?, ?)",
        [(p["name"], p["category"], p["price"], p["description"]) for p in PRODUCTS],
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE products_fts USING fts5(
            name, category, description,
            content='products',
            content_rowid='id'
        )
        """
    )
    conn.execute(
        "INSERT INTO products_fts (rowid, name, category, description) "
        "SELECT id, name, category, description FROM products"
    )
    conn.commit()
    return conn


class SQLiteKnowledgeProvider(BaseContextProvider):
    """Inject relevant product knowledge from SQLite before the model runs."""

    def __init__(self, db_conn: sqlite3.Connection, max_results: int = 3):
        super().__init__(source_id="sqlite-knowledge")
        self.db_conn = db_conn
        self.max_results = max_results

    def _search(self, query: str) -> list[dict[str, Any]]:
        words = re.findall(r"[a-zA-Z]+", query)
        tokens = [word.lower() for word in words if len(word) > 2]
        if not tokens:
            return []

        fts_query = " OR ".join(tokens)
        try:
            cursor = self.db_conn.execute(
                """
                SELECT p.name, p.category, p.price, p.description
                FROM products_fts
                JOIN products p ON products_fts.rowid = p.id
                WHERE products_fts MATCH ?
                LIMIT ?
                """,
                (fts_query, self.max_results),
            )
            return [
                {"name": row[0], "category": row[1], "price": row[2], "description": row[3]}
                for row in cursor.fetchall()
            ]
        except sqlite3.Error:
            logger.debug("FTS query failed for %s", fts_query, exc_info=True)
            return []

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        lines = ["Relevant product information from our catalog:"]
        for product in results:
            lines.append(
                f"- {product['name']} ({product['category']}, ${product['price']:.2f}): "
                f"{product['description']}"
            )
        return "\n".join(lines)

    async def before_run(
        self,
        *,
        agent: SupportsAgentRun,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        user_text = next((msg.text for msg in reversed(context.input_messages) if msg.role == "user" and msg.text), None)
        if not user_text:
            return

        results = self._search(user_text)
        if not results:
            logger.info("[Knowledge] No matching products found for: %s", user_text)
            return

        logger.info("[Knowledge] Found %d matching product(s) for: %s", len(results), user_text)
        context.extend_messages(
            self.source_id,
            [Message(role="user", text=self._format_results(results))],
        )


DB_PATH = ":memory:"
db_conn = create_knowledge_db(DB_PATH)
knowledge_provider = SQLiteKnowledgeProvider(db_conn=db_conn)

agent = Agent(
    client=client,
    instructions=(
        "You are a helpful outdoor-gear shopping assistant for the store 'TrailBuddy'. "
        "Answer customer questions using only the product information provided in the context. "
        "If no relevant products are found in the context, say you do not have information about that item. "
        "Include prices when recommending products."
    ),
    context_providers=[knowledge_provider],
)


async def main() -> None:
    """Demonstrate the knowledge retrieval pattern with several queries."""
    safe_print("\n=== Knowledge Retrieval (RAG) Demo ===")

    safe_print("User: I'm planning a hiking trip. What boots and poles do you recommend?")
    response = await agent.run("I'm planning a hiking trip. What boots and poles do you recommend?")
    safe_print(f"Agent: {response.text}\n")

    safe_print("User: Do you have any surfboards?")
    response = await agent.run("Do you have any surfboards?")
    safe_print(f"Agent: {response.text}\n")

    db_conn.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
