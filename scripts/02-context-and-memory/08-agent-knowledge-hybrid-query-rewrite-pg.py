"""
Knowledge retrieval with query rewriting for multi-turn conversations.

Diagram:

  Conversation -> Rewrite query -> Hybrid search -> Inject knowledge -> LLM answer
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import psycopg
from openai import OpenAI
from pgvector import Vector
from pgvector.psycopg import register_vector

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
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
BASE_URL = normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL"))
API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
CHAT_MODEL = (
    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("LLM_MODEL")
    or "gpt-4.1-mini"
)
EMBED_MODEL = (
    os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    or os.getenv("OPENAI_EMBEDDING_MODEL")
    or os.getenv("MEM0_EMBEDDING_MODEL")
    or "text-embedding-3-small"
)
EMBEDDING_DIMENSIONS = int(
    os.getenv("PGVECTOR_EMBEDDING_DIMS")
    or os.getenv("MEM0_EMBEDDING_DIMS")
    or ("3072" if "large" in EMBED_MODEL else "1536")
)

chat_client = OpenAIChatClient(
    base_url=BASE_URL,
    api_key=API_KEY,
    model_id=CHAT_MODEL,
)
embed_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def get_embedding(text: str) -> list[float]:
    response = embed_client.embeddings.create(
        input=text,
        model=EMBED_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


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


def create_knowledge_db(conn: psycopg.Connection) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute(
        f"""
        CREATE TABLE products (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            description TEXT NOT NULL,
            embedding vector({EMBEDDING_DIMENSIONS})
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX products_fts_idx
        ON products
        USING GIN (to_tsvector('english', name || ' ' || category || ' ' || description))
        """
    )

    logger.info("[Knowledge] Generating embeddings for %d products...", len(PRODUCTS))
    for product in PRODUCTS:
        text_for_embedding = f"{product['name']} - {product['category']}: {product['description']}"
        embedding = get_embedding(text_for_embedding)
        conn.execute(
            "INSERT INTO products (name, category, price, description, embedding) VALUES (%s, %s, %s, %s, %s)",
            (product["name"], product["category"], product["price"], product["description"], embedding),
        )
    conn.commit()
    logger.info("[Knowledge] Product catalog seeded with embeddings.")


HYBRID_SEARCH_SQL = """
WITH semantic_search AS (
    SELECT
        id,
        RANK() OVER (ORDER BY embedding <=> %(embedding)s) AS semantic_rank
    FROM products
    ORDER BY embedding <=> %(embedding)s
    LIMIT 20
),
keyword_search AS (
    SELECT
        id,
        RANK() OVER (
            ORDER BY ts_rank_cd(
                to_tsvector('english', name || ' ' || category || ' ' || description),
                plainto_tsquery('english', %(query)s)
            ) DESC
        ) AS keyword_rank
    FROM products
    WHERE to_tsvector('english', name || ' ' || category || ' ' || description)
          @@ plainto_tsquery('english', %(query)s)
    LIMIT 20
),
fused AS (
    SELECT
        COALESCE(semantic_search.id, keyword_search.id) AS id,
        COALESCE(1.0 / (%(rrf_k)s + semantic_search.semantic_rank), 0.0) +
        COALESCE(1.0 / (%(rrf_k)s + keyword_search.keyword_rank), 0.0) AS score
    FROM semantic_search
    FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
)
SELECT p.name, p.category, p.price, p.description
FROM fused
JOIN products p ON p.id = fused.id
ORDER BY fused.score DESC, p.id ASC
LIMIT %(limit)s
"""


QUERY_REWRITE_PROMPT = (
    "You are a search query optimizer for an outdoor gear product catalog. "
    "Given a conversation between a user and an assistant, generate one concise, "
    "self-contained search query that captures what the user is currently looking for. "
    "Include relevant details from earlier messages that clarify the user's intent. "
    "Respond with only the search query."
)


class PostgresQueryRewriteProvider(BaseContextProvider):
    """Retrieve product knowledge using an LLM-rewritten query for multi-turn conversations."""

    def __init__(self, conn: psycopg.Connection, rewrite_client: OpenAIChatClient, max_results: int = 3):
        super().__init__(source_id="postgres-knowledge-rewrite")
        self.conn = conn
        self.rewrite_client = rewrite_client
        self.max_results = max_results

    async def _rewrite_query(self, conversation_messages: list[Message]) -> str:
        conversation_text = "\n".join(f"{msg.role}: {msg.text}" for msg in conversation_messages if msg.text)
        rewrite_messages = [
            Message(role="system", text=QUERY_REWRITE_PROMPT),
            Message(role="user", text=f"Conversation:\n{conversation_text}"),
        ]
        response = await self.rewrite_client.get_response(rewrite_messages)
        rewritten = (response.text or "").strip().strip('"')
        return rewritten or conversation_messages[-1].text or ""

    def _search(self, query: str) -> list[dict[str, Any]]:
        query_embedding = Vector(get_embedding(query))
        cursor = self.conn.execute(
            HYBRID_SEARCH_SQL,
            {
                "embedding": query_embedding,
                "query": query,
                "rrf_k": 60,
                "limit": self.max_results,
            },
        )
        return [
            {"name": row[0], "category": row[1], "price": row[2], "description": row[3]}
            for row in cursor.fetchall()
        ]

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
        all_messages = list(context.get_messages()) + list(context.input_messages)
        conversation = [msg for msg in all_messages if msg.role in ("user", "assistant") and msg.text]
        if not conversation:
            return

        search_query = await self._rewrite_query(conversation)
        logger.info("[Query Rewrite] -> %s", search_query[:120])

        results = self._search(search_query)
        if not results:
            logger.info("[Knowledge] No matching products found for: %s", search_query)
            return

        logger.info("[Knowledge] Found %d matching product(s)", len(results))
        context.extend_messages(
            self.source_id,
            [Message(role="user", text=self._format_results(results))],
        )


def setup_db() -> psycopg.Connection | None:
    try:
        conn = psycopg.connect(POSTGRES_URL, connect_timeout=5)
    except Exception as exc:
        logger.error("Cannot connect to PostgreSQL at %s: %s", POSTGRES_URL, exc)
        logger.error(
            "Start PostgreSQL with pgvector first. Example: "
            "docker run -d --name pgvector-demo -e POSTGRES_PASSWORD=postgres "
            "-p 5432:5432 pgvector/pgvector:pg17"
        )
        return None

    try:
        create_knowledge_db(conn)
        return conn
    except Exception as exc:
        logger.error("Failed to initialize the PostgreSQL knowledge store: %s", exc)
        conn.close()
        return None


def build_agent(conn: psycopg.Connection) -> Agent:
    knowledge_provider = PostgresQueryRewriteProvider(conn=conn, rewrite_client=chat_client)
    return Agent(
        client=chat_client,
        instructions=(
            "You are a helpful outdoor-gear shopping assistant for the store 'TrailBuddy'. "
            "Answer customer questions using only the product information provided in the context. "
            "If no relevant products are found in the context, say you do not have information about that item. "
            "Include prices when recommending products."
        ),
        context_providers=[knowledge_provider],
    )


async def main() -> None:
    conn = setup_db()
    if conn is None:
        return

    agent = build_agent(conn)
    session = agent.create_session()
    try:
        safe_print("\n=== Knowledge Retrieval with Query Rewriting ===")

        user_msg = "I need protection from rain on rocky paths."
        safe_print(f"User: {user_msg}")
        response = await agent.run(user_msg, session=session)
        safe_print(f"Agent: {response.text}\n")

        user_msg = "What similar gear do you have for snowy situations?"
        safe_print(f"User: {user_msg}")
        response = await agent.run(user_msg, session=session)
        safe_print(f"Agent: {response.text}\n")

        user_msg = "Anything lighter weight I could bring along?"
        safe_print(f"User: {user_msg}")
        response = await agent.run(user_msg, session=session)
        safe_print(f"Agent: {response.text}\n")
    finally:
        conn.close()


if __name__ == "__main__":
    if "--devui" in sys.argv:
        conn = setup_db()
        if conn is not None:
            from agent_framework.devui import serve

            serve(entities=[build_agent(conn)], auto_open=True)
            conn.close()
    else:
        asyncio.run(main())
