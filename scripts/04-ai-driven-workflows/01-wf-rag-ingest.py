r"""RAG ingestion pipeline using plain Python executors.

Demonstrates:
- `Executor` subclasses
- `@handler`
- typed `WorkflowContext`
- `WorkflowBuilder` with explicit edges

This sample does not use AI agents. It is a plain workflow:

    Extract -> Chunk -> Embed

Run:
    uv run .\scripts\04-ai-driven-workflows\01-wf-rag-ingest.py
    uv run .\scripts\04-ai-driven-workflows\01-wf-rag-ingest.py --devui

In DevUI, enter a filename relative to this folder, for example:
`sample_document.pdf`
"""

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler
from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import OpenAI
from rich.logging import RichHandler
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


log_handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[log_handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

EMBEDDING_DIMENSIONS = 256  # Smaller dimension for efficiency.

embedding_base_url = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")
embedding_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or "text-embedding-3-small"

embed_client = OpenAI(
    base_url=normalize_base_url(embedding_base_url),
    api_key=embedding_api_key,
)


@dataclass
class EmbeddedChunk:
    """A text chunk paired with its embedding vector."""

    text: str
    vector: list[float] = field(default_factory=list)


class ExtractExecutor(Executor):
    """Convert a local file to plain markdown text."""

    @handler
    async def extract(self, path: str, ctx: WorkflowContext[str]) -> None:
        """Convert the file at the given path to markdown."""
        path = path.strip("'\"")
        resolved = Path(path) if Path(path).is_absolute() else Path(__file__).parent / path
        if not resolved.exists():
            raise FileNotFoundError(f"Input file not found: {resolved}")
        result = MarkItDown().convert(str(resolved))
        await ctx.send_message(result.text_content)


class ChunkExecutor(Executor):
    """Split markdown into paragraphs, keeping only substantive ones."""

    @handler
    async def chunk(self, markdown: str, ctx: WorkflowContext[list[str]]) -> None:
        """Split on blank lines and filter out headings and short fragments."""
        paragraphs = markdown.split("\n\n")
        chunks = [p.strip() for p in paragraphs if len(p.strip()) >= 80 and not p.strip().startswith("#")]
        logger.info("-> %s chunks extracted", len(chunks))
        await ctx.send_message(chunks)


class EmbedExecutor(Executor):
    """Embed each chunk with the configured OpenAI embedding model."""

    @handler
    async def embed(self, chunks: list[str], ctx: WorkflowContext[Never, list[EmbeddedChunk]]) -> None:
        """Call the embeddings API for each chunk and yield the results."""
        embedded = []
        for chunk in chunks:
            response = embed_client.embeddings.create(
                input=chunk,
                model=embedding_model,
                dimensions=EMBEDDING_DIMENSIONS,
            )
            embedded.append(EmbeddedChunk(text=chunk, vector=response.data[0].embedding))
        logger.info("-> %s chunks embedded (%sd each)", len(embedded), EMBEDDING_DIMENSIONS)
        await ctx.yield_output(embedded)


extract = ExtractExecutor(id="extract")
chunk = ChunkExecutor(id="chunk")
embed = EmbedExecutor(id="embed")

workflow = WorkflowBuilder(start_executor=extract).add_edge(extract, chunk).add_edge(chunk, embed).build()


async def main() -> None:
    if not embedding_base_url or not embedding_api_key:
        logger.error(
            "No embedding client configuration found. Set Azure/OpenAI credentials in .env before running this sample."
        )
        return

    input_path = Path(__file__).parent / "sample_document.pdf"
    if not input_path.exists():
        logger.error("Sample input file not found: %s", input_path)
        logger.info("Place a file like sample_document.pdf in this folder, or run with --devui and provide a file path.")
        return

    logger.info("Processing: %s", input_path)
    events = await workflow.run(str(input_path))
    outputs = events.get_outputs()
    for result in outputs:
        logger.info("Embedded %s chunks:", len(result))
        for chunk in result:
            preview = chunk.text[:80].replace("\n", " ")
            logger.info("  [%sd] %s...", len(chunk.vector), preview)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[workflow], port=8090, auto_open=True)
    else:
        asyncio.run(main())
