"""
OpenTelemetry demo with Aspire export.

Run with Aspire dashboard:
    docker run --rm -it -d `
      -p 18888:18888 `
      -p 4317:18889 `
      --name aspire-dashboard `
      mcr.microsoft.com/dotnet/aspire-dashboard:latest

    $env:OTEL_EXPORTER_OTLP_PROTOCOL="grpc"
    $env:OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
    $env:OTEL_SERVICE_NAME="agents-msft-agentframework"
    uv run .\scripts\03-monitoring-and-evaluating\01-agent-otel-aspire.py

Optional console telemetry during local debugging:
    $env:ENABLE_CONSOLE_EXPORTERS="true"

Open Aspire dashboard in a browser at http://localhost:18888, find your
service, and inspect the traces and logs emitted by the agent.

If you wanted Azure Application Insights instead of Aspire, the
observability setup would change roughly like this:

    from agent_framework.observability import create_resource, enable_instrumentation
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(
        connection_string=os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"],
        resource=create_resource(),
        enable_live_metrics=True,
    )
    enable_instrumentation(enable_sensitive_data=True)

In that version, you would replace `configure_otel_providers(...)` with the
Azure Monitor setup above and provide `APPLICATIONINSIGHTS_CONNECTION_STRING`.
"""

import asyncio
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.observability import configure_otel_providers
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

# Configure OpenTelemetry using any OTEL_* environment variables that are present.
configure_otel_providers(enable_sensitive_data=True)

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
    city: Annotated[str, Field(description="City name, spelled out fully")],
) -> dict:
    """Return mock weather data for a given city."""
    logger.info("Getting weather for %s", city)
    weather_options = [
        {"temperature": 72, "description": "Sunny"},
        {"temperature": 60, "description": "Rainy"},
        {"temperature": 55, "description": "Cloudy"},
        {"temperature": 45, "description": "Windy"},
    ]
    return random.choice(weather_options)


@tool
def get_current_time(
    timezone_name: Annotated[str, Field(description="Timezone name, e.g. 'US/Eastern', 'Asia/Tokyo', 'UTC'")],
) -> str:
    """Return the current time in UTC, labeled with the requested timezone name."""
    logger.info("Getting current time for %s", timezone_name)
    now = datetime.now(timezone.utc)
    return f"The current time in {timezone_name} is approximately {now.strftime('%Y-%m-%d %H:%M:%S')} UTC."


agent = Agent(
    name="weather-time-agent",
    client=client,
    instructions="You are a helpful assistant that can look up weather and time information.",
    tools=[get_weather, get_current_time],
)


async def main() -> None:
    if not (
        os.getenv("AZURE_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    ):
        logger.error("No LLM API key found. Set LLM_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY.")
        return

    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is not set. The sample will run, but telemetry may stay local "
            "instead of showing up in Aspire."
        )

    response = await agent.run("What's the weather in Seattle and what time is it in Tokyo?")
    safe_print(response.text)


if __name__ == "__main__":
    if "--devui" in sys.argv:
        from agent_framework.devui import serve

        serve(entities=[agent], auto_open=True)
    else:
        asyncio.run(main())
