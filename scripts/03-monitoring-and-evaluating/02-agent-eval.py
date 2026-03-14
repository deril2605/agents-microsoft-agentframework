"""
Evaluate an agent response with Azure AI Evaluation.

This sample runs a travel-planning agent, then scores the final response with
built-in evaluators for intent resolution, response completeness, task
adherence, and tool-call accuracy.

This file shows the inline pattern:
1. run the agent
2. collect the response and message history
3. call evaluators directly in Python

You can also evaluate with Azure AI Evaluation's batch-style `evaluate()`
function instead of invoking each evaluator one by one. That approach is
useful when you already have evaluation rows in JSONL form and want to score
many runs in one call, optionally logging results to Azure AI Foundry.

In other words:
- this sample = direct, per-run evaluation inside the script
- `evaluate()` pattern = dataset-driven evaluation over saved rows

The script follows the same `.env` pattern as the other samples in this repo:

    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

It also supports Azure OpenAI directly with:

    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_CHAT_DEPLOYMENT

If you choose the `evaluate()` approach, you would typically:
- write evaluation inputs such as `query`, `response`, `response_text`,
  `ground_truth`, and `tool_definitions` to JSONL
- construct evaluators in a dictionary
- call `evaluate(data=..., evaluators=..., evaluator_config=...)`
- save or upload the results separately

To send batch-evaluation results to Azure AI Foundry, set
`AZURE_AI_PROJECT` to your project URL, for example:

    https://your-account.services.ai.azure.com/api/projects/your-project

Then pass `azure_ai_project=AZURE_AI_PROJECT` to `evaluate(...)` instead of
an `output_path`. Azure AI Evaluation will log the run to Foundry, and the
returned result can include a Studio URL for viewing the evaluation report.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Annotated

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    IntentResolutionEvaluator,
    OpenAIModelConfiguration,
    ResponseCompletenessEvaluator,
    TaskAdherenceEvaluator,
    ToolCallAccuracyEvaluator,
)
from dotenv import load_dotenv
from pydantic import Field
from rich import print
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.rstrip("/")
    if stripped.endswith("/openai/v1"):
        return f"{stripped}/"
    if "openai.azure.com" in stripped:
        return f"{stripped}/openai/v1/"
    return stripped


handler = RichHandler(show_path=False, rich_tracebacks=True, show_level=False)
logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_base_url = os.getenv("LLM_BASE_URL")
llm_api_key = os.getenv("LLM_API_KEY")
llm_model = os.getenv("LLM_MODEL")

if azure_endpoint and azure_api_key and azure_deployment:
    client = OpenAIChatClient(
        base_url=normalize_base_url(azure_endpoint),
        api_key=azure_api_key,
        model_id=azure_deployment,
    )
    eval_model_config = AzureOpenAIModelConfiguration(
        type="azure_openai",
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        api_key=azure_api_key,
    )
elif llm_api_key and llm_base_url:
    model_id = llm_model or "gpt-4.1-mini"
    normalized_base_url = normalize_base_url(llm_base_url)
    client = OpenAIChatClient(base_url=normalized_base_url, api_key=llm_api_key, model_id=model_id)
    eval_model_config = OpenAIModelConfiguration(
        type="openai",
        base_url=normalized_base_url,
        api_key=llm_api_key,
        model=model_id,
    )
else:
    model_id = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAIChatClient(api_key=openai_api_key, model_id=model_id)
    eval_model_config = OpenAIModelConfiguration(
        type="openai",
        api_key=openai_api_key,
        model=model_id,
    )


@tool
def get_weather(
    city: Annotated[str, Field(description="The city to get the weather forecast for.")],
    date_range: Annotated[str, Field(description="Date range in format 'YYYY-MM-DD to YYYY-MM-DD'.")],
) -> dict:
    """Returns a weather forecast for a city over a date range, including temperature and conditions."""
    logger.info(f"Getting weather for {city} ({date_range})")
    return {
        "city": city,
        "date_range": date_range,
        "forecast": [
            {"date": "Day 1", "high_f": 65, "low_f": 52, "conditions": "Partly cloudy"},
            {"date": "Day 2", "high_f": 70, "low_f": 55, "conditions": "Sunny"},
            {"date": "Day 3", "high_f": 62, "low_f": 50, "conditions": "Light rain"},
        ],
    }


@tool
def search_flights(
    origin: Annotated[str, Field(description="Departure city or airport code.")],
    destination: Annotated[str, Field(description="Arrival city or airport code.")],
    departure_date: Annotated[str, Field(description="Departure date in YYYY-MM-DD format.")],
    return_date: Annotated[str, Field(description="Return date in YYYY-MM-DD format.")],
) -> list[dict]:
    """Searches for round-trip flights and returns options with prices."""
    logger.info(f"Searching flights {origin} -> {destination} ({departure_date} to {return_date})")
    return [
        {"airline": "SkyAir", "price_usd": 850, "duration": "14h 20m", "stops": 1},
        {"airline": "OceanWings", "price_usd": 720, "duration": "16h 45m", "stops": 2},
        {"airline": "DirectJet", "price_usd": 1100, "duration": "12h 30m", "stops": 0},
    ]


@tool
def search_hotels(
    city: Annotated[str, Field(description="The city to search hotels in.")],
    checkin: Annotated[str, Field(description="Check-in date in YYYY-MM-DD format.")],
    checkout: Annotated[str, Field(description="Check-out date in YYYY-MM-DD format.")],
    max_price_per_night: Annotated[int, Field(description="Maximum price per night in USD.")],
) -> list[dict]:
    """Searches for hotels within a nightly budget and returns options with ratings."""
    logger.info(f"Searching hotels in {city} ({checkin} to {checkout}, max ${max_price_per_night}/night)")
    return [
        {"name": "Budget Inn Tokyo", "price_per_night_usd": 80, "rating": 3.8, "neighborhood": "Asakusa"},
        {"name": "Sakura Hotel", "price_per_night_usd": 120, "rating": 4.2, "neighborhood": "Shinjuku"},
        {"name": "Tokyo Garden Suites", "price_per_night_usd": 200, "rating": 4.6, "neighborhood": "Ginza"},
    ]


@tool
def get_activities(
    city: Annotated[str, Field(description="The city to find activities in.")],
    interests: Annotated[list[str], Field(description="List of interests, e.g. ['hiking', 'museums'].")],
) -> list[dict]:
    """Returns activity suggestions for a city based on user interests."""
    logger.info(f"Getting activities in {city} for interests: {interests}")
    activities = []
    if "hiking" in [i.lower() for i in interests]:
        activities.extend(
            [
                {"name": "Mt. Takao Day Hike", "cost_usd": 15, "duration": "4-5 hours"},
                {"name": "Kamakura Trail Walk", "cost_usd": 25, "duration": "3 hours"},
            ]
        )
    if "museums" in [i.lower() for i in interests]:
        activities.extend(
            [
                {"name": "Tokyo National Museum", "cost_usd": 10, "duration": "2-3 hours"},
                {"name": "teamLab Borderless", "cost_usd": 30, "duration": "2 hours"},
            ]
        )
    if not activities:
        activities = [{"name": "City walking tour", "cost_usd": 0, "duration": "3 hours"}]
    return activities


@tool
def estimate_budget(
    total_budget: Annotated[int, Field(description="Total trip budget in USD.")],
    num_days: Annotated[int, Field(description="Number of days for the trip.")],
) -> dict:
    """Provides a recommended budget breakdown for flights, hotels, activities, and food."""
    logger.info(f"Estimating budget: ${total_budget} for {num_days} days")
    flight_pct = 0.40
    hotel_pct = 0.30
    activities_pct = 0.15
    food_pct = 0.15
    return {
        "total_budget_usd": total_budget,
        "flights_usd": int(total_budget * flight_pct),
        "hotels_usd": int(total_budget * hotel_pct),
        "hotels_per_night_usd": int(total_budget * hotel_pct / num_days),
        "activities_usd": int(total_budget * activities_pct),
        "food_usd": int(total_budget * food_pct),
        "food_per_day_usd": int(total_budget * food_pct / num_days),
    }


tools = [get_weather, search_flights, search_hotels, get_activities, estimate_budget]

tool_definitions = [t.to_json_schema_spec()["function"] for t in tools]

AGENT_INSTRUCTIONS = (
    "You are a travel planning assistant. Help users plan trips by checking weather, "
    "finding flights and hotels within budget, and suggesting activities based on their interests. "
    "Always provide a complete itinerary with costs for each component and ensure the total stays "
    "within the user's budget. Include weather information to help with packing."
)

agent = Agent(
    client=client,
    instructions=AGENT_INSTRUCTIONS,
    tools=tools,
)


def convert_to_evaluator_messages(messages) -> list[dict]:
    """Convert agent framework ChatMessages to the Azure AI Evaluation message schema.

    Remaps content types: function_call -> tool_call, function_result -> tool_result.
    See: https://learn.microsoft.com/azure/ai-foundry/how-to/develop/agent-evaluate-sdk#agent-message-schema
    """
    evaluator_messages = []
    for msg in messages:
        role = str(msg.role.value) if hasattr(msg.role, "value") else str(msg.role)
        content_items = []
        for c in msg.contents:
            if c.type == "function_call":
                arguments = json.loads(c.arguments) if isinstance(c.arguments, str) else c.arguments
                content_items.append(
                    {
                        "type": "tool_call",
                        "tool_call_id": c.call_id,
                        "name": c.name,
                        "arguments": arguments,
                        "tool_call": {
                            "id": c.call_id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": arguments,
                            },
                        },
                    }
                )
            elif c.type == "function_result":
                if c.call_id:
                    if content_items:
                        evaluator_messages.append({"role": role, "content": content_items})
                        content_items = []
                    evaluator_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": c.call_id,
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_result": c.result,
                                }
                            ],
                        }
                    )
                    continue
                content_items.append(
                    {
                        "type": "tool_result",
                        "tool_result": c.result,
                    }
                )
            elif c.type == "text" and c.text:
                content_items.append({"type": "text", "text": c.text})
        if content_items:
            evaluator_messages.append({"role": role, "content": content_items})
    return evaluator_messages


def display_evaluation_results(results: dict[str, dict]) -> None:
    """Display evaluation results in a formatted table using rich."""
    table = Table(title="Agent Evaluation Results", show_lines=True)
    table.add_column("Evaluator", style="cyan", width=28)
    table.add_column("Score", style="bold", justify="center", width=8)
    table.add_column("Result", justify="center", width=8)
    table.add_column("Reason", style="dim", width=70)

    for evaluator_name, result in results.items():
        score = str(result.get("score", "N/A"))
        pass_fail = result.get("result", "N/A")
        reason = result.get("reason", "N/A")

        if pass_fail == "pass":
            result_str = "[green]pass[/green]"
        elif pass_fail == "fail":
            result_str = "[red]fail[/red]"
        else:
            result_str = str(pass_fail)

        table.add_row(evaluator_name, score, result_str, reason)

    print()
    print(table)


async def main():
    if not ((azure_endpoint and azure_api_key and azure_deployment) or (llm_api_key and llm_base_url) or openai_api_key):
        logger.error(
            "No model configuration found. Set Azure OpenAI variables or LLM_BASE_URL/LLM_API_KEY/LLM_MODEL."
        )
        return

    query = "Plan a 3-day trip from New York (JFK) to Tokyo, departing March 15 and returning March 18, 2026. My budget is $2000 total. I like hiking and museums. Please search for flights, hotels under $150/night, check the weather, and suggest activities."

    logger.info("Running travel planner agent...")
    response = await agent.run(query)
    print(Panel(response.text, title="Agent Response", border_style="blue"))

    eval_query = [
        {"role": "system", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": [{"type": "text", "text": query}]},
    ]
    eval_response = convert_to_evaluator_messages(response.messages)

    ground_truth = (
        "A complete 3-day Tokyo trip itinerary from New York including: round-trip flight options with prices, "
        "hotel recommendations within nightly budget, hiking activities (e.g. Mt. Takao), museum visits "
        "(e.g. Tokyo National Museum, teamLab Borderless), weather forecast for the travel dates, "
        "a full cost breakdown showing total under $2000, and packing suggestions based on weather."
    )

    logger.info("Running agent evaluators...")

    evaluator_kwargs = {
        "model_config": eval_model_config,
        "is_reasoning_model": False,
        "max_tokens": 4096,
    }
    result_keys = {
        "IntentResolution": "intent_resolution",
        "ResponseCompleteness": "response_completeness",
        "TaskAdherence": "task_adherence",
        "ToolCallAccuracy": "tool_call_accuracy",
    }

    intent_evaluator = IntentResolutionEvaluator(**evaluator_kwargs)
    completeness_evaluator = ResponseCompletenessEvaluator(**evaluator_kwargs)
    adherence_evaluator = TaskAdherenceEvaluator(**evaluator_kwargs)
    tool_accuracy_evaluator = ToolCallAccuracyEvaluator(**evaluator_kwargs)

    intent_result = intent_evaluator(query=eval_query, response=eval_response, tool_definitions=tool_definitions)
    completeness_result = completeness_evaluator(response=response.text, ground_truth=ground_truth)
    adherence_result = adherence_evaluator(
        query=eval_query, response=eval_response, tool_definitions=tool_definitions
    )
    tool_accuracy_result = tool_accuracy_evaluator(
        query=eval_query, response=eval_response, tool_definitions=tool_definitions
    )

    evaluation_results = {}
    for name, result in [
        ("IntentResolution", intent_result),
        ("ResponseCompleteness", completeness_result),
        ("TaskAdherence", adherence_result),
        ("ToolCallAccuracy", tool_accuracy_result),
    ]:
        key = result_keys[name]
        evaluation_results[name] = {
            "score": result.get(key, "N/A"),
            "result": result.get(f"{key}_result", "N/A"),
            "reason": result.get(f"{key}_reason", result.get("error_message", "N/A")),
        }

    display_evaluation_results(evaluation_results)

if __name__ == "__main__":
    asyncio.run(main())
