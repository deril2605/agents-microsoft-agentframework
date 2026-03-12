import json
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv


def normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.rstrip("/")
    if stripped.endswith("/openai/v1"):
        return f"{stripped}/"
    if "openai.azure.com" in stripped:
        return f"{stripped}/openai/v1/"
    return stripped


# Configure Azure OpenAI client based on environment
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
MODEL_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("LLM_MODEL") or "gpt-4.1-mini"
client = openai.OpenAI(
    base_url=normalize_base_url(os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BASE_URL")),
    api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
)
STDOUT_ENCODING = sys.stdout.encoding or "utf-8"


def lookup_weather(city_name: str | None = None, zip_code: str | None = None) -> dict:
    """Lookup the weather for a given city name or zip code."""
    message = f"Looking up weather for {city_name or zip_code}...\n"
    print(message.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))
    return {
        "city_name": city_name,
        "zip_code": zip_code,
        "weather": "sunny",
        "temperature": 75,
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Lookup the weather for a given city name or zip code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The city name",
                    },
                    "zip_code": {
                        "type": "string",
                        "description": "The zip code",
                    },
                },
                "additionalProperties": False,
            },
        },
    }
]

messages = [
    {"role": "system", "content": "You are a weather chatbot."},
    {"role": "user", "content": "is it sunny in LA, CA?"},
]
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",
)


# Now actually call the function as indicated
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_arguments = json.loads(tool_call.function.arguments)
    print(function_name)
    print(function_arguments)

    if function_name == "lookup_weather":
        messages.append(response.choices[0].message)
        result = lookup_weather(**function_arguments)
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
        print(f"Response from {MODEL_NAME}:")
        print(response.choices[0].message.content.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))

else:
    print(response.choices[0].message.content.encode(STDOUT_ENCODING, errors="replace").decode(STDOUT_ENCODING))
