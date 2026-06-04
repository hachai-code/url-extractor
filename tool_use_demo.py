"""Minimal tool-use demo: the model calls a weather tool, we run it, the
model uses the result in its final answer. Single Python file, no frameworks
beyond the official Anthropic SDK.

Run:
    uv run python tool_use_demo.py
"""

from __future__ import annotations

import json

import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# 1. The local Python function the model can ask us to run.
#    Fake data — the point of the demo is the protocol, not real weather.
# ---------------------------------------------------------------------------
def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    fake = {"Berlin": (22, "cloudy"), "San Francisco": (16, "foggy"), "Tokyo": (28, "humid")}
    city = location.split(",")[0].strip()
    temp_c, conditions = fake.get(city, (20, "clear"))
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    return {"location": location, "unit": unit, "temperature": temp, "conditions": conditions}


# ---------------------------------------------------------------------------
# 2. The tool definition Claude sees.
#    The `description` is the main signal the model uses to decide when to
#    call this tool, so be specific about what it does AND what it doesn't.
#    `input_schema` is JSON Schema — same shape Pydantic's model_json_schema()
#    produces, which is why Instructor and Anthropic interop so cleanly.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a given city. Returns the temperature "
            "and a one-word description of conditions. Use this whenever the "
            "user asks about weather, temperature, or whether to dress warmly. "
            "The tool returns CURRENT conditions only — it does not forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country/state, e.g. 'Berlin, Germany' or 'San Francisco, CA'.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Default is fahrenheit.",
                },
            },
            "required": ["location"],
        },
    },
]

MODEL = "claude-haiku-4-5"
USER_QUESTION = "What's the weather like in Berlin right now? Use celsius please."


# ---------------------------------------------------------------------------
# 3. Turn 1: send the user's question along with the tool definitions.
#    `tool_choice` defaults to "auto" — the model decides whether to call
#    a tool or just answer. For this question it should choose to call.
# ---------------------------------------------------------------------------
messages = [{"role": "user", "content": USER_QUESTION}]
print(f"[user] {USER_QUESTION}\n")

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=TOOLS,
    messages=messages,
)

# stop_reason tells us why the model stopped generating:
#   "end_turn"   — model finished its answer normally
#   "tool_use"   — model wants us to run one or more tools
#   "max_tokens" — hit the budget; truncated
print(f"[turn 1] stop_reason={response.stop_reason}")
for block in response.content:
    if block.type == "text":
        print(f"  text:     {block.text}")
    elif block.type == "tool_use":
        print(f"  tool_use: id={block.id} name={block.name} input={block.input}")

assert response.stop_reason == "tool_use", "Expected the model to call get_weather"


# ---------------------------------------------------------------------------
# 4. Append the assistant turn — including the tool_use block — to the
#    conversation. The next message MUST follow this assistant turn; we
#    cannot send a tool_result in isolation.
# ---------------------------------------------------------------------------
messages.append({"role": "assistant", "content": response.content})


# ---------------------------------------------------------------------------
# 5. Find the tool_use block, run the matching Python function locally,
#    and build a tool_result block to send back.
#    The `tool_use_id` MUST match the id from the model's request — that's
#    how Claude pairs the result with the right invocation when there are
#    multiple parallel tool calls.
#    Note: tool_result blocks live inside a USER-role message. Anthropic's
#    protocol treats "tool output we're giving back to the model" as user
#    input, not as another assistant turn.
# ---------------------------------------------------------------------------
tool_use_block = next(b for b in response.content if b.type == "tool_use")
result = get_weather(**tool_use_block.input)
print(f"\n[local] get_weather({tool_use_block.input}) -> {result}")

messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": json.dumps(result),
        }
    ],
})


# ---------------------------------------------------------------------------
# 6. Turn 2: send the tool result back. The model now has the data and
#    produces a final natural-language answer for the user.
#    We still pass `tools=` — the model may decide to call another tool if
#    needed. With our trivial example it shouldn't.
# ---------------------------------------------------------------------------
response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=TOOLS,
    messages=messages,
)

print(f"\n[turn 2] stop_reason={response.stop_reason}")
for block in response.content:
    if block.type == "text":
        print(f"\n[assistant] {block.text}")
