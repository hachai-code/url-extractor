"""Minimal Instructor + Anthropic extraction against the PageAnalysis schema."""

from __future__ import annotations

import anthropic
import instructor
from dotenv import load_dotenv

from schemas import ActionItem, Claim, Entity, PageAnalysis, Sentiment

load_dotenv()

URL = "https://www.anthropic.com/news/claude-4"
PAGE_TEXT = (
    "Claude 4 is now available. Anthropic, the AI safety company headquartered in San "
    "Francisco, released the new model family today, June 1, 2026. Claude Opus 4 sets "
    "a new state of the art on coding and agentic tasks, while Claude Sonnet 4 offers "
    "a strong balance of capability and cost. Developers can try Claude 4 in the Claude "
    "API starting today; sign up at console.anthropic.com to get access. Pricing remains "
    "the same as the Claude 3.5 generation. Early partner evaluations suggest substantial "
    "improvements on long-horizon agent tasks."
)

client = instructor.from_anthropic(anthropic.Anthropic())

result: PageAnalysis = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=2048,
    max_retries=2,
    response_model=PageAnalysis,
    system="You analyze web pages and produce structured JSON. Be precise, conservative with confidence, never fabricate.",
    messages=[{
        "role": "user",
        "content": f"<page_url>{URL}</page_url>\n<page_content>{PAGE_TEXT}</page_content>\nExtract a PageAnalysis.",
    }],
)

print(result.model_dump_json(indent=2))

assert isinstance(result, PageAnalysis)
assert str(result.url).rstrip("/") == URL.rstrip("/")
assert isinstance(result.sentiment, Sentiment) and -1.0 <= result.sentiment.score <= 1.0
assert all(isinstance(e, Entity) and 0.0 <= e.confidence <= 1.0 for e in result.entities)
assert all(isinstance(a, ActionItem) for a in result.action_items)
assert all(isinstance(c, Claim) for c in result.key_claims)
print(f"\nOK — {len(result.entities)} entities, {len(result.key_claims)} claims, {len(result.action_items)} actions, sentiment={result.sentiment.label.value}")
