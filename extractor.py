"""Reusable LLM extraction step: page text + URL → validated PageAnalysis."""

from __future__ import annotations

import anthropic
import instructor

from schemas import PageAnalysis

SYSTEM_PROMPT = (
    "You analyze web pages and produce structured JSON for downstream processing. "
    "Be precise, conservative with confidence scores, and never fabricate content "
    "not present in the page."
)

_client = instructor.from_anthropic(anthropic.Anthropic())


def extract_page(
    text: str,
    url: str,
    *,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 2048,
    max_retries: int = 2,
) -> PageAnalysis:
    """Call Claude to extract a PageAnalysis from `text`. Raises on validation failure."""
    return _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        response_model=PageAnalysis,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"<page_url>{url}</page_url>\n"
                f"<page_content>{text}</page_content>\n"
                "Extract a PageAnalysis matching the schema."
            ),
        }],
    )
