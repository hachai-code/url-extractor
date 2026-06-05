"""Tools the extractor LLM can call.

Each function is a tool the model invokes via Anthropic's tool-use protocol.
The docstring is what the model sees as the tool's description — write them
for the model, not for IDE tooltips. Keep return shapes small and JSON-safe.

Conventions:
- Returns None on any failure path. The LLM gets `null` and decides what to do
  next. We do not surface internal error reasons — they cost tokens without
  changing the LLM's behavior.
- Returns are Pydantic models so callers (and the tool-result serializer) get
  consistent JSON.
"""

from __future__ import annotations

import urllib.parse

import httpx
from pydantic import BaseModel, Field, HttpUrl

from fetch import fetch_page

WIKIPEDIA_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_USER_AGENT = "url-extractor/0.1 (contact: mainstream998@googlemail.com)"
WIKIPEDIA_TIMEOUT = 10.0

MAX_EXCERPT_WORDS = 500


class WikipediaLookup(BaseModel):
    """Successful Wikipedia summary lookup result."""

    title: str = Field(..., description="Canonical Wikipedia article title.")
    description: str | None = Field(
        default=None,
        description="Short subtitle from Wikidata, e.g. 'AI safety company'. May be absent.",
    )
    summary: str = Field(..., description="Plain-text intro paragraph (~1–3 sentences).")
    url: HttpUrl = Field(..., description="Canonical Wikipedia article URL.")
    is_disambiguation: bool = Field(
        default=False,
        description="True when the page lists multiple meanings rather than a single subject.",
    )


def lookup_wikipedia(entity_name: str) -> WikipediaLookup | None:
    """Look up a named entity on English Wikipedia and return its canonical
    title, a short description, the intro paragraph, and the page URL.

    Use this whenever you need to (a) disambiguate an entity — e.g. is
    "Anthropic" the AI company or a literary device? — or (b) verify that a
    Wikipedia title you guessed actually exists and refers to the right
    subject. Returns null when Wikipedia has no matching article.

    The query is case-insensitive but should be reasonably specific:
    "Anthropic" works; "AI company" does not. For ambiguous queries Wikipedia
    may return a disambiguation page (is_disambiguation=true); when that
    happens, re-query with a more specific name (e.g. "Claude (AI assistant)"
    instead of "Claude") if you can construct one.
    """
    encoded = urllib.parse.quote(entity_name.strip().replace(" ", "_"))
    try:
        response = httpx.get(
            WIKIPEDIA_SUMMARY_API.format(title=encoded),
            headers={"User-Agent": WIKIPEDIA_USER_AGENT, "Accept": "application/json"},
            timeout=WIKIPEDIA_TIMEOUT,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    data = response.json()
    extract = data.get("extract")
    if not extract:
        return None
    return WikipediaLookup(
        title=data["title"],
        description=data.get("description"),
        summary=extract,
        url=data["content_urls"]["desktop"]["page"],
        is_disambiguation=data.get("type") == "disambiguation",
    )


class ArticleSummary(BaseModel):
    """Lightweight result for a fetched linked article."""

    url: HttpUrl = Field(..., description="The article URL after redirects.")
    title: str | None = Field(default=None, description="Page title if available.")
    excerpt: str = Field(..., description=f"First ~{MAX_EXCERPT_WORDS} words of main content.")
    word_count: int = Field(..., ge=0, description="Total word count of the full extracted article.")


def fetch_linked_article(url: str) -> ArticleSummary | None:
    """Fetch a linked article and return its title plus the first ~500 words
    of cleaned main content.

    Use this when the page you are analyzing links to another article whose
    contents would meaningfully change your understanding — for example, when
    a blog post cites a paper or news article and the citation alone isn't
    enough to evaluate the claim or sentiment. The excerpt is truncated to
    limit context cost; it is NOT the full article, and you should not rely
    on it to make claims about the article's later sections.

    Returns null on any fetch failure: 4xx/5xx responses, paywalled pages,
    JavaScript-rendered pages with no extractable text, or non-HTML content
    types. When this happens, do not retry with the same URL — assume the
    article is unavailable.
    """
    result = fetch_page(url)
    if not result.ok or result.text is None:
        return None
    words = result.text.split()
    excerpt = " ".join(words[:MAX_EXCERPT_WORDS])
    return ArticleSummary(
        url=str(result.final_url or url),
        title=result.title,
        excerpt=excerpt,
        word_count=result.word_count,
    )


if __name__ == "__main__":
    print("=== lookup_wikipedia ===")
    for name in ["Anthropic", "Claude (AI assistant)", "Photosynthesis", "NonexistentXYZ12345"]:
        result = lookup_wikipedia(name)
        print(f"\nquery: {name!r}")
        print(result.model_dump_json(indent=2) if result else "  -> None")

    print("\n\n=== fetch_linked_article ===")
    for url in [
        "https://simonwillison.net/2024/Jun/20/claude-35-sonnet/",
        "https://www.nytimes.com/2024/05/13/technology/openai-chatgpt-4o.html",
    ]:
        result = fetch_linked_article(url)
        print(f"\nurl: {url}")
        if result:
            preview = result.model_dump()
            preview["excerpt"] = preview["excerpt"][:200] + "..."
            print(preview)
        else:
            print("  -> None")
