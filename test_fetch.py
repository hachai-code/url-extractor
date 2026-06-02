"""Smoke-test fetch.py on 10 varied URLs.

Hits live sites sequentially with a polite User-Agent and a 15s timeout.
Run: `uv run python test_fetch.py`
"""

from __future__ import annotations

from fetch import fetch_page

URLS = [
    # --- expected to work ---
    ("news (BBC)",          "https://www.bbc.com/news/world-us-canada-68528282"),
    ("blog (Anthropic)",    "https://www.anthropic.com/news/claude-3-family"),
    ("blog (Simon W)",      "https://simonwillison.net/2024/Jun/20/claude-35-sonnet/"),
    ("docs (Python)",       "https://docs.python.org/3/library/asyncio-task.html"),
    ("docs (Pydantic)",     "https://docs.pydantic.dev/latest/concepts/models/"),
    ("Wikipedia",           "https://en.wikipedia.org/wiki/Anthropic"),
    # --- expected to be tricky ---
    ("GitHub README",       "https://github.com/anthropics/anthropic-sdk-python"),
    ("raw markdown",        "https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/README.md"),
    ("Substack",            "https://thezvi.substack.com/p/the-most-forbidden-technique"),
    ("NYT (paywall)",       "https://www.nytimes.com/2024/05/13/technology/openai-chatgpt-4o.html"),
]


def main() -> None:
    print(f"{'category':<20} {'status':<14} {'words':>6}  title / error")
    print("-" * 100)
    for label, url in URLS:
        result = fetch_page(url)
        if result.ok:
            status = f"OK ({result.status_code})"
            tail = (result.title or "<no title>")[:60]
        else:
            status = result.error or "?"
            tail = (result.error_detail or "")[:60]
        print(f"{label:<20} {status:<14} {result.word_count:>6}  {tail}")


if __name__ == "__main__":
    main()
