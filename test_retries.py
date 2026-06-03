"""Run the extractor on 10 varied URLs and log how often retries happen.

Reads `extractor.last_stats` after each call to find out how many LLM
round-trips Instructor needed and which validation errors triggered them.
"""

from __future__ import annotations

import time

from extractor import extract_page, last_stats
from fetch import fetch_page

URLS = [
    ("anthropic news (clean)",   "https://www.anthropic.com/news/claude-3-family"),
    ("simon willison blog",      "https://simonwillison.net/2024/Jun/20/claude-35-sonnet/"),
    ("python docs",              "https://docs.python.org/3/library/asyncio-task.html"),
    ("wikipedia anthropic",      "https://en.wikipedia.org/wiki/Anthropic"),
    ("wikipedia pydantic",       "https://en.wikipedia.org/wiki/Pydantic_(library)"),
    ("paul graham essay",        "https://paulgraham.com/wisdom.html"),
    ("arxiv abstract",           "https://arxiv.org/abs/2403.09611"),
    ("BBC news article",         "https://www.bbc.com/news/technology-69013882"),
    ("github readme",            "https://github.com/anthropics/anthropic-sdk-python"),
    ("NYT paywall (expect fail)", "https://www.nytimes.com/2024/05/13/technology/openai-chatgpt-4o.html"),
]


def main() -> None:
    print(f"{'#':>2}  {'label':<28} {'fetch':<10} {'calls':>5}  {'time':>6}  result")
    print("-" * 100)
    totals = {"runs": 0, "single_attempt": 0, "retried": 0, "extraction_failed": 0, "fetch_failed": 0}
    all_errors: list[tuple[str, str]] = []

    for i, (label, url) in enumerate(URLS, 1):
        fetched = fetch_page(url)
        if not fetched.ok:
            totals["fetch_failed"] += 1
            print(f"{i:>2}  {label:<28} {fetched.error or '?':<10} {'-':>5}  {'-':>6}  {fetched.error_detail or ''}")
            continue

        t0 = time.monotonic()
        try:
            result = extract_page(fetched.text, str(fetched.final_url or url))  # type: ignore[arg-type]
            elapsed = time.monotonic() - t0
            calls = last_stats["llm_calls"]
            errors = list(last_stats["parse_errors"])
            totals["runs"] += 1
            if calls > 1:
                totals["retried"] += 1
                for err in errors:
                    all_errors.append((label, err))
            else:
                totals["single_attempt"] += 1
            tag = "OK" if calls <= 1 else f"OK (retried×{calls - 1})"
            print(f"{i:>2}  {label:<28} {'200':<10} {calls:>5}  {elapsed:>5.1f}s  {tag}: {result.title[:40]}")
        except Exception as exc:
            elapsed = time.monotonic() - t0
            totals["extraction_failed"] += 1
            print(f"{i:>2}  {label:<28} {'200':<10} {'?':>5}  {elapsed:>5.1f}s  FAILED: {type(exc).__name__}: {str(exc)[:60]}")

    print("-" * 100)
    print(f"\nTotals over {len(URLS)} URLs:")
    print(f"  successful extractions:        {totals['single_attempt'] + totals['retried']}/{len(URLS)}")
    print(f"  single-attempt success:        {totals['single_attempt']}")
    print(f"  succeeded only after retry:    {totals['retried']}")
    print(f"  extraction failed (gave up):   {totals['extraction_failed']}")
    print(f"  fetch failed (never reached):  {totals['fetch_failed']}")

    if all_errors:
        print(f"\nValidation errors that triggered retries ({len(all_errors)}):")
        for label, err in all_errors:
            print(f"  - [{label}]")
            for line in err.splitlines()[:3]:
                print(f"      {line}")


if __name__ == "__main__":
    main()
