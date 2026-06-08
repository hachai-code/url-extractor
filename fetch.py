"""Fetch a URL and extract its main-content text plus metadata.

Returns a `FetchResult` whose `error` field is set on any failure path
(network, HTTP status, or empty extraction). Callers check `result.text`
before sending it to the LLM.
"""

from __future__ import annotations

import threading
from typing import Literal

import httpx
import trafilatura
from pydantic import BaseModel, Field, HttpUrl

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 url-extractor/0.1"
)
DEFAULT_TIMEOUT = 15.0
MAX_BYTES = 5_000_000  # 5 MB — anything bigger is almost certainly not an article

# Cap concurrent fetch+parse work. trafilatura builds an lxml DOM that can use
# ~10x the page size in RAM, so a handful of large pages in flight at once can
# OOM a small instance (e.g. Render's 512 MB tier). This bounds peak memory no
# matter how many requests arrive. Raise it on bigger instances.
MAX_CONCURRENT_FETCHES = 3
_fetch_slots = threading.BoundedSemaphore(MAX_CONCURRENT_FETCHES)


ErrorKind = Literal[
    "network",          # connect/read timeout, DNS, TLS, etc.
    "http_status",      # 4xx/5xx
    "too_large",        # body exceeded MAX_BYTES
    "no_content",       # trafilatura returned nothing usable
    "unsupported",      # content-type wasn't HTML
]


class FetchResult(BaseModel):
    """Result of attempting to fetch and extract a single URL."""

    url: HttpUrl
    final_url: HttpUrl | None = Field(default=None, description="After redirects.")
    status_code: int | None = None
    text: str | None = Field(default=None, description="Main-content plain text; None on failure.")
    title: str | None = None
    author: str | None = None
    date: str | None = Field(default=None, description="Publication date (YYYY-MM-DD when available).")
    sitename: str | None = None
    word_count: int = 0
    error: ErrorKind | None = None
    error_detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def fetch_page(url: str, *, timeout: float = DEFAULT_TIMEOUT) -> FetchResult:
    """Fetch `url` and return main content + metadata, or a populated error.

    Holds one of MAX_CONCURRENT_FETCHES slots for the whole fetch+parse, so a
    burst of large pages can't exhaust memory on a small instance.
    """
    with _fetch_slots:
        return _fetch(url, timeout=timeout)


def _fetch(url: str, *, timeout: float) -> FetchResult:
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"},
        ) as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        return FetchResult(url=url, error="network", error_detail=f"{type(exc).__name__}: {exc}")

    if response.status_code >= 400:
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            error="http_status",
            error_detail=f"HTTP {response.status_code}",
        )

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.lower():
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            error="unsupported",
            error_detail=f"content-type: {content_type or '<missing>'}",
        )

    if len(response.content) > MAX_BYTES:
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            error="too_large",
            error_detail=f"{len(response.content)} bytes",
        )

    extracted = trafilatura.bare_extraction(
        response.text,
        url=str(response.url),
        favor_precision=True,
        include_comments=False,
        include_tables=True,
        with_metadata=True,
    )
    if extracted is None or not getattr(extracted, "text", None):
        return FetchResult(
            url=url,
            final_url=str(response.url),
            status_code=response.status_code,
            error="no_content",
            error_detail="trafilatura returned no main text",
        )

    text = extracted.text.strip()
    return FetchResult(
        url=url,
        final_url=str(response.url),
        status_code=response.status_code,
        text=text,
        title=getattr(extracted, "title", None),
        author=getattr(extracted, "author", None),
        date=getattr(extracted, "date", None),
        sitename=getattr(extracted, "sitename", None),
        word_count=len(text.split()),
    )


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "https://www.anthropic.com/news/claude-4"
    result = fetch_page(target)
    if result.ok:
        print(f"[OK] {result.title}  ({result.word_count} words)")
        print(f"     site={result.sitename}  date={result.date}  author={result.author}")
        print(f"     first 240 chars: {result.text[:240]}...")  # type: ignore[index]
    else:
        print(f"[FAIL:{result.error}] {result.error_detail}  ({target})")
