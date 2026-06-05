"""FastAPI app exposing POST /extract: URL in → typed PageAnalysis JSON out."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, HttpUrl

import hashlib

from db import (
    get_analysis,
    get_cached,
    init_db,
    log_call,
    put_cached,
    save_analysis,
    stats as db_stats,
)
from extractor import (
    DEFAULT_MODEL,
    SYSTEM_PROMPT,
    extract_page,
    extract_page_stream,
    last_stats,
    reset_stats,
)
from fetch import fetch_page
from schemas import PageAnalysis

load_dotenv()

# Quiet noisy stdlib loggers (httpx, anthropic, etc.); use structlog for app code.
logging.basicConfig(level=logging.WARNING)
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.dev.ConsoleRenderer(),
    ],
    cache_logger_on_first_use=True,
)
log = structlog.get_logger("api")
init_db()

app = FastAPI(title="url-extractor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["content-type"],
)


class ExtractRequest(BaseModel):
    url: HttpUrl = Field(..., description="The page to fetch and analyze.")
    force_refresh: bool = Field(
        default=False,
        description="If true, skip the cache and re-extract from the page.",
    )


# Hash of the system prompt — when the prompt changes, cache keys change too,
# so old entries naturally fall out without us having to invalidate them.
PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


class ExtractError(BaseModel):
    stage: str = Field(..., description="Which pipeline stage failed: 'fetch' or 'extract'.")
    kind: str | None = None
    detail: str


# Map fetch error kinds to appropriate HTTP status codes.
_FETCH_STATUS = {
    "network": 502,        # we couldn't reach the target — bad gateway
    "http_status": 502,    # target returned an error to us
    "too_large": 413,      # too big to process
    "no_content": 422,     # page fetched but unextractable
    "unsupported": 415,    # wrong content type
}


def _log(url: str, t0: float, *, status: str, error_detail: str | None = None) -> None:
    """Persist the current request's telemetry. Reads from extractor.last_stats."""
    log_call(
        url=url,
        model=last_stats["model"],
        input_tokens=last_stats["input_tokens"],
        output_tokens=last_stats["output_tokens"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        llm_calls=last_stats["llm_calls"],
        validation_errors=list(last_stats["parse_errors"]),
        status=status,
        error_detail=error_detail,
    )


@app.post("/extract", response_model=PageAnalysis)
def extract(request: ExtractRequest) -> PageAnalysis:
    t0 = time.monotonic()
    url = str(request.url)
    rlog = log.bind(request_id=uuid.uuid4().hex[:8], url=url)
    reset_stats()
    rlog.info("request_received", force_refresh=request.force_refresh)

    if not request.force_refresh:
        cached = get_cached(url, DEFAULT_MODEL, PROMPT_HASH)
        if cached is not None:
            _log(url, t0, status="ok")  # zero tokens / zero cost from reset_stats
            rlog.info("cache_hit", latency_ms=int((time.monotonic() - t0) * 1000))
            return PageAnalysis.model_validate(cached)

    fetched = fetch_page(url)
    if not fetched.ok:
        _log(url, t0, status="fetch_failed", error_detail=f"{fetched.error}: {fetched.error_detail}")
        status = _FETCH_STATUS.get(fetched.error or "", 502)
        rlog.warning("fetch_failed", error=fetched.error, detail=fetched.error_detail)
        raise HTTPException(
            status_code=status,
            detail=ExtractError(stage="fetch", kind=fetched.error, detail=fetched.error_detail or "").model_dump(),
        )

    try:
        assert fetched.text is not None  # fetched.ok guarantees this
        result = extract_page(fetched.text, str(fetched.final_url or url))
        _log(url, t0, status="ok")
        put_cached(url, DEFAULT_MODEL, PROMPT_HASH, result.model_dump(mode="json"))
        rlog.info(
            "extract_ok",
            llm_calls=last_stats["llm_calls"],
            input_tokens=last_stats["input_tokens"],
            output_tokens=last_stats["output_tokens"],
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        return result
    except Exception as exc:
        _log(url, t0, status="extract_failed", error_detail=f"{type(exc).__name__}: {str(exc)[:300]}")
        rlog.exception("extract_failed", kind=type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail=ExtractError(stage="extract", kind=type(exc).__name__, detail=str(exc)).model_dump(),
        ) from exc


@app.post("/extract/stream")
async def extract_stream(request: ExtractRequest) -> StreamingResponse:
    """Same pipeline as /extract, but streams partial PageAnalysis snapshots as
    Server-Sent Events. Each `data:` line is the JSON of the model so far.
    A final `event: done` line marks the end. On extraction error, an
    `event: error` line is sent and the stream closes.

    Note: partial snapshots skip Pydantic validators; the final snapshot may
    violate invariants the non-streaming /extract enforces. Use /extract when
    you need full validation; use /extract/stream for progressive UI.
    """
    t0 = time.monotonic()
    url = str(request.url)
    rlog = log.bind(request_id=uuid.uuid4().hex[:8], url=url)
    reset_stats()
    rlog.info("stream_request_received", force_refresh=request.force_refresh)

    if not request.force_refresh:
        cached = get_cached(url, DEFAULT_MODEL, PROMPT_HASH)
        if cached is not None:
            _log(url, t0, status="ok")  # zero tokens / zero cost
            rlog.info("cache_hit", latency_ms=int((time.monotonic() - t0) * 1000))
            saved_id = save_analysis(url, cached)

            def cached_stream():
                yield f"data: {json.dumps(cached)}\n\n"
                yield f"event: done\ndata: {json.dumps({'id': saved_id})}\n\n"

            return StreamingResponse(cached_stream(), media_type="text/event-stream")

    fetched = fetch_page(url)
    if not fetched.ok:
        _log(url, t0, status="fetch_failed", error_detail=f"{fetched.error}: {fetched.error_detail}")
        rlog.warning("fetch_failed", error=fetched.error, detail=fetched.error_detail)
        status_code = _FETCH_STATUS.get(fetched.error or "", 502)
        raise HTTPException(
            status_code=status_code,
            detail=ExtractError(stage="fetch", kind=fetched.error, detail=fetched.error_detail or "").model_dump(),
        )

    async def event_stream():
        last_partial: dict | None = None
        sentinel = object()
        try:
            assert fetched.text is not None
            # iter() handles both real iterators and plain iterables (Instructor
            # returns a list on some paths instead of a generator).
            iterator = iter(extract_page_stream(fetched.text, str(fetched.final_url or url)))
            loop = asyncio.get_running_loop()
            # Pull from the sync Instructor iterator inside an executor so each
            # yield can flush through the async response stream in real time.
            while True:
                partial = await loop.run_in_executor(None, next, iterator, sentinel)
                if partial is sentinel:
                    break
                last_partial = partial.model_dump(mode="json")
                yield f"data: {json.dumps(last_partial)}\n\n"
            _log(url, t0, status="ok")
            rlog.info(
                "stream_complete",
                llm_calls=last_stats["llm_calls"],
                input_tokens=last_stats["input_tokens"],
                output_tokens=last_stats["output_tokens"],
                latency_ms=int((time.monotonic() - t0) * 1000),
            )
            saved_id = save_analysis(url, last_partial) if last_partial else None
            if last_partial is not None:
                put_cached(url, DEFAULT_MODEL, PROMPT_HASH, last_partial)
            yield f"event: done\ndata: {json.dumps({'id': saved_id})}\n\n"
        except Exception as exc:
            _log(url, t0, status="extract_failed", error_detail=f"{type(exc).__name__}: {str(exc)[:300]}")
            rlog.exception("stream_failed", kind=type(exc).__name__)
            yield f"event: error\ndata: {json.dumps({'kind': type(exc).__name__, 'detail': str(exc)[:200]})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/analyses/{aid}")
def get_saved_analysis(aid: str) -> dict:
    """Fetch a previously-saved analysis by its share id."""
    record = get_analysis(aid)
    if record is None:
        raise HTTPException(status_code=404, detail={"error": "analysis not found"})
    return record


@app.get("/metrics")
def metrics(since: str | None = None) -> dict:
    """Aggregate stats. Defaults to today UTC; pass ?since=YYYY-MM-DDTHH:MM:SS+00:00 to override."""
    return db_stats(since)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
