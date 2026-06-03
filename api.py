"""FastAPI app exposing POST /extract: URL in → typed PageAnalysis JSON out."""

from __future__ import annotations

import logging
import time
import uuid

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from db import init_db, log_call, stats as db_stats
from extractor import extract_page, last_stats, reset_stats
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


class ExtractRequest(BaseModel):
    url: HttpUrl = Field(..., description="The page to fetch and analyze.")


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
    rlog.info("request_received")

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


@app.get("/metrics")
def metrics(since: str | None = None) -> dict:
    """Aggregate stats. Defaults to today UTC; pass ?since=YYYY-MM-DDTHH:MM:SS+00:00 to override."""
    return db_stats(since)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
