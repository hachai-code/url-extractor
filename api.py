"""FastAPI app exposing POST /extract: URL in → typed PageAnalysis JSON out."""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from extractor import extract_page
from fetch import fetch_page
from schemas import PageAnalysis

load_dotenv()
logger = logging.getLogger("url-extractor.api")
logging.basicConfig(level=logging.INFO)

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


@app.post("/extract", response_model=PageAnalysis)
def extract(request: ExtractRequest) -> PageAnalysis:
    fetched = fetch_page(str(request.url))
    if not fetched.ok:
        status = _FETCH_STATUS.get(fetched.error or "", 502)
        logger.warning("fetch failed url=%s error=%s detail=%s", request.url, fetched.error, fetched.error_detail)
        raise HTTPException(
            status_code=status,
            detail=ExtractError(stage="fetch", kind=fetched.error, detail=fetched.error_detail or "").model_dump(),
        )

    try:
        assert fetched.text is not None  # for type-checker; fetched.ok guarantees this
        return extract_page(fetched.text, str(fetched.final_url or request.url))
    except Exception as exc:
        logger.exception("extract failed url=%s", request.url)
        raise HTTPException(
            status_code=502,
            detail=ExtractError(stage="extract", kind=type(exc).__name__, detail=str(exc)).model_dump(),
        ) from exc


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
