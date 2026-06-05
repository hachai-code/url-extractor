"""SQLite log of extraction calls: tokens, cost, latency, retries, errors.

One table, three functions. Use `init_db()` once at app start, `log_call()`
after each request, and `stats()` to read aggregates.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "logs.db"

# Per-million-token USD prices (base input, output).
# Source: platform.claude.com/docs/en/docs/about-claude/pricing (2026-06-03).
# Cache-read and cache-write pricing not handled here yet — add when we enable
# prompt caching. cache_read = 0.1× input, cache_write_5m = 1.25× input.
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":  (1.00,  5.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-7":   (5.00, 25.00),
    "claude-opus-4-8":   (5.00, 25.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost from token counts. Returns 0.0 for unknown model IDs."""
    if model not in PRICES:
        return 0.0
    in_rate, out_rate = PRICES[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def init_db() -> None:
    """Create the extractions table if it doesn't exist. Safe to call repeatedly."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS extractions (
                id                INTEGER PRIMARY KEY,
                ts                TEXT    NOT NULL,
                url               TEXT    NOT NULL,
                model             TEXT    NOT NULL,
                input_tokens      INTEGER NOT NULL,
                output_tokens     INTEGER NOT NULL,
                cost_usd          REAL    NOT NULL,
                latency_ms        INTEGER NOT NULL,
                llm_calls         INTEGER NOT NULL,
                validation_errors TEXT    NOT NULL,
                status            TEXT    NOT NULL,
                error_detail      TEXT
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS extractions_ts ON extractions(ts)")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses_saved (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                url        TEXT NOT NULL,
                analysis   TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS extractions_cache (
                cache_key   TEXT PRIMARY KEY,
                url         TEXT NOT NULL,
                model       TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                analysis    TEXT NOT NULL
            )
            """
        )


def save_analysis(url: str, analysis: dict) -> str:
    """Persist a completed analysis and return its short share id."""
    aid = secrets.token_urlsafe(8)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "INSERT INTO analyses_saved (id, created_at, url, analysis) VALUES (?, ?, ?, ?)",
            (aid, datetime.now(timezone.utc).isoformat(), url, json.dumps(analysis)),
        )
    return aid


def _cache_key(url: str, model: str, prompt_hash: str) -> str:
    """Stable hash of (url, model, prompt) — same inputs always yield same key."""
    return hashlib.sha256(f"{url}|{model}|{prompt_hash}".encode()).hexdigest()[:32]


def get_cached(
    url: str,
    model: str,
    prompt_hash: str,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> dict | None:
    """Return cached analysis if it exists and is fresher than ttl_seconds, else None."""
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=ttl_seconds)).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT analysis FROM extractions_cache WHERE cache_key = ? AND created_at >= ?",
            (_cache_key(url, model, prompt_hash), cutoff),
        ).fetchone()
    return None if row is None else json.loads(row[0])


def put_cached(url: str, model: str, prompt_hash: str, analysis: dict) -> None:
    """Insert or replace a cache entry."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO extractions_cache
                (cache_key, url, model, prompt_hash, created_at, analysis)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                _cache_key(url, model, prompt_hash),
                url,
                model,
                prompt_hash,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(analysis),
            ),
        )


def get_analysis(aid: str) -> dict | None:
    """Fetch a saved analysis by id, or None if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT url, analysis, created_at FROM analyses_saved WHERE id = ?",
            (aid,),
        ).fetchone()
    if row is None:
        return None
    return {"url": row[0], "analysis": json.loads(row[1]), "created_at": row[2]}


def log_call(
    *,
    url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    llm_calls: int,
    validation_errors: list[str],
    status: str,
    error_detail: str | None = None,
) -> None:
    """Insert one row describing the just-completed request."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            INSERT INTO extractions
                (ts, url, model, input_tokens, output_tokens, cost_usd,
                 latency_ms, llm_calls, validation_errors, status, error_detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                url,
                model,
                input_tokens,
                output_tokens,
                cost_usd(model, input_tokens, output_tokens),
                latency_ms,
                llm_calls,
                json.dumps(validation_errors),
                status,
                error_detail,
            ),
        )


def stats(since_iso: str | None = None) -> dict:
    """Aggregate stats for rows with ts >= since_iso. Default: start of today UTC."""
    if since_iso is None:
        today = datetime.now(timezone.utc).date()
        since_iso = datetime(today.year, today.month, today.day, tzinfo=timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(cost_usd), 0.0),
                   COALESCE(SUM(input_tokens), 0),
                   COALESCE(SUM(output_tokens), 0),
                   COALESCE(AVG(latency_ms), 0.0),
                   COALESCE(SUM(CASE WHEN llm_calls > 1 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END), 0)
            FROM extractions
            WHERE ts >= ?
            """,
            (since_iso,),
        ).fetchone()

    requests, total_cost, in_tokens, out_tokens, avg_latency, retried, failed = row
    return {
        "since": since_iso,
        "requests": requests,
        "total_cost_usd": round(total_cost, 6),
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "avg_latency_ms": round(avg_latency, 1),
        "validation_failure_rate": (retried / requests) if requests else 0.0,
        "failure_rate": (failed / requests) if requests else 0.0,
    }
