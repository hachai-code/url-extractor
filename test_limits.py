"""Verify each abuse-protection limit triggers correctly.

Run: `uv run python test_limits.py`. Exits non-zero on any failure. No real
LLM or network calls — uses TestClient against the FastAPI app, mocks fetch,
and unit-tests the pure check functions in isolation.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import api
from extractor_tools import (
    CostBudgetExceeded,
    ToolCallBudgetExceeded,
    _check_cost_budget,
    _check_tool_budget,
)
from fetch import FetchResult


def _ok(label: str) -> None:
    print(f"  PASS  {label}")


def _reset_rate_limit_state() -> None:
    api._rate_limit_state.clear()


def test_text_size_check_unit() -> None:
    """Direct call: small text passes, oversize text raises 413."""
    api._check_text_size("small text")
    _ok("text_size: small text passes")

    big = "x" * (api.MAX_TEXT_BYTES + 1)
    try:
        api._check_text_size(big)
    except HTTPException as exc:
        assert exc.status_code == 413, f"expected 413, got {exc.status_code}"
        assert exc.detail["size_bytes"] > api.MAX_TEXT_BYTES
        _ok("text_size: oversize text raises 413")
    else:
        raise AssertionError("expected HTTPException for oversize text")


def test_text_size_via_endpoint() -> None:
    """/extract returns 413 when the fetched page exceeds MAX_TEXT_BYTES."""
    _reset_rate_limit_state()
    huge_text = "x " * (api.MAX_TEXT_BYTES // 2 + 1)
    fake_fetch = FetchResult(
        url="https://example.com/huge",
        final_url="https://example.com/huge",
        status_code=200,
        text=huge_text,
        title="Huge",
        word_count=len(huge_text.split()),
    )
    with patch("api.fetch_page", return_value=fake_fetch):
        with TestClient(api.app) as client:
            response = client.post(
                "/extract",
                json={"url": "https://example.com/huge", "force_refresh": True},
            )
    assert response.status_code == 413, f"expected 413, got {response.status_code}"
    body = response.json()
    assert body["detail"]["error"] == "page_too_large"
    _ok("text_size: /extract returns 413 on huge page")


def test_rate_limit_unit() -> None:
    """Sliding window allows N requests, blocks the (N+1)th, with retry_after."""
    _reset_rate_limit_state()
    ip = "192.0.2.1"
    for i in range(api.RATE_LIMIT_REQUESTS):
        ok, retry = api._check_rate_limit(ip)
        assert ok, f"request {i + 1} should be allowed"
        assert retry == 0
    ok, retry = api._check_rate_limit(ip)
    assert not ok, "should be rate limited"
    assert retry > 0, f"retry_after must be positive, got {retry}"
    _ok(f"rate_limit: first {api.RATE_LIMIT_REQUESTS} pass; next blocked with retry={retry}s")


def test_rate_limit_per_ip_isolation() -> None:
    """Different IPs maintain independent windows."""
    _reset_rate_limit_state()
    for _ in range(api.RATE_LIMIT_REQUESTS):
        api._check_rate_limit("10.0.0.1")
    ok, _ = api._check_rate_limit("10.0.0.2")
    assert ok, "different IP should not be blocked"
    _ok("rate_limit: per-IP isolation works")


def test_rate_limit_via_endpoint() -> None:
    """/healthz isn't rate-limited; /extract returns 429 after the limit."""
    _reset_rate_limit_state()
    # Force every request to fast-fail on fetch so we don't try real LLM calls
    # — rate limiting still runs before fetch, so this is sufficient.
    fake_fetch = FetchResult(
        url="https://example.com/x",
        error="http_status",
        error_detail="HTTP 418",
    )
    with patch("api.fetch_page", return_value=fake_fetch):
        with TestClient(api.app) as client:
            # No rate-limit on /healthz
            for _ in range(api.RATE_LIMIT_REQUESTS + 2):
                assert client.get("/healthz").status_code == 200
            # /extract gets rate-limited after the threshold
            for i in range(api.RATE_LIMIT_REQUESTS):
                r = client.post("/extract", json={"url": "https://example.com/x", "force_refresh": True})
                assert r.status_code != 429, f"request {i + 1} unexpectedly rate-limited"
            r = client.post("/extract", json={"url": "https://example.com/x", "force_refresh": True})
            assert r.status_code == 429, f"expected 429, got {r.status_code}"
            assert r.headers.get("Retry-After"), "missing Retry-After header"
    _ok("rate_limit: /extract returns 429 after limit, /healthz unaffected")


def test_tool_call_budget() -> None:
    """_check_tool_budget passes below limit, raises at/above limit."""
    _check_tool_budget(0, 5)
    _check_tool_budget(4, 5)
    try:
        _check_tool_budget(5, 5)
    except ToolCallBudgetExceeded:
        _ok("tool_call_budget: raises when calls >= limit")
    else:
        raise AssertionError("expected ToolCallBudgetExceeded at limit")


def test_cost_budget() -> None:
    """_check_cost_budget passes below limit, raises above."""
    _check_cost_budget(0.49, 0.50)
    try:
        _check_cost_budget(0.51, 0.50)
    except CostBudgetExceeded:
        _ok("cost_budget: raises when spend > limit")
    else:
        raise AssertionError("expected CostBudgetExceeded over budget")


TESTS = [
    test_text_size_check_unit,
    test_text_size_via_endpoint,
    test_rate_limit_unit,
    test_rate_limit_per_ip_isolation,
    test_rate_limit_via_endpoint,
    test_tool_call_budget,
    test_cost_budget,
]


def main() -> int:
    failures = 0
    for test in TESTS:
        print(f"\n{test.__name__}:")
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {exc}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR  {type(exc).__name__}: {exc}")

    print()
    print(f"{'-' * 50}")
    print(f"Total: {len(TESTS)} tests, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
