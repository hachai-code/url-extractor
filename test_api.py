"""End-to-end test of POST /extract on 5 varied URLs using FastAPI's TestClient.

In-process (no separate server needed). Hits live URLs and the Anthropic API.
Run: `uv run python test_api.py`
"""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from api import app

URLS = [
    ("news/blog",      "https://www.anthropic.com/news/claude-3-family"),
    ("personal blog",  "https://simonwillison.net/2024/Jun/20/claude-35-sonnet/"),
    ("technical docs", "https://docs.python.org/3/library/asyncio-task.html"),
    ("Wikipedia",      "https://en.wikipedia.org/wiki/Anthropic"),
    ("paywalled (NYT)", "https://www.nytimes.com/2024/05/13/technology/openai-chatgpt-4o.html"),
]


def main() -> None:
    client = TestClient(app)
    for label, url in URLS:
        print("\n" + "=" * 80)
        print(f"[{label}]  {url}")
        t0 = time.monotonic()
        response = client.post("/extract", json={"url": url})
        elapsed = time.monotonic() - t0
        print(f"  status={response.status_code}  elapsed={elapsed:.1f}s")

        body = response.json()
        if response.status_code == 200:
            print(f"  title: {body['title']}")
            print(f"  summary: {body['summary'][:200]}...")
            print(f"  sentiment: {body['sentiment']['label']} (score={body['sentiment']['score']:.2f}, conf={body['sentiment']['confidence']:.2f})")
            print(f"  entities ({len(body['entities'])}):")
            for e in body["entities"][:5]:
                print(f"    - {e['name']:<30} type={e['type']:<14} conf={e['confidence']:.2f}")
            print(f"  claims ({len(body['key_claims'])}):")
            for c in body["key_claims"][:3]:
                opinion = " [OPINION]" if c.get("is_opinion") else ""
                print(f"    - {c['statement'][:90]}  (conf={c['confidence']:.2f}){opinion}")
            print(f"  action_items ({len(body['action_items'])}):")
            for a in body["action_items"][:3]:
                print(f"    - [{a['priority']}] {a['description'][:80]}")
        else:
            print(f"  ERROR: {json.dumps(body, indent=2)}")


if __name__ == "__main__":
    main()
