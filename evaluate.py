"""Run the extractor against eval_set.jsonl and produce a baseline report.

For each URL:
- Fetch + extract
- Compare extracted entity names to expected_entities (case-insensitive)
- Save full PageAnalysis + comparison + per-call stats to eval_results.json
- Print a compact per-row summary plus aggregate recall

Human-review-first: the JSON file is what you read for full diffs; the printed
summary is the at-a-glance. No LLM-as-judge — set-difference on entity names.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from extractor import extract_page, last_stats
from fetch import fetch_page

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
RESULTS = Path(__file__).parent / "eval_results.json"


def normalize(name: str) -> str:
    return name.casefold().strip().rstrip(".,")


def compare_entities(expected: list[str], extracted: list[str]) -> dict:
    exp = {normalize(e): e for e in expected}
    got = {normalize(e): e for e in extracted}
    matched = sorted(exp[k] for k in exp.keys() & got.keys())
    missing = sorted(exp[k] for k in exp.keys() - got.keys())
    extras = sorted(got[k] for k in got.keys() - exp.keys())
    return {
        "matched": matched,
        "missing": missing,
        "extras": extras,
        "recall": (len(matched) / len(exp)) if exp else 1.0,
    }


def evaluate_one(row: dict) -> dict:
    url = row["url"]
    out = {**row, "status": "ok"}
    t0 = time.monotonic()

    fetched = fetch_page(url)
    if not fetched.ok:
        out["status"] = "fetch_failed"
        out["error_detail"] = f"{fetched.error}: {fetched.error_detail}"
        return out

    try:
        result = extract_page(fetched.text, str(fetched.final_url or url))
    except Exception as exc:
        out["status"] = "extract_failed"
        out["error_detail"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        out["stats"] = {
            "llm_calls": last_stats["llm_calls"],
            "input_tokens": last_stats["input_tokens"],
            "output_tokens": last_stats["output_tokens"],
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "validation_errors": list(last_stats["parse_errors"]),
        }
        return out

    extracted_names = [e.name for e in result.entities]
    out["extraction"] = result.model_dump(mode="json")
    out["comparison"] = compare_entities(row["expected_entities"], extracted_names)
    out["stats"] = {
        "llm_calls": last_stats["llm_calls"],
        "input_tokens": last_stats["input_tokens"],
        "output_tokens": last_stats["output_tokens"],
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "validation_errors": list(last_stats["parse_errors"]),
    }
    return out


def main() -> None:
    rows = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]
    print(f"Evaluating {len(rows)} URLs...\n")
    results: list[dict] = []

    for i, row in enumerate(rows, 1):
        result = evaluate_one(row)
        results.append(result)

        if result["status"] == "ok":
            cmp = result["comparison"]
            line = f"[{i:>2}/{len(rows)}] {row['genre']:<10} recall={cmp['recall']:.2f}"
            if cmp["missing"]:
                line += f"  missing={cmp['missing']}"
            print(line)
        else:
            print(f"[{i:>2}/{len(rows)}] {row['genre']:<10} {result['status'].upper()}: {result.get('error_detail', '')[:80]}")

    RESULTS.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if r["status"] == "ok"]
    print("\n--- Summary ---")
    print(f"Successful extractions: {len(ok)}/{len(rows)}")
    if not ok:
        return

    avg_recall = sum(r["comparison"]["recall"] for r in ok) / len(ok)
    print(f"Average entity recall:  {avg_recall:.2f}")
    print(f"Total LLM calls:        {sum(r['stats']['llm_calls'] for r in ok)}")
    print(f"Total input tokens:     {sum(r['stats']['input_tokens'] for r in ok):,}")
    print(f"Total output tokens:    {sum(r['stats']['output_tokens'] for r in ok):,}")

    by_genre: dict = {}
    for r in ok:
        by_genre.setdefault(r["genre"], []).append(r["comparison"]["recall"])
    print("\nPer-genre recall:")
    for g, scores in sorted(by_genre.items()):
        print(f"  {g:<12} n={len(scores):>2}  recall={sum(scores) / len(scores):.2f}")

    print(f"\nFull results written to: {RESULTS.name}")


if __name__ == "__main__":
    main()
