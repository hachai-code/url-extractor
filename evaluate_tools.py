"""Evaluate the tool-enabled extractor on the first N URLs of eval_set.jsonl.

Prints per-row tool call traces, cost, and entity recall. Smaller N than the
non-tool evaluator because each extraction makes multiple LLM round-trips.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from db import cost_usd
from evaluate import compare_entities
from extractor_tools import extract_page_with_tools, last_stats
from fetch import fetch_page

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
N = 5


def main() -> None:
    rows = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()][:N]
    print(f"Tool-enabled extraction on first {N} URLs.\n")

    total_cost = 0.0
    total_llm_calls = 0
    total_tool_calls = 0
    recalls: list[float] = []

    for i, row in enumerate(rows, 1):
        url = row["url"]
        print(f"\n[{i}/{N}] {row['genre']:<10} {url}")

        fetched = fetch_page(url)
        if not fetched.ok:
            print(f"  fetch_failed: {fetched.error_detail}")
            continue

        t0 = time.monotonic()
        try:
            result = extract_page_with_tools(fetched.text, str(fetched.final_url or url))
        except Exception as exc:
            print(f"  EXTRACT_FAILED ({type(exc).__name__}): {str(exc)[:120]}")
            continue
        elapsed = time.monotonic() - t0

        c = cost_usd(last_stats["model"], last_stats["input_tokens"], last_stats["output_tokens"])
        total_cost += c
        total_llm_calls += last_stats["llm_calls"]
        total_tool_calls += last_stats["tool_calls"]

        print(
            f"  llm_calls={last_stats['llm_calls']} "
            f"tool_calls={last_stats['tool_calls']} "
            f"tokens=(in={last_stats['input_tokens']:,}, out={last_stats['output_tokens']:,}) "
            f"cost=${c:.3f} latency={elapsed:.1f}s"
        )
        for tc in last_stats["tool_call_log"]:
            input_preview = json.dumps(tc["input"])[:90]
            print(f"    -> {tc['name']}({input_preview})")

        extracted_names = [e.name for e in result.entities]
        cmp = compare_entities(row["expected_entities"], extracted_names)
        recalls.append(cmp["recall"])
        print(f"  recall={cmp['recall']:.2f}  missing={cmp['missing']}")
        print(f"  title: {result.title}")

    print("\n--- Aggregate ---")
    print(f"Successful extractions: {len(recalls)}/{N}")
    if recalls:
        print(f"Average entity recall:  {sum(recalls) / len(recalls):.2f}")
    print(f"Total LLM calls:        {total_llm_calls}")
    print(f"Total tool calls:       {total_tool_calls}")
    print(f"Total cost:             ${total_cost:.3f}")


if __name__ == "__main__":
    main()
