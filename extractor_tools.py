"""Tool-enabled extraction using Anthropic's native tool-use loop.

This path does NOT use Instructor. Structured output is achieved by treating
PageAnalysis as a final `submit_analysis` tool — the model calls enrichment
tools as needed, then invokes submit_analysis with the completed JSON.

Why this exists alongside extractor.py:
- extractor.py — fast path, single call, Instructor handles structured output
  with retries. Used by /extract and /extract/stream.
- extractor_tools.py — agentic path, model can enrich entities via Wikipedia
  and linked-article fetches before finalizing. Higher latency, higher cost,
  potentially higher recall on ambiguous entities.

The native loop here is intentional: it's the way to internalize how the
tool_use / tool_result protocol actually works, before delegating to a
library that hides it.
"""

from __future__ import annotations

import json
import time

import anthropic
from dotenv import load_dotenv
from langfuse import get_client
from pydantic import ValidationError

from extractor import SYSTEM_PROMPT, USER_TEMPLATE
from schemas import PageAnalysis
from tools import fetch_linked_article, lookup_wikipedia

load_dotenv()
_client = anthropic.Anthropic()

# Langfuse client. Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
# from env. When credentials are missing it logs a warning once and becomes a
# no-op, so the rest of this file works either way.
_langfuse = get_client()


# ---------------------------------------------------------------------------
# Tool definitions. Each `description` is taken from the Python function's
# docstring — those were written for an LLM reader in tools.py.
# ---------------------------------------------------------------------------
WIKIPEDIA_TOOL = {
    "name": "lookup_wikipedia",
    "description": lookup_wikipedia.__doc__ or "",
    "input_schema": {
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "Entity name to look up, e.g. 'Anthropic' or 'Photosynthesis'.",
            },
        },
        "required": ["entity_name"],
    },
}

FETCH_TOOL = {
    "name": "fetch_linked_article",
    "description": fetch_linked_article.__doc__ or "",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute HTTP(S) URL of the linked article.",
            },
        },
        "required": ["url"],
    },
}

# Final-step tool: the model calls this once with the completed PageAnalysis.
# Using PageAnalysis.model_json_schema() means the model sees the same JSON
# schema it would for Instructor's response_model — including additionalProperties:
# false and our enum constraints — but expressed through the tool-use protocol.
SUBMIT_TOOL = {
    "name": "submit_analysis",
    "description": (
        "Submit the final PageAnalysis once you have gathered enough context. "
        "Call this exactly once, at the end. Do not call this before you have "
        "decided whether to use the enrichment tools (lookup_wikipedia, "
        "fetch_linked_article) for ambiguous or important entities."
    ),
    "input_schema": PageAnalysis.model_json_schema(),
}

TOOLS = [WIKIPEDIA_TOOL, FETCH_TOOL, SUBMIT_TOOL]

# Dispatch table for enrichment tools (submit_analysis is handled inline).
TOOL_FUNCTIONS = {
    "lookup_wikipedia": lookup_wikipedia,
    "fetch_linked_article": fetch_linked_article,
}


# Module-level stats for the most recent extract_page_with_tools() call.
last_stats: dict = {
    "model": "-",
    "llm_calls": 0,
    "tool_calls": 0,
    "tool_call_log": [],
    "input_tokens": 0,
    "output_tokens": 0,
    "validation_errors": [],
}


def reset_stats() -> None:
    last_stats.update(
        model="-",
        llm_calls=0,
        tool_calls=0,
        tool_call_log=[],
        input_tokens=0,
        output_tokens=0,
        validation_errors=[],
    )


def _run_enrichment_tool(name: str, tool_input: dict) -> str:
    """Run an enrichment tool and return its JSON-string result for tool_result.

    Wrapped in a Langfuse span so we can see name, input, output, latency in
    the trace.
    """
    with _langfuse.start_as_current_observation(as_type="span", name=f"tool:{name}") as span:
        t0 = time.monotonic()
        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            output_str = json.dumps({"error": f"unknown tool: {name}"})
            span.update(input=tool_input, output=output_str, metadata={"unknown_tool": True})
            return output_str

        result = fn(**tool_input)
        latency_ms = int((time.monotonic() - t0) * 1000)
        output_obj = result.model_dump(mode="json") if result is not None else None
        span.update(
            input=tool_input,
            output=output_obj,
            metadata={"latency_ms": latency_ms, "found": result is not None},
        )
        return result.model_dump_json() if result is not None else "null"


def extract_page_with_tools(
    text: str,
    url: str,
    *,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 8192,
    max_iterations: int = 6,
) -> PageAnalysis:
    """Extract a PageAnalysis via a multi-step tool-use loop.

    The model may call lookup_wikipedia / fetch_linked_article any number of
    times to enrich its understanding, then must call submit_analysis with
    the completed PageAnalysis. A validation failure on submit_analysis is
    sent back as a tool error so the model can fix it on the next iteration.

    The whole loop is wrapped in a Langfuse trace, with one generation
    observation per LLM call and one span per tool call.
    """
    reset_stats()
    last_stats["model"] = model

    messages: list[dict] = [
        {"role": "user", "content": USER_TEMPLATE.format(url=url, text=text)},
    ]

    with _langfuse.start_as_current_observation(
        as_type="span",
        name="extract_page_with_tools",
        input={"url": url, "text_chars": len(text)},
        metadata={"model": model},
    ) as trace:
        try:
            for iteration in range(max_iterations):
                with _langfuse.start_as_current_observation(
                    as_type="generation",
                    name=f"llm_call_{iteration + 1}",
                    model=model,
                ) as gen:
                    response = _client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        tools=TOOLS,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                    )
                    last_stats["llm_calls"] += 1
                    last_stats["input_tokens"] += response.usage.input_tokens
                    last_stats["output_tokens"] += response.usage.output_tokens

                    gen.update(
                        input=messages,
                        output=[b.model_dump() for b in response.content],
                        usage_details={
                            "input": response.usage.input_tokens,
                            "output": response.usage.output_tokens,
                        },
                        metadata={"stop_reason": response.stop_reason},
                    )

                if response.stop_reason != "tool_use":
                    raise RuntimeError(
                        f"Model stopped without calling submit_analysis "
                        f"(stop_reason={response.stop_reason})"
                    )

                # Protocol: every tool_use must be followed by a matching tool_result.
                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict] = []
                final_result: PageAnalysis | None = None

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    if block.name == "submit_analysis":
                        try:
                            final_result = PageAnalysis(**block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": "Analysis accepted.",
                            })
                        except ValidationError as exc:
                            last_stats["validation_errors"].append(str(exc)[:300])
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(exc)[:1500],
                                "is_error": True,
                            })
                    else:
                        last_stats["tool_calls"] += 1
                        last_stats["tool_call_log"].append(
                            {"name": block.name, "input": block.input}
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _run_enrichment_tool(block.name, block.input),
                        })

                if final_result is not None:
                    trace.update(
                        output={
                            "entities": len(final_result.entities),
                            "claims": len(final_result.key_claims),
                            "title": final_result.title,
                        },
                        metadata={
                            "llm_calls": last_stats["llm_calls"],
                            "tool_calls": last_stats["tool_calls"],
                            "input_tokens": last_stats["input_tokens"],
                            "output_tokens": last_stats["output_tokens"],
                        },
                    )
                    return final_result

                messages.append({"role": "user", "content": tool_results})

            raise RuntimeError(
                f"Model did not call submit_analysis within {max_iterations} iterations"
            )
        finally:
            # Short-lived script: flush queued events before returning.
            _langfuse.flush()
