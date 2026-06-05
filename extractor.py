"""Reusable LLM extraction step: page text + URL → validated PageAnalysis.

Prompt structure follows Anthropic's guidance: XML-tagged sections for
<task>, <instructions>, <output_requirements>, and <examples>, with the
per-call <source_text> living in the user message. Few-shot examples
target the failure modes that hurt entity quality most: alias
deduplication, type discipline (organization vs product), calibrated
confidence, and not hallucinating Wikipedia titles.
"""

from __future__ import annotations

import anthropic
import instructor
from dotenv import load_dotenv

from schemas import PageAnalysis

load_dotenv()

ROLE = (
    "You analyze web pages and produce structured JSON for downstream "
    "processing. Be precise, conservative with confidence scores, and "
    "never fabricate content not present in the page."
)

SYSTEM_PROMPT = ROLE + """

<task>
Read the page provided in <source_text> and produce a PageAnalysis: a tight
summary, the named entities, an overall sentiment, suggested action items,
and the most important factual claims, each with a confidence score.
</task>

<instructions>
- Use only information that appears in the source text. Do not introduce
  facts from outside knowledge.
- For entities: deduplicate aliases (e.g. "Apple", "Apple Inc.", and
  "Apple Computer, Inc." are one entity). Use the most common surface form
  as the name. Choose the most specific type; use "other" only when no
  category fits.
- When the page is ABOUT a topic (encyclopedia entry, paper abstract,
  technical reference, tutorial), extract that topic as the first entity
  even if it's a common-noun concept like "photosynthesis", "large
  language model", or "fine-tuning". The page's subject is always an
  entity; do not skip it because it isn't a proper noun.
- For wikipedia_title: set this only when you are confident a Wikipedia
  article exists for this exact entity. Use null otherwise. Do not guess
  URL slugs.
- For sentiment: judge the page's primary content, not its comments,
  chrome, or sidebars.
- For action_items: extract ONLY explicit calls to action that the page
  directs at the reader (e.g. "sign up", "download X", "try the demo",
  "read the paper at <link>"). Return an empty list when the page is a
  reference, encyclopedia entry, news report, or review that does not ask
  the reader to do anything. Do not invent actions a reader "could" take.
- For key_claims: state each claim as a self-contained sentence.
  Set is_opinion=true when the source frames it as opinion or speculation.
- Calibrate confidence honestly: 0.95+ for facts stated plainly, 0.7-0.9
  for clearly implied, 0.5-0.7 for inferences. Do not default to 0.9.
</instructions>

<output_requirements>
- Match the PageAnalysis JSON schema exactly. Extra fields are rejected.
- Entity list: most prominent first; max 50.
- Summary: 2-4 sentences, neutral tone.
- Every confidence score must reflect actual certainty, not a placeholder.
</output_requirements>

<examples>
<example>
  <source_text>
    <page_url>https://example.com/anthropic-news</page_url>
    <page_content>
Anthropic, the AI safety lab, announced today that Claude Opus 4 is now
generally available. The model, designed in San Francisco, sets a new bar
for agentic coding tasks. CEO Dario Amodei said in a blog post that the
company will continue prioritizing safety research alongside capability
gains.
    </page_content>
  </source_text>
  <output_entities>
    [
      {"name": "Anthropic", "type": "organization", "mentions": 2, "context": "AI safety lab releasing the Claude model family.", "wikipedia_title": "Anthropic", "confidence": 0.98},
      {"name": "Claude Opus 4", "type": "product", "mentions": 1, "context": "Anthropic's flagship model for agentic coding tasks.", "wikipedia_title": null, "confidence": 0.95},
      {"name": "San Francisco", "type": "location", "mentions": 1, "context": "Where Claude Opus 4 was designed.", "wikipedia_title": "San Francisco", "confidence": 0.9},
      {"name": "Dario Amodei", "type": "person", "mentions": 1, "context": "CEO of Anthropic.", "wikipedia_title": "Dario Amodei", "confidence": 0.95}
    ]
  </output_entities>
</example>

<example>
  <source_text>
    <page_url>https://example.com/tim-cook-profile</page_url>
    <page_content>
Tim Cook, Apple's CEO, joined the company in 1998. Apple Inc. - formerly
Apple Computer, Inc. - is headquartered in Cupertino, California. The
firm reported record iPhone sales this quarter, driven by demand in Asia.
    </page_content>
  </source_text>
  <output_entities>
    [
      {"name": "Apple", "type": "organization", "mentions": 3, "context": "Technology company headquartered in Cupertino; the page also uses 'Apple Inc.' and 'Apple Computer, Inc.' for the same entity.", "wikipedia_title": "Apple Inc.", "confidence": 0.99},
      {"name": "Tim Cook", "type": "person", "mentions": 1, "context": "CEO of Apple who joined the company in 1998.", "wikipedia_title": "Tim Cook", "confidence": 0.99},
      {"name": "Cupertino", "type": "location", "mentions": 1, "context": "California city where Apple is headquartered.", "wikipedia_title": "Cupertino, California", "confidence": 0.95},
      {"name": "iPhone", "type": "product", "mentions": 1, "context": "Apple's smartphone product line.", "wikipedia_title": "IPhone", "confidence": 0.98}
    ]
  </output_entities>
</example>

<example>
  <source_text>
    <page_url>https://docs.example.org/asyncio</page_url>
    <page_content>
Python's asyncio module provides infrastructure for writing single-threaded
concurrent code using coroutines. It is part of the standard library since
Python 3.4, originally proposed by Guido van Rossum in PEP 3156.
    </page_content>
  </source_text>
  <output_entities>
    [
      {"name": "asyncio", "type": "product", "mentions": 1, "context": "Python's standard-library module for single-threaded concurrent code using coroutines.", "wikipedia_title": null, "confidence": 0.9},
      {"name": "Python", "type": "product", "mentions": 2, "context": "The programming language whose standard library includes asyncio.", "wikipedia_title": "Python (programming language)", "confidence": 0.98},
      {"name": "Guido van Rossum", "type": "person", "mentions": 1, "context": "Creator of Python who originally proposed asyncio.", "wikipedia_title": "Guido van Rossum", "confidence": 0.95},
      {"name": "PEP 3156", "type": "work_of_art", "mentions": 1, "context": "Python Enhancement Proposal that introduced asyncio.", "wikipedia_title": null, "confidence": 0.85}
    ]
  </output_entities>
</example>

<example>
  <source_text>
    <page_url>https://blog.example.com/q3-update</page_url>
    <page_content>
We talked to Alex from the partnerships team about the new arrangement,
though we couldn't confirm the surname. The deal involves an unnamed
Bay Area startup, rumored to be around $50M in size. Our sources at
"a major cloud provider" declined to comment publicly. Internally,
some team members refer to the project as "Project Lighthouse," though
this name has not been formally announced.
    </page_content>
  </source_text>
  <output_entities>
    [
      {"name": "Alex", "type": "person", "mentions": 1, "context": "Member of the partnerships team; surname not confirmed in the source.", "wikipedia_title": null, "confidence": 0.55},
      {"name": "Bay Area", "type": "location", "mentions": 1, "context": "Region where the unnamed startup is based.", "wikipedia_title": "San Francisco Bay Area", "confidence": 0.85},
      {"name": "Project Lighthouse", "type": "event", "mentions": 1, "context": "Internal project codename; informal usage, not formally announced.", "wikipedia_title": null, "confidence": 0.6}
    ]
  </output_entities>
  <notes>
The "unnamed Bay Area startup" and "a major cloud provider" are deliberately
not named in the source — do not invent names for them. "Alex" gets 0.55
because their identity is partially unknown. "Project Lighthouse" gets 0.6
because it's described as informal/unannounced. Use the full 0.5-0.7 range
for entities you can extract but where the source itself signals uncertainty.
  </notes>
</example>

<example>
  <source_text>
    <page_url>https://en.example.org/wiki/Photosynthesis</page_url>
    <page_content>
Photosynthesis is a biological process used by plants, algae, and certain
bacteria to convert light energy into chemical energy stored in carbohydrate
molecules. The process uses water and carbon dioxide as inputs and produces
glucose and oxygen as outputs. Most plants use chlorophyll, a pigment found
in chloroplasts, to absorb light during photosynthesis.
    </page_content>
  </source_text>
  <output_entities>
    [
      {"name": "photosynthesis", "type": "other", "mentions": 2, "context": "Biological process that converts light energy into chemical energy; the subject of this page.", "wikipedia_title": "Photosynthesis", "confidence": 0.99},
      {"name": "chlorophyll", "type": "other", "mentions": 1, "context": "Pigment in chloroplasts used to absorb light during photosynthesis.", "wikipedia_title": "Chlorophyll", "confidence": 0.9},
      {"name": "chloroplast", "type": "other", "mentions": 1, "context": "Plant cell organelle that contains chlorophyll.", "wikipedia_title": "Chloroplast", "confidence": 0.85}
    ]
  </output_entities>
  <notes>
The page's subject ("photosynthesis") is extracted as the first entity even
though it is a common-noun concept, not a proper noun. The same rule applies
to articles about "large language model", "RLHF", "fine-tuning", "attention
mechanism", or any other technical concept. If the page is about it, it
belongs in the entity list.
  </notes>
</example>

<example>
  <source_text>
    <page_url>https://en.example.org/wiki/Carl_Linnaeus</page_url>
    <page_content>
Carl Linnaeus (1707-1778) was a Swedish botanist, zoologist, and physician
who formalised binomial nomenclature, the modern system of naming organisms.
He is known as the "father of modern taxonomy." Linnaeus was born in the
village of Rashult in Sweden and studied medicine at Lund University and
Uppsala University, later becoming a professor at Uppsala.
    </page_content>
  </source_text>
  <output_action_items>
    []
  </output_action_items>
  <notes>
This is a reference page: it describes facts about Linnaeus but does not
ask the reader to do anything. action_items is an empty list. Do NOT
generate plausible-sounding actions like "Learn more about Linnaeus" or
"Visit Uppsala University" — the page does not direct the reader to do
these things. Only an explicit call to action in the source produces an
action_item entry. The same rule applies to news reports, encyclopedia
entries, and reviews: if no call to action, return [].
  </notes>
</example>
</examples>
"""

USER_TEMPLATE = """\
<source_text>
<page_url>{url}</page_url>
<page_content>
{text}
</page_content>
</source_text>

Produce a PageAnalysis from the source above, following the instructions
and output requirements.
"""

DEFAULT_MODEL = "claude-haiku-4-5"

_client = instructor.from_anthropic(anthropic.Anthropic())

# Stats from the most recent extract_page() call. Read after each call.
# `input_tokens` and `output_tokens` are summed across retries.
last_stats: dict = {
    "model": "-",
    "llm_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "parse_errors": [],
}


def _on_completion_response(response) -> None:
    last_stats["llm_calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    last_stats["input_tokens"] += getattr(usage, "input_tokens", 0)
    last_stats["output_tokens"] += getattr(usage, "output_tokens", 0)


def _on_parse_error(error) -> None:
    last_stats["parse_errors"].append(str(error)[:300])


_client.on("completion:response", _on_completion_response)
_client.on("parse:error", _on_parse_error)


def reset_stats() -> None:
    """Zero out last_stats. Call before any code path that may skip extract_page()."""
    last_stats.update(model="-", llm_calls=0, input_tokens=0, output_tokens=0, parse_errors=[])


def extract_page(
    text: str,
    url: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    max_retries: int = 3,
) -> PageAnalysis:
    """Call Claude to extract a PageAnalysis from `text`. Raises on validation failure."""
    reset_stats()
    last_stats["model"] = model
    return _client.messages.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        response_model=PageAnalysis,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(url=url, text=text),
        }],
    )


def extract_page_stream(
    text: str,
    url: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
):
    """Yield progressively-complete PageAnalysis snapshots as Claude generates them.

    Each yielded object has all fields Optional (Pydantic Partial semantics).
    Instructor does NOT run @model_validator / @field_validator on partials,
    so a streamed snapshot may violate invariants the non-streaming path
    enforces (sentiment label-score consistency, unique entity names). The
    caller streams these as best-effort intermediate state.
    """
    reset_stats()
    last_stats["model"] = model
    return _client.create_partial(
        response_model=PageAnalysis,
        model=model,
        max_tokens=max_tokens,
        stream=True,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": USER_TEMPLATE.format(url=url, text=text),
        }],
    )
