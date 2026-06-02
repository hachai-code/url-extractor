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

_client = instructor.from_anthropic(anthropic.Anthropic())


def extract_page(
    text: str,
    url: str,
    *,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 8192,
    max_retries: int = 2,
) -> PageAnalysis:
    """Call Claude to extract a PageAnalysis from `text`. Raises on validation failure."""
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
