# Prompting notes

Notes from re-reading Anthropic's prompting guidance (June 2026). The old per-technique sub-pages (`be-clear-and-direct`, `multishot-prompting`, `chain-of-thought`, `use-xml-tags`, `system-prompts`) have been folded into one page: **Prompting best practices** at `platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices`. The overview page now just points to it. Everything below is from the consolidated page, applied to this project.

---

## What React/JS-dev intuition gets wrong

Three specific instincts I had to unlearn:

### 1. DRY is wrong here — restate context, redundantly

Coming from React, the instinct is: extract, factor, deduplicate. In prompts, the opposite. The docs frame it as "treat Claude like a brilliant but new employee." You wouldn't refactor your onboarding doc to remove "context repetition" — you'd be explicit about purpose, audience, end goal, and constraints, even when they overlap. The "less effective" / "more effective" pair in the docs makes this concrete: `Create an analytics dashboard` → `Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.` The verbose version is the right one.

### 2. "Tell what not to do" is wrong — always state the positive

JS devs reach for validation/rejection: "don't use ellipses," "no markdown," "never include preamble." The doc says explicitly: **positive examples beat negative instructions**.
- Instead of: `Do not use markdown in your response`
- Use: `Your response should be composed of smoothly flowing prose paragraphs.`

Quote: "Positive examples showing how Claude can communicate with the appropriate level of concision tend to be more effective than negative examples or instructions that tell the model what not to do."

### 3. Aggressive emphasis backfires now

Old tutorials taught `CRITICAL: You MUST...`. With Opus 4.5+, that **causes over-triggering** — the model is already responsive; aggressive language pushes it past where you want. Quote: "Where you might have said 'CRITICAL: You MUST use this tool when...', you can use more normal prompting like 'Use this tool when...'." Same logic for `ALWAYS`, `NEVER`, ALL-CAPS, repeated exclamation. Calm imperatives win.

### Honorable mentions (surprised me less but still notable)

- **Put long documents at the TOP, not the end.** "Queries at the end can improve response quality by up to 30%." React intuition says "recency = relevance" — wrong here.
- **Prompt style influences output style.** If you write your prompt in markdown, you get more markdown back. Match your prompt's format to what you want out.
- **Prefilled assistant responses are gone** in 4.6+. Anything tutorial-based that says "prefill `{` to force JSON" is dead. Use structured outputs instead.
- **`think step by step` is mostly obsolete** in adaptive-thinking models. The `effort` param and `thinking: {type: "adaptive"}` replace manual CoT for most use cases.

---

## Patterns I'll actually use for the URL extractor

These are the techniques that map directly to the work coming up (`extractor.py` and beyond).

### A. System prompt with a tight role, no theatrics

```text
You analyze web pages and produce structured JSON analyses for downstream
processing. You are precise, conservative with confidence scores, and never
fabricate content not supported by the page.
```

Single sentence is enough per the docs. No "you are the world's greatest analyst." No "CRITICAL."

### B. XML tags around every distinct content type

```xml
<page_url>{url}</page_url>
<page_content>
{cleaned HTML or extracted text}
</page_content>
<instructions>
Extract a PageAnalysis matching the schema below. ...
</instructions>
<schema>
{json.dumps(PageAnalysis.model_json_schema(), indent=2)}
</schema>
```

Per the docs: XML tags reduce misinterpretation when prompts mix instructions, context, examples, and inputs. Tag names should be consistent and descriptive across calls.

### C. Multishot examples (3–5), diverse, in `<example>` tags

The docs are explicit: 3–5 examples, wrapped in `<example>` tags inside an `<examples>` parent. Diverse cases matter — for this project: one news article (claims-heavy), one product landing page (action-items-heavy), one opinion essay (sentiment-heavy), one technical doc (entities-heavy). This forces the model to handle variation rather than memorize a single shape.

### D. Field descriptions ARE the prompt

`schemas.py` already encodes most of the prompt: every `Field(description=...)` is what the model sees in the schema. "Imperative-mood action," "neutral tone, no editorializing," "Verbatim quote from the page" — those *are* my instructions. The user-facing prompt only needs to cover what the schema can't say (e.g. how to choose between two plausible entity types).

### E. Tell, don't ask

For action-taking calls (the tool-use stage with Wikipedia lookups): not `Can you look up the entity on Wikipedia?` → use `Look up the entity on Wikipedia and enrich its context field.` From the docs: "If you say 'can you suggest some changes,' Claude will sometimes provide suggestions rather than implementing them."

### F. Self-check before returning

Append:
```text
Before finishing, verify that every confidence score reflects your actual
certainty (not a default), that no entity is duplicated, and that every claim
has a supporting quote when one exists in the page.
```
The docs call this out as reliable for catching errors, especially on structured tasks.

### G. Adaptive thinking, not manual CoT

For the extraction call, use `thinking: {type: "adaptive"}` with `effort: "high"`. No `<thinking>` scratchpad in the prompt — the API handles it. The docs are explicit: "In internal evaluations, adaptive thinking reliably drives better performance than extended thinking."

---

## Patterns I'm deliberately NOT using

- **No prefill.** Deprecated on 4.6+ — would 400.
- **No "think step by step" string.** Adaptive thinking covers it; manual CoT is for older models.
- **No CRITICAL / MUST / ALWAYS in caps.** Causes over-triggering.
- **No "you are an expert..." role inflation.** Single descriptive sentence beats a paragraph.
- **No per-field instructions in the user message.** They belong in the schema's field descriptions.
- **No JSON-formatting reminders** ("return valid JSON, no trailing commas, etc."). Structured outputs + Pydantic validation handle this; the model doesn't need the reminder.

---

## Concrete template for the extraction call

```text
SYSTEM:
You analyze web pages and produce structured JSON for downstream processing.
You are precise, conservative with confidence scores, and never fabricate
content not present in the page.

USER:
<page_url>{url}</page_url>

<page_content>
{extracted_text}
</page_content>

<examples>
<example>
  <input>{abbreviated example page 1}</input>
  <output>{example PageAnalysis JSON 1}</output>
</example>
<!-- 2-4 more, covering article / landing / opinion / technical -->
</examples>

<instructions>
Produce a PageAnalysis object matching the provided schema. Use the page
content as the only source of truth — do not introduce facts not present
in the page. Confidence scores should reflect how clearly each item is
supported by the page text.

Before finishing, verify that no entity name is duplicated, every claim's
supporting_quote (if present) appears verbatim in the page, and confidence
scores are not default placeholders.
</instructions>
```

That's the prompt I'll port into `extractor.py` when we build it.
