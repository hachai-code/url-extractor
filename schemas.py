"""Pydantic schemas for URL-to-structured-data extraction.

These models define the contract between the LLM and our application.
The JSON schemas generated from these classes are sent to the model
as the structured-output spec; field descriptions become prompt context.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    EVENT = "event"
    WORK_OF_ART = "work_of_art"
    OTHER = "other"


class SentimentLabel(str, Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Reusable confidence type: 0.0–1.0, inclusive.
Confidence = Annotated[
    float,
    Field(
        ge=0.0,
        le=1.0,
        description="Model's confidence in this item, from 0.0 (no confidence) to 1.0 (certain).",
        examples=[0.82],
    ),
]


class Entity(BaseModel):
    """A named entity mentioned in the page."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Canonical surface form of the entity as it appears in the page.",
        examples=["Anthropic", "San Francisco", "Claude 4"],
    )
    type: EntityType = Field(
        ...,
        description="Coarse category of the entity. Use 'other' only when no other category fits.",
    )
    mentions: int = Field(
        default=1,
        ge=1,
        le=10_000,
        description="Approximate number of times this entity appears in the page.",
        examples=[3],
    )
    context: str | None = Field(
        default=None,
        max_length=500,
        description="One short sentence describing the role this entity plays in the page.",
        examples=["AI safety company that publishes the Claude model family."],
    )
    wikipedia_title: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Best guess at the matching Wikipedia article title, if one likely exists. "
            "Null when the entity is unlikely to have a Wikipedia page."
        ),
        examples=["Anthropic"],
    )
    confidence: Confidence


_LABEL_BANDS: dict["SentimentLabel", tuple[float, float]] = {
    SentimentLabel.VERY_NEGATIVE: (-1.0, -0.6),
    SentimentLabel.NEGATIVE:      (-0.6, -0.2),
    SentimentLabel.NEUTRAL:       (-0.2,  0.2),
    SentimentLabel.POSITIVE:      ( 0.2,  0.6),
    SentimentLabel.VERY_POSITIVE: ( 0.6,  1.0),
}


class Sentiment(BaseModel):
    """Overall sentiment of the page's primary content."""

    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel = Field(
        ...,
        description="Coarse sentiment bucket for the page as a whole.",
    )
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Continuous sentiment score from -1.0 (very negative) to +1.0 (very positive).",
        examples=[0.35],
    )
    rationale: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="One- or two-sentence justification for the chosen label, citing tone or word choice.",
    )
    confidence: Confidence

    @model_validator(mode="after")
    def _label_matches_score(self) -> "Sentiment":
        low, high = _LABEL_BANDS[self.label]
        if not (low <= self.score <= high):
            raise ValueError(
                f"sentiment.score={self.score} is inconsistent with label={self.label.value!r} "
                f"(expected [{low}, {high}]). Either fix the label or the score."
            )
        return self


class ActionItem(BaseModel):
    """A concrete next step a reader could take after consuming the page."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    description: str = Field(
        ...,
        min_length=4,
        max_length=300,
        description="Imperative-mood action a reader could take, e.g. 'Sign up for the beta'.",
        examples=["Read the linked policy paper before commenting."],
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="How important this action seems relative to others on the page.",
    )
    deadline: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Free-form deadline as written on the page (e.g. 'by Friday', '2026-07-01'). "
            "Null if the page does not give one."
        ),
        examples=["2026-07-01"],
    )
    confidence: Confidence


class Claim(BaseModel):
    """A factual claim asserted by the page, with the model's confidence."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    statement: str = Field(
        ...,
        min_length=8,
        max_length=500,
        description="The claim, rewritten as a single self-contained sentence.",
        examples=["Anthropic was founded in 2021."],
    )
    supporting_quote: str | None = Field(
        default=None,
        max_length=500,
        description="Verbatim quote from the page that supports the claim, when available.",
    )
    is_opinion: bool = Field(
        default=False,
        description="True when the claim is framed as opinion or speculation rather than fact.",
    )
    confidence: Confidence


class PageAnalysis(BaseModel):
    """Root model: the full structured analysis of a single web page."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    url: HttpUrl = Field(
        ...,
        description="The canonical URL that was analyzed.",
        examples=["https://www.anthropic.com/news/claude-4"],
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Page title as it appears in <title> or the dominant H1.",
    )
    summary: str = Field(
        ...,
        min_length=40,
        max_length=1200,
        description=(
            "Tight 2–4 sentence summary of the page's main content. "
            "Neutral tone, no editorializing."
        ),
    )
    language: str = Field(
        default="en",
        min_length=2,
        max_length=8,
        description="BCP-47 language tag of the dominant page language.",
        examples=["en", "en-US", "fr"],
    )
    entities: list[Entity] = Field(
        default_factory=list,
        max_length=50,
        description="Named entities found in the page, most prominent first.",
    )
    sentiment: Sentiment = Field(
        ...,
        description="Overall sentiment of the page's primary content.",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        max_length=20,
        description="Concrete next steps suggested by or implied by the page.",
    )
    key_claims: list[Claim] = Field(
        default_factory=list,
        max_length=20,
        description="Most important factual claims asserted by the page.",
    )

    @field_validator("entities")
    @classmethod
    def _unique_entity_names(cls, value: list[Entity]) -> list[Entity]:
        seen: set[str] = set()
        for entity in value:
            key = entity.name.casefold()
            if key in seen:
                raise ValueError(f"Duplicate entity name: {entity.name!r}")
            seen.add(key)
        return value


if __name__ == "__main__":
    import json

    models: list[type[BaseModel]] = [
        Entity,
        Sentiment,
        ActionItem,
        Claim,
        PageAnalysis,
    ]
    for model in models:
        print(f"\n=== {model.__name__} ===")
        print(json.dumps(model.model_json_schema(), indent=2))
