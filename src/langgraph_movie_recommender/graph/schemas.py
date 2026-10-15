from typing import Any, Literal

from pydantic import BaseModel, Field

from langgraph_movie_recommender.graph.state import Intent


class IntentClassification(BaseModel):
    intent: Intent
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Recommendation filters, when relevant. Supported keys include media_type (movie, tv, or both), genres (list of genre names), year, year_from, year_to, max_runtime, and query for a specific title or search phrase. Use null when there are no filters.",
    )


class FeedbackExtraction(BaseModel):
    title: str | None = Field(description="The title explicitly mentioned by the user, or null.")
    media_type: Literal["movie", "tv"] | None = Field(
        description="Movie or TV show, only when clear from the message."
    )
    year: int | None = Field(
        description="Release year or first-air year, only when stated or unambiguous."
    )
    watched: bool | None = Field(
        description="True if clearly watched, false if explicitly not watched, otherwise null."
    )
    rating: int | None = Field(description="Explicit user rating on a 1-10 scale, or null.")
    review: str | None = Field(
        description="The user's overall opinion or review, preserving their meaning."
    )
    positive_aspects: list[str] = Field(description="Specific aspects the user explicitly liked.")
    negative_aspects: list[str] = Field(
        description="Specific aspects the user explicitly disliked."
    )


class PreferenceStatement(BaseModel):
    text: str
    kind: Literal["like", "dislike"]


class PreferenceExtraction(BaseModel):
    preferences: list[PreferenceStatement] = Field(
        description="Only general preferences the user explicitly stated."
    )
