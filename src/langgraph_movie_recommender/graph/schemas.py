from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from langgraph_movie_recommender.graph.state import Intent

Year = Annotated[int, Field(ge=1870, le=2100)]


class RecommendationFilters(BaseModel):
    media_type: Literal["movie", "tv", "both"] | None = None
    genres: list[str] | None = None
    year: Year | None = None
    year_from: Year | None = None
    year_to: Year | None = None
    max_runtime: Annotated[int, Field(ge=1, le=600)] | None = None
    query: str | None = None
    allow_rewatch: bool | None = None

    @field_validator("genres", mode="before")
    @classmethod
    def _normalize_genres(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("genres must be a list of names")

        genres = [genre.strip() for genre in value if isinstance(genre, str) and genre.strip()]
        return genres or None

    @model_validator(mode="after")
    def _validate_year_range(self):
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not be greater than year_to")
        return self


def normalize_filters(filters):
    if not filters:
        return {}

    return RecommendationFilters.model_validate(filters).model_dump(exclude_none=True)


class IntentClassification(BaseModel):
    intent: Intent
    filters: RecommendationFilters | None = Field(
        default=None,
        description="Recommendation filters, when relevant. Use null when there are no filters.",
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
    rating: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Explicit user rating on a 1-10 scale, or null.",
    )
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
