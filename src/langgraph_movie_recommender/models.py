from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MediaType = Literal["movie", "tv"]
PreferenceKind = Literal["like", "dislike", "neutral"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class MediaItem(BaseModel):
    tmdb_id: int
    media_type: MediaType
    title: str
    year: int | None = None
    overview: str = ""
    genres: list[str] = Field(default_factory=list)
    vote_average: float = 0.0
    popularity: float = 0.0
    tmdb_url: str


class LibraryItem(BaseModel):
    media_type: MediaType
    tmdb_id: int = Field(gt=0)
    title: str = Field(min_length=1)
    year: int | None = None
    watched: bool = False
    rating: int | None = Field(default=None, ge=1, le=10)
    review: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title cannot be blank")
        return cleaned

    @field_validator("updated_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class Preference(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: PreferenceKind
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("id", "text")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value cannot be blank")
        return cleaned

    @field_validator("updated_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return _normalize_utc(value)
