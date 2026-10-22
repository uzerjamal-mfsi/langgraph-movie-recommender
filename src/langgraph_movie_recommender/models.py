from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MediaType = Literal["movie", "tv"]
PreferenceKind = Literal["like", "dislike", "neutral"]
PreferenceSource = Literal["explicit", "inferred"]


class DomainModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def utc_now() -> datetime:
    return datetime.now(UTC)


def watched_id(media_type: str, tmdb_id: int) -> str:
    return f"{media_type}:{tmdb_id}"


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class MediaItem(DomainModel):
    tmdb_id: int
    media_type: MediaType
    title: str
    year: int | None = None
    overview: str = ""
    genres: list[str] = Field(default_factory=list)
    vote_average: float = 0.0
    popularity: float = 0.0
    tmdb_url: str


class LibraryItem(DomainModel):
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


class Preference(DomainModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: PreferenceKind
    source: PreferenceSource = "explicit"
    source_title: str | None = None
    source_media_type: MediaType | None = None
    source_tmdb_id: int | None = Field(default=None, gt=0)
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
