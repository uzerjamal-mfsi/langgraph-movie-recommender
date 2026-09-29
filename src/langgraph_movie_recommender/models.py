from typing import Literal

from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    tmdb_id: int
    media_type: Literal["movie", "tv"]
    title: str
    year: int | None = None
    overview: str = ""
    genres: list[str] = Field(default_factory=list)
    tmdb_url: str
