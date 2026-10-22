from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LibraryItemRecord(Base):
    __tablename__ = "library_items"
    __table_args__ = (
        UniqueConstraint(
            "media_type",
            "tmdb_id",
            name="uq_library_media_tmdb",
        ),
        CheckConstraint("media_type IN ('movie', 'tv')", name="ck_library_media_type"),
        CheckConstraint("tmdb_id > 0", name="ck_library_tmdb_id"),
        CheckConstraint(
            "rating IS NULL OR (rating BETWEEN 1 AND 10)",
            name="ck_library_rating",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    media_type: Mapped[str] = mapped_column(String, nullable=False)
    tmdb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    year: Mapped[int | None] = mapped_column(Integer)
    watched: Mapped[bool] = mapped_column(default=False, nullable=False)
    rating: Mapped[int | None] = mapped_column(Integer)
    review: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PreferenceRecord(Base):
    __tablename__ = "preferences"
    __table_args__ = (
        UniqueConstraint(
            "text",
            "kind",
            name="uq_preference_text_kind",
        ),
        CheckConstraint(
            "kind IN ('like', 'dislike', 'neutral')",
            name="ck_preference_kind",
        ),
        CheckConstraint(
            "source IN ('explicit', 'inferred')",
            name="ck_preference_source",
        ),
        CheckConstraint(
            "source_media_type IS NULL OR source_media_type IN ('movie', 'tv')",
            name="ck_preference_source_media_type",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, default="explicit", nullable=False)
    source_title: Mapped[str | None] = mapped_column(String)
    source_media_type: Mapped[str | None] = mapped_column(String)
    source_tmdb_id: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
