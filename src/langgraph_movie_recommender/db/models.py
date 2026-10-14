from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LibraryItemRecord(Base):
    __tablename__ = "library_items"
    __table_args__ = (
        UniqueConstraint(
            "profile_id",
            "media_type",
            "tmdb_id",
            name="uq_library_profile_media_tmdb",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
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
            "profile_id",
            "text",
            "kind",
            name="uq_preference_profile_text_kind",
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    profile_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, default="explicit", nullable=False)
    source_title: Mapped[str | None] = mapped_column(String)
    source_media_type: Mapped[str | None] = mapped_column(String)
    source_tmdb_id: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
