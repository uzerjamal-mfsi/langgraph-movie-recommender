from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class LibraryItemRecord(Base):
    __tablename__ = "library_items"
    __table_args__ = (
        CheckConstraint("media_type IN ('movie', 'tv')"),
        CheckConstraint("tmdb_id > 0"),
        CheckConstraint("length(trim(title)) > 0"),
        CheckConstraint("watched IN (0, 1)"),
        CheckConstraint("rating IS NULL OR rating BETWEEN 1 AND 10"),
    )

    media_type = Column(String(5), primary_key=True)
    tmdb_id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    year = Column(Integer)
    watched = Column(Boolean, nullable=False, default=False, index=True)
    rating = Column(Integer)
    review = Column(Text)
    updated_at = Column(DateTime(timezone=True), nullable=False)


class PreferenceRecord(Base):
    __tablename__ = "preferences"
    __table_args__ = (
        CheckConstraint("length(trim(id)) > 0"),
        CheckConstraint("length(trim(text)) > 0"),
        CheckConstraint("kind IN ('like', 'dislike', 'constraint')"),
    )

    id = Column(String(64), primary_key=True)
    text = Column(Text, nullable=False)
    kind = Column(String(16), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False)
