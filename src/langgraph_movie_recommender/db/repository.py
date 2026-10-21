from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from langgraph_movie_recommender.db.models import (
    LibraryItemRecord,
    PreferenceRecord,
)
from langgraph_movie_recommender.db.session import get_session
from langgraph_movie_recommender.models import (
    LibraryItem,
    Preference,
    utc_now,
)

DEFAULT_PROFILE_ID = "default"


def _library_item(record):
    return LibraryItem(
        media_type=record.media_type,
        tmdb_id=record.tmdb_id,
        title=record.title,
        year=record.year,
        watched=record.watched,
        rating=record.rating,
        review=record.review,
        updated_at=record.updated_at,
    )


def _preference(record):
    return Preference(
        id=record.id,
        text=record.text,
        kind=record.kind,
        source=record.source,
        source_title=record.source_title,
        source_media_type=record.source_media_type,
        source_tmdb_id=record.source_tmdb_id,
        updated_at=record.updated_at,
    )


def get_library(profile_id=DEFAULT_PROFILE_ID):
    with get_session() as session:
        records = session.scalars(
            select(LibraryItemRecord)
            .where(LibraryItemRecord.profile_id == profile_id)
            .order_by(LibraryItemRecord.title)
        ).all()

        return [_library_item(record) for record in records]


def get_preferences(profile_id=DEFAULT_PROFILE_ID):
    with get_session() as session:
        records = session.scalars(
            select(PreferenceRecord)
            .where(PreferenceRecord.profile_id == profile_id)
            .order_by(PreferenceRecord.text)
        ).all()

        return [_preference(record) for record in records]


def find_library_item(profile_id, media_type, tmdb_id):
    with get_session() as session:
        record = session.scalar(
            select(LibraryItemRecord).where(
                LibraryItemRecord.profile_id == profile_id,
                LibraryItemRecord.media_type == media_type,
                LibraryItemRecord.tmdb_id == tmdb_id,
            )
        )

        return _library_item(record) if record else None


def save_library_item(profile_id, item):
    with get_session() as session:
        with session.begin():
            stmt = sqlite_insert(LibraryItemRecord).values(
                profile_id=profile_id,
                media_type=item.media_type,
                tmdb_id=item.tmdb_id,
                title=item.title,
                year=item.year,
                watched=item.watched,
                rating=item.rating,
                review=item.review,
                updated_at=utc_now(),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["profile_id", "media_type", "tmdb_id"],
                set_={
                    "title": stmt.excluded.title,
                    "year": stmt.excluded.year,
                    "watched": stmt.excluded.watched,
                    "rating": stmt.excluded.rating,
                    "review": stmt.excluded.review,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            session.execute(stmt)

            record = session.scalar(
                select(LibraryItemRecord).where(
                    LibraryItemRecord.profile_id == profile_id,
                    LibraryItemRecord.media_type == item.media_type,
                    LibraryItemRecord.tmdb_id == item.tmdb_id,
                )
            )

        return _library_item(record)


def save_preference(profile_id, preference):
    kind = str(getattr(preference.kind, "value", preference.kind))
    source = str(getattr(preference.source, "value", preference.source))
    source_media_type = getattr(preference.source_media_type, "value", None)
    if source_media_type is None:
        source_media_type = preference.source_media_type

    with get_session() as session:
        with session.begin():
            stmt = sqlite_insert(PreferenceRecord).values(
                id=preference.id,
                profile_id=profile_id,
                text=preference.text,
                kind=kind,
                source=source,
                source_title=preference.source_title,
                source_media_type=source_media_type,
                source_tmdb_id=preference.source_tmdb_id,
                updated_at=utc_now(),
            )
            update_values = {
                "text": stmt.excluded.text,
                "kind": stmt.excluded.kind,
                "source": stmt.excluded.source,
                "source_title": stmt.excluded.source_title,
                "source_media_type": stmt.excluded.source_media_type,
                "source_tmdb_id": stmt.excluded.source_tmdb_id,
                "updated_at": stmt.excluded.updated_at,
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_=update_values,
                where=PreferenceRecord.profile_id == profile_id,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["profile_id", "text", "kind"],
                set_=update_values,
            )
            session.execute(stmt)

            record = session.scalar(
                select(PreferenceRecord).where(
                    PreferenceRecord.profile_id == profile_id,
                    PreferenceRecord.text == preference.text,
                    PreferenceRecord.kind == kind,
                )
            )

        return _preference(record)


def get_watched_ids(profile_id=DEFAULT_PROFILE_ID):
    with get_session() as session:
        records = session.execute(
            select(
                LibraryItemRecord.media_type,
                LibraryItemRecord.tmdb_id,
            ).where(
                LibraryItemRecord.profile_id == profile_id,
                LibraryItemRecord.watched.is_(True),
            )
        ).all()

        return {(media_type, tmdb_id) for media_type, tmdb_id in records}
