from sqlalchemy import select

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
            record = session.scalar(
                select(LibraryItemRecord).where(
                    LibraryItemRecord.profile_id == profile_id,
                    LibraryItemRecord.media_type == item.media_type,
                    LibraryItemRecord.tmdb_id == item.tmdb_id,
                )
            )

            if record is None:
                record = LibraryItemRecord(
                    profile_id=profile_id,
                    media_type=item.media_type,
                    tmdb_id=item.tmdb_id,
                )
                session.add(record)

            record.title = item.title
            record.year = item.year
            record.watched = item.watched
            record.rating = item.rating
            record.review = item.review
            record.updated_at = utc_now()

        return _library_item(record)


def save_preference(profile_id, preference):
    with get_session() as session:
        with session.begin():
            record = session.get(PreferenceRecord, preference.id)

            if record is not None and record.profile_id != profile_id:
                record = None

            if record is None:
                record = session.scalar(
                    select(PreferenceRecord).where(
                        PreferenceRecord.profile_id == profile_id,
                        PreferenceRecord.text == preference.text,
                        PreferenceRecord.kind
                        == str(getattr(preference.kind, "value", preference.kind)),
                    )
                )

            kind = str(getattr(preference.kind, "value", preference.kind))

            if record is None:
                record = PreferenceRecord(
                    id=preference.id,
                    profile_id=profile_id,
                    text=preference.text,
                    kind=kind,
                    updated_at=utc_now(),
                )
                session.add(record)

            record.text = preference.text
            record.kind = kind
            record.source = preference.source
            record.source_title = preference.source_title
            record.source_media_type = preference.source_media_type
            record.source_tmdb_id = preference.source_tmdb_id
            record.updated_at = utc_now()

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
