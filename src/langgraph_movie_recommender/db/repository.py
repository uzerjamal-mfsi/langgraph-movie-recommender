from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from langgraph_movie_recommender.db.models import LibraryItemRecord, PreferenceRecord
from langgraph_movie_recommender.db.session import get_session
from langgraph_movie_recommender.models import LibraryItem, Preference, utc_now


def _upsert_library_item(session, item):
    stmt = sqlite_insert(LibraryItemRecord).values(
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
        index_elements=["media_type", "tmdb_id"],
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

    return session.scalar(
        select(LibraryItemRecord).where(
            LibraryItemRecord.media_type == item.media_type,
            LibraryItemRecord.tmdb_id == item.tmdb_id,
        )
    )


def _upsert_preference(session, preference):
    existing = session.scalar(
        select(PreferenceRecord).where(
            (PreferenceRecord.id == preference.id)
            | (
                (PreferenceRecord.text == preference.text)
                & (PreferenceRecord.kind == preference.kind)
            )
        )
    )
    if existing:
        existing.text = preference.text
        existing.kind = preference.kind
        existing.source = preference.source
        existing.source_title = preference.source_title
        existing.source_media_type = preference.source_media_type
        existing.source_tmdb_id = preference.source_tmdb_id
        existing.updated_at = utc_now()
        session.flush()
        return existing

    record = PreferenceRecord(
        id=preference.id,
        text=preference.text,
        kind=preference.kind,
        source=preference.source,
        source_title=preference.source_title,
        source_media_type=preference.source_media_type,
        source_tmdb_id=preference.source_tmdb_id,
        updated_at=utc_now(),
    )
    session.add(record)
    session.flush()
    return record


def get_library():
    with get_session() as session:
        records = session.scalars(select(LibraryItemRecord).order_by(LibraryItemRecord.title)).all()

        return [LibraryItem.model_validate(record) for record in records]


def get_preferences():
    with get_session() as session:
        records = session.scalars(select(PreferenceRecord).order_by(PreferenceRecord.text)).all()

        return [Preference.model_validate(record) for record in records]


def find_library_item(media_type, tmdb_id):
    with get_session() as session:
        record = session.scalar(
            select(LibraryItemRecord).where(
                LibraryItemRecord.media_type == media_type,
                LibraryItemRecord.tmdb_id == tmdb_id,
            )
        )

        return LibraryItem.model_validate(record) if record else None


def save_library_item(item):
    with get_session() as session, session.begin():
        record = _upsert_library_item(session, item)
        saved = LibraryItem.model_validate(record)

    return saved


def save_preference(preference):
    with get_session() as session, session.begin():
        record = _upsert_preference(session, preference)
        saved = Preference.model_validate(record)

    return saved


def save_preferences(preferences):
    with get_session() as session, session.begin():
        records = [_upsert_preference(session, preference) for preference in preferences]
        saved = [Preference.model_validate(record) for record in records]

    return saved


def save_feedback_and_preferences(item, preferences):
    with get_session() as session, session.begin():
        item_record = _upsert_library_item(session, item)
        preference_records = [_upsert_preference(session, preference) for preference in preferences]
        saved_item = LibraryItem.model_validate(item_record)
        saved_preferences = [Preference.model_validate(record) for record in preference_records]

    return saved_item, saved_preferences


def get_watched_ids():
    with get_session() as session:
        records = session.execute(
            select(
                LibraryItemRecord.media_type,
                LibraryItemRecord.tmdb_id,
            ).where(LibraryItemRecord.watched.is_(True))
        ).all()

        return {(media_type, tmdb_id) for media_type, tmdb_id in records}
