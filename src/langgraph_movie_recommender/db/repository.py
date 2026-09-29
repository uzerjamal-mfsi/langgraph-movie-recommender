from sqlalchemy import select

from langgraph_movie_recommender.db.models import LibraryItemRecord, PreferenceRecord
from langgraph_movie_recommender.models import LibraryItem, Preference, utc_now


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
        updated_at=record.updated_at,
    )


class LibraryRepository:
    def __init__(self, database):
        self._database = database

    def get(self, media_type, tmdb_id):
        with self._database.session_factory() as session:
            record = session.get(LibraryItemRecord, (media_type, tmdb_id))
            return _library_item(record) if record else None

    def upsert(self, media_item, watched=False, rating=None, review=None):
        with self._database.session_factory.begin() as session:
            record = session.get(
                LibraryItemRecord,
                (media_item.media_type, media_item.tmdb_id),
            )
            if record is None:
                record = LibraryItemRecord(
                    media_type=media_item.media_type,
                    tmdb_id=media_item.tmdb_id,
                )
                session.add(record)

            record.title = media_item.title
            record.year = media_item.year
            record.watched = watched
            record.rating = rating
            record.review = review
            record.updated_at = utc_now()
            session.flush()
            return _library_item(record)

    def list(self):
        statement = select(LibraryItemRecord).order_by(
            LibraryItemRecord.updated_at.desc(),
            LibraryItemRecord.title.collate("NOCASE"),
        )
        with self._database.session_factory() as session:
            return [_library_item(record) for record in session.scalars(statement)]

    def delete(self, media_type, tmdb_id):
        with self._database.session_factory.begin() as session:
            record = session.get(LibraryItemRecord, (media_type, tmdb_id))
            if record is None:
                return False
            session.delete(record)
            return True


class PreferenceRepository:
    def __init__(self, database):
        self._database = database

    def get(self, preference_id):
        with self._database.session_factory() as session:
            record = session.get(PreferenceRecord, preference_id)
            return _preference(record) if record else None

    def upsert(self, preference_id, text, kind):
        with self._database.session_factory.begin() as session:
            record = session.get(PreferenceRecord, preference_id)
            if record is None:
                record = PreferenceRecord(id=preference_id)
                session.add(record)

            record.text = text
            record.kind = kind
            record.updated_at = utc_now()
            session.flush()
            return _preference(record)

    def list(self):
        statement = select(PreferenceRecord).order_by(
            PreferenceRecord.updated_at.desc(),
            PreferenceRecord.id,
        )
        with self._database.session_factory() as session:
            return [_preference(record) for record in session.scalars(statement)]

    def delete(self, preference_id):
        with self._database.session_factory.begin() as session:
            record = session.get(PreferenceRecord, preference_id)
            if record is None:
                return False
            session.delete(record)
            return True
