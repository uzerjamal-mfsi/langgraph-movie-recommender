from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.models import Base


def _enable_foreign_keys(connection, _):
    connection.execute("PRAGMA foreign_keys=ON")


@lru_cache
def get_engine():
    settings = get_settings()
    settings.library_db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{settings.library_db_path.as_posix()}")
    event.listen(engine, "connect", _enable_foreign_keys)
    return engine


@lru_cache
def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session():
    return get_session_factory()()


def init_db(db_engine=None):
    target_engine = db_engine or get_engine()
    Base.metadata.create_all(target_engine)
