from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.migrations import migrate

SQLITE_TIMEOUT_SECONDS = 30


def _configure_connection(connection, _):
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(f"PRAGMA busy_timeout={SQLITE_TIMEOUT_SECONDS * 1000}")


def _create_engine(db_path):
    path = Path(db_path)

    if str(path) == ":memory:":
        url = "sqlite://"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path.as_posix()}"

    engine = create_engine(url, connect_args={"timeout": SQLITE_TIMEOUT_SECONDS})
    event.listen(engine, "connect", _configure_connection)
    return engine


@lru_cache
def get_engine(db_path=None):
    target_path = db_path or get_settings().library_db_path
    return _create_engine(target_path)


@lru_cache
def get_session_factory():
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session():
    return get_session_factory()()


def init_db(db_engine=None):
    target_engine = db_engine or get_engine()

    with target_engine.begin() as connection:
        migrate(connection)
