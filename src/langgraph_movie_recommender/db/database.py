from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from langgraph_movie_recommender.db.models import Base

DEFAULT_DATABASE_PATH = Path("data/library.db")
SQLITE_BUSY_TIMEOUT_MILLISECONDS = 5000


def _configure_sqlite_connection(dbapi_connection):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA journal_mode = WAL")
    cursor.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MILLISECONDS}")
    cursor.close()


class Database:
    def __init__(self, db_path=DEFAULT_DATABASE_PATH):
        path = Path(db_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        url = URL.create("sqlite+pysqlite", database=path.as_posix())
        self.engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        event.listen(self.engine, "connect", _configure_sqlite_connection)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def initialize(self):
        Base.metadata.create_all(self.engine)
