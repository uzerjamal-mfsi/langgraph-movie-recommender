from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.models import Base

settings = get_settings()
settings.library_db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.library_db_path.as_posix()}",
)


@event.listens_for(engine, "connect")
def enable_foreign_keys(connection, _):
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


def get_session():
    return SessionFactory()


def init_db():
    Base.metadata.create_all(engine)
