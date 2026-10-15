from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from langgraph_movie_recommender.db import repository
from langgraph_movie_recommender.db.models import Base
from langgraph_movie_recommender.models import LibraryItem, Preference


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(repository, "get_session", factory)
    yield factory
    engine.dispose()


def make_library_item(**overrides):
    data = {
        "media_type": "movie",
        "tmdb_id": 1,
        "title": "Heat",
        "year": 1995,
        "watched": True,
        "rating": 9,
        "review": "Great",
    }
    data.update(overrides)
    return LibraryItem(**data)


def make_preference(**overrides):
    data = {"id": "p1", "text": "likes sci-fi", "kind": "like"}
    data.update(overrides)
    return Preference(**data)


def library_record(**overrides):
    data = {
        "media_type": "movie",
        "tmdb_id": 1,
        "title": "Heat",
        "year": 1995,
        "watched": True,
        "rating": 9,
        "review": "Great",
        "updated_at": datetime.now(UTC),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def preference_record(**overrides):
    data = {
        "id": "p1",
        "text": "likes sci-fi",
        "kind": "like",
        "source": "explicit",
        "source_title": None,
        "source_media_type": None,
        "source_tmdb_id": None,
        "updated_at": datetime.now(UTC),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_library_item_success():
    item = repository._library_item(library_record())

    assert isinstance(item, LibraryItem)
    assert item.title == "Heat"


def test_library_item_failure():
    record = library_record()
    del record.title

    with pytest.raises(AttributeError):
        repository._library_item(record)


def test_preference_success():
    preference = repository._preference(preference_record())

    assert isinstance(preference, Preference)
    assert preference.text == "likes sci-fi"


def test_preference_failure():
    record = preference_record()
    del record.id

    with pytest.raises(AttributeError):
        repository._preference(record)


def test_get_library_success(session_factory):
    repository.save_library_item("default", make_library_item(title="Zulu"))
    repository.save_library_item("default", make_library_item(tmdb_id=2, title="Alien"))

    titles = [item.title for item in repository.get_library()]

    assert titles == ["Alien", "Zulu"]


def test_get_library_unknown_profile_failure(session_factory):
    assert repository.get_library("missing") == []


def test_get_preferences_success(session_factory):
    repository.save_preference("default", make_preference(id="p2", text="zeta"))
    repository.save_preference("default", make_preference(id="p1", text="alpha"))

    texts = [preference.text for preference in repository.get_preferences()]

    assert texts == ["alpha", "zeta"]


def test_get_preferences_unknown_profile_failure(session_factory):
    assert repository.get_preferences("missing") == []


def test_find_library_item_success(session_factory):
    repository.save_library_item("default", make_library_item())

    item = repository.find_library_item("default", "movie", 1)

    assert item is not None
    assert item.title == "Heat"


def test_find_library_item_failure(session_factory):
    assert repository.find_library_item("default", "movie", 999) is None


def test_save_library_item_success(session_factory):
    saved = repository.save_library_item("default", make_library_item())
    updated = repository.save_library_item("default", make_library_item(rating=10))

    assert saved.title == "Heat"
    assert updated.rating == 10
    assert len(repository.get_library()) == 1


def test_save_library_item_failure(session_factory):
    invalid = SimpleNamespace(media_type="movie", tmdb_id=2)

    with pytest.raises(AttributeError):
        repository.save_library_item("default", invalid)


def test_save_preference_success(session_factory):
    saved = repository.save_preference("default", make_preference())
    updated = repository.save_preference("default", make_preference(text="loves sci-fi"))

    assert saved.text == "likes sci-fi"
    assert updated.text == "loves sci-fi"
    assert len(repository.get_preferences()) == 1


def test_save_preference_failure(session_factory):
    invalid = SimpleNamespace(id="p2")

    with pytest.raises(AttributeError):
        repository.save_preference("default", invalid)


def test_get_watched_ids_success(session_factory):
    repository.save_library_item("default", make_library_item(watched=True))
    repository.save_library_item(
        "default", make_library_item(tmdb_id=2, title="Alien", watched=False)
    )

    assert repository.get_watched_ids() == {("movie", 1)}


def test_get_watched_ids_unknown_profile_failure(session_factory):
    assert repository.get_watched_ids("missing") == set()
