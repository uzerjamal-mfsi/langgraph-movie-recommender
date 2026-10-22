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


def test_get_library_success(session_factory):
    repository.save_library_item(make_library_item(title="Zulu"))
    repository.save_library_item(make_library_item(tmdb_id=2, title="Alien"))

    titles = [item.title for item in repository.get_library()]

    assert titles == ["Alien", "Zulu"]


def test_get_library_empty(session_factory):
    assert repository.get_library() == []


def test_get_preferences_success(session_factory):
    repository.save_preference(make_preference(id="p2", text="zeta"))
    repository.save_preference(make_preference(id="p1", text="alpha"))

    texts = [preference.text for preference in repository.get_preferences()]

    assert texts == ["alpha", "zeta"]


def test_get_preferences_empty(session_factory):
    assert repository.get_preferences() == []


def test_find_library_item_success(session_factory):
    repository.save_library_item(make_library_item())

    item = repository.find_library_item("movie", 1)

    assert item is not None
    assert item.title == "Heat"


def test_find_library_item_failure(session_factory):
    assert repository.find_library_item("movie", 999) is None


def test_save_library_item_success(session_factory):
    saved = repository.save_library_item(make_library_item())
    updated = repository.save_library_item(make_library_item(rating=10))

    assert saved.title == "Heat"
    assert updated.rating == 10
    assert len(repository.get_library()) == 1


def test_save_library_item_failure(session_factory):
    invalid = SimpleNamespace(media_type="movie", tmdb_id=2)

    with pytest.raises(AttributeError):
        repository.save_library_item(invalid)


def test_save_preference_success(session_factory):
    saved = repository.save_preference(make_preference())
    updated = repository.save_preference(make_preference(text="loves sci-fi"))

    assert saved.text == "likes sci-fi"
    assert updated.text == "loves sci-fi"
    assert len(repository.get_preferences()) == 1


def test_save_preference_failure(session_factory):
    invalid = SimpleNamespace(id="p2")

    with pytest.raises(AttributeError):
        repository.save_preference(invalid)


def test_save_preferences_saves_all_in_one_call(session_factory):
    saved = repository.save_preferences(
        [
            make_preference(id="p1", text="likes sci-fi"),
            make_preference(id="p2", text="dislikes horror", kind="dislike"),
        ]
    )

    assert [preference.id for preference in saved] == ["p1", "p2"]
    assert len(repository.get_preferences()) == 2


def test_save_feedback_and_preferences_is_atomic(session_factory):
    item = make_library_item()
    preferences = [make_preference(id="p1", text="likes sci-fi")]

    saved_item, saved_preferences = repository.save_feedback_and_preferences(item, preferences)

    assert saved_item.title == "Heat"
    assert saved_preferences[0].text == "likes sci-fi"
    assert len(repository.get_library()) == 1
    assert len(repository.get_preferences()) == 1


def test_get_watched_ids_success(session_factory):
    repository.save_library_item(make_library_item(watched=True))
    repository.save_library_item(make_library_item(tmdb_id=2, title="Alien", watched=False))

    assert repository.get_watched_ids() == {("movie", 1)}


def test_get_watched_ids_empty(session_factory):
    assert repository.get_watched_ids() == set()
