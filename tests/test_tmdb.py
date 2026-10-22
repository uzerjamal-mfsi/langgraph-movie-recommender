from types import SimpleNamespace

import pytest
import requests

from langgraph_movie_recommender import tmdb


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self.payload = payload or {"results": []}
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        tmdb,
        "get_settings",
        lambda: SimpleNamespace(tmdb_access_token="token", tmdb_api_key=None),
    )
    monkeypatch.setattr(tmdb.time, "sleep", lambda seconds: None)
    return tmdb.TMDBClient()


def attach_session(client, *responses):
    session = FakeSession(responses)
    client._session = session
    return session


def test_search_normalizes_movie_results(client):
    session = attach_session(
        client,
        FakeResponse(
            payload={
                "results": [
                    {
                        "id": 949,
                        "title": "Heat",
                        "release_date": "1995-12-15",
                        "overview": " A crew of thieves. ",
                        "genre_ids": [80, 18],
                        "vote_average": 8.3,
                        "popularity": 12.5,
                    }
                ]
            }
        ),
    )

    results = client.search(" Heat ", "movie", limit=5)

    assert session.calls[0][0] == f"{tmdb.TMDB_API_URL}/search/movie"
    assert session.calls[0][1] == {"query": "Heat"}
    assert results[0].title == "Heat"
    assert results[0].year == 1995
    assert results[0].genres == ["Crime", "Drama"]
    assert results[0].overview == "A crew of thieves."


def test_discover_builds_tv_date_filters(client):
    session = attach_session(
        client,
        FakeResponse(
            payload={
                "results": [
                    {
                        "id": 1396,
                        "name": "Breaking Bad",
                        "first_air_date": "2008-01-20",
                        "genre_ids": [10765, 18],
                        "vote_average": 9.5,
                        "popularity": 50,
                    }
                ]
            }
        ),
    )

    results = client.discover(
        "tv",
        genre_ids=[10765],
        year_from=2000,
        year_to=2020,
        limit=1,
    )

    params = session.calls[0][1]
    assert params["with_genres"] == "10765"
    assert params["first_air_date.gte"] == "2000-01-01"
    assert params["first_air_date.lte"] == "2020-12-31"
    assert results[0].title == "Breaking Bad"
    assert results[0].genres == ["Sci-Fi & Fantasy", "Drama"]


def test_search_returns_empty_for_blank_query(client):
    session = attach_session(client)

    assert client.search("   ") == []
    assert session.calls == []


def test_search_rejects_invalid_media_type(client):
    with pytest.raises(ValueError):
        client.search("Heat", "book")


def test_client_requires_token(monkeypatch):
    monkeypatch.setattr(
        tmdb,
        "get_settings",
        lambda: SimpleNamespace(tmdb_access_token=None, tmdb_api_key=None),
    )

    with pytest.raises(tmdb.TMDBError):
        tmdb.TMDBClient()


def test_authentication_failure_raises(client):
    attach_session(client, FakeResponse(status_code=401))

    with pytest.raises(tmdb.TMDBError, match="authentication"):
        client.search("Heat")


def test_server_error_is_retried(client):
    session = attach_session(
        client,
        FakeResponse(status_code=500),
        FakeResponse(payload={"results": []}),
    )

    assert client.search("Heat") == []
    assert len(session.calls) == 2


def test_timeouts_raise_after_max_attempts(client):
    attach_session(
        client,
        *[requests.exceptions.Timeout() for _ in range(tmdb.MAX_ATTEMPTS)],
    )

    with pytest.raises(tmdb.TMDBError, match="timed out"):
        client.search("Heat")
