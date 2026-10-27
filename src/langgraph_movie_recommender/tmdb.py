import contextlib
import logging
import time

import requests

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.models import MediaItem

TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_WEB_URL = "https://www.themoviedb.org"
REQUEST_TIMEOUT_SECONDS = 10.0
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 1.0
MAX_RETRY_AFTER_SECONDS = 30.0
TMDB_SEARCH_LIMIT = 10
VALID_MEDIA_TYPES = {"movie", "tv"}
GENRE_NAMES = {
    "movie": {
        28: "Action",
        12: "Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        14: "Fantasy",
        36: "History",
        27: "Horror",
        10402: "Music",
        9648: "Mystery",
        10749: "Romance",
        878: "Science Fiction",
        10770: "TV Movie",
        53: "Thriller",
        10752: "War",
        37: "Western",
    },
    "tv": {
        10759: "Action & Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        99: "Documentary",
        18: "Drama",
        10751: "Family",
        10762: "Kids",
        9648: "Mystery",
        10763: "News",
        10764: "Reality",
        10765: "Sci-Fi & Fantasy",
        10766: "Soap",
        10767: "Talk",
        10768: "War & Politics",
        37: "Western",
    },
}

logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class TMDBError(Exception):
    pass


class TMDBClient:
    def __init__(self):
        settings = get_settings()
        token = (settings.tmdb_access_token or settings.tmdb_api_key or "").strip()
        if not token:
            raise TMDBError("TMDB_ACCESS_TOKEN is not configured.")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        self._session = requests.Session()
        self._session.headers.update(headers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self._session.close()

    def search(self, query, media_type="movie", limit=5):
        self._validate_media_type(media_type)
        clean_query = query.strip()

        if not clean_query or limit <= 0:
            return []

        payload = self._request(f"/search/{media_type}", {"query": clean_query})
        return self._normalize_results(payload.get("results"), media_type, limit)

    def discover(
        self,
        media_type="movie",
        genre_ids=None,
        max_runtime=None,
        year=None,
        year_from=None,
        year_to=None,
        limit=10,
    ):
        self._validate_media_type(media_type)

        if limit <= 0:
            return []

        date_field = "primary_release_date" if media_type == "movie" else "first_air_date"
        params = {"sort_by": "vote_average.desc", "vote_count.gte": 100}

        if genre_ids:
            params["with_genres"] = ",".join(str(genre_id) for genre_id in genre_ids)
        if year is not None:
            params[f"{date_field}.gte"] = f"{year}-01-01"
            params[f"{date_field}.lte"] = f"{year}-12-31"
        else:
            if year_from is not None:
                params[f"{date_field}.gte"] = f"{year_from}-01-01"
            if year_to is not None:
                params[f"{date_field}.lte"] = f"{year_to}-12-31"
        if max_runtime is not None and media_type == "movie":
            params["with_runtime.lte"] = max_runtime

        payload = self._request(f"/discover/{media_type}", params)
        return self._normalize_results(payload.get("results"), media_type, limit)

    def _normalize_results(self, results, media_type, limit):
        if not isinstance(results, list):
            return []

        items = []
        for record in results:
            if len(items) >= limit:
                break

            item = self._normalize(record, media_type)
            if item is not None:
                items.append(item)
        return items

    def _request(self, path, params=None):
        request_params = dict(params or {})

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._session.get(
                    f"{TMDB_API_URL}{path}",
                    params=request_params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.Timeout as error:
                if attempt == MAX_ATTEMPTS:
                    raise TMDBError("TMDB request timed out. Please try again.") from error
                self._wait_before_retry(attempt)
                continue
            except requests.exceptions.ConnectionError as error:
                if attempt == MAX_ATTEMPTS:
                    raise TMDBError(
                        "Could not reach TMDB. Check your connection and try again."
                    ) from error
                self._wait_before_retry(attempt)
                continue

            status_code = response.status_code
            if status_code in (401, 403):
                raise TMDBError("TMDB authentication failed. Verify TMDB_ACCESS_TOKEN.")
            if status_code == 404:
                raise TMDBError("TMDB could not find the requested title.")
            if status_code == 429 or status_code >= 500:
                if attempt == MAX_ATTEMPTS:
                    raise TMDBError("TMDB is temporarily unavailable. Please try again later.")
                logger.warning(
                    "TMDB returned status %s; retrying attempt %s/%s",
                    status_code,
                    attempt,
                    MAX_ATTEMPTS,
                )
                self._wait_before_retry(attempt, response)
                continue
            if status_code >= 400:
                raise TMDBError(f"TMDB rejected the request with status {status_code}.")

            try:
                return response.json()
            except ValueError as error:
                raise TMDBError("TMDB returned an invalid response.") from error

        raise TMDBError("TMDB request failed after multiple attempts.")

    def _wait_before_retry(self, attempt, response=None):
        delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        if response is not None and response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                with contextlib.suppress(ValueError):
                    delay = min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_SECONDS)
        time.sleep(delay)

    def _validate_media_type(self, media_type):
        if media_type not in VALID_MEDIA_TYPES:
            raise ValueError("media_type must be 'movie' or 'tv'.")

    def _normalize(self, payload, media_type):
        if not isinstance(payload, dict):
            return None

        tmdb_id = payload.get("id")
        title = payload.get("title") if media_type == "movie" else payload.get("name")
        if not isinstance(tmdb_id, int) or not isinstance(title, str) or not title.strip():
            return None

        release_date = (
            payload.get("release_date") if media_type == "movie" else payload.get("first_air_date")
        )
        return MediaItem(
            tmdb_id=tmdb_id,
            media_type=media_type,
            title=title.strip(),
            year=self._parse_year(release_date),
            overview=self._clean_overview(payload.get("overview")),
            genres=self._extract_genres(payload, media_type),
            vote_average=payload.get("vote_average") or 0.0,
            popularity=payload.get("popularity") or 0.0,
            tmdb_url=f"{TMDB_WEB_URL}/{media_type}/{tmdb_id}",
        )

    def _parse_year(self, release_date):
        if not isinstance(release_date, str):
            return None
        year_part = release_date.split("-", 1)[0]
        if len(year_part) != 4 or not year_part.isdigit():
            return None
        return int(year_part)

    def _clean_overview(self, overview):
        if not isinstance(overview, str):
            return ""
        return overview.strip()

    def _extract_genres(self, payload, media_type):
        genres = payload.get("genres")
        if isinstance(genres, list):
            names = []
            for genre in genres:
                if not isinstance(genre, dict):
                    continue
                name = genre.get("name")
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())
            if names:
                return list(dict.fromkeys(names))

        genre_ids = payload.get("genre_ids")
        if not isinstance(genre_ids, list):
            return []

        mapping = GENRE_NAMES[media_type]
        return list(
            dict.fromkeys(mapping[genre_id] for genre_id in genre_ids if genre_id in mapping)
        )


def genre_ids_from_names(genres, media_type):
    if media_type not in VALID_MEDIA_TYPES:
        return []

    lookup = {name.casefold(): genre_id for genre_id, name in GENRE_NAMES[media_type].items()}
    return [
        lookup[genre.casefold()]
        for genre in genres
        if isinstance(genre, str) and genre.casefold() in lookup
    ]


def find_media_candidates(title, media_type, year, limit=None):
    cleaned_title = title.strip()
    if not cleaned_title:
        raise ValueError("title must not be blank.")
    if media_type not in VALID_MEDIA_TYPES:
        raise ValueError("media_type must be 'movie' or 'tv'.")

    with TMDBClient() as client:
        candidates = client.search(cleaned_title, media_type, limit=TMDB_SEARCH_LIMIT)

    title_key = _title_key(cleaned_title)
    matches = [candidate for candidate in candidates if _title_key(candidate.title) == title_key]
    if year is not None:
        matches = [candidate for candidate in matches if candidate.year == year]

    if limit is not None:
        matches = matches[:limit]

    return matches


def _title_key(title):
    return "".join(character for character in title.casefold() if character.isalnum())
