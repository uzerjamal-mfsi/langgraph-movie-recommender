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
        token = get_settings().tmdb_api_key.strip()
        if not token:
            raise TMDBError("TMDB_API_KEY is not configured.")

        headers = {"Accept": "application/json"}
        self._auth_params = {"api_key": token}

        self._session = requests.Session()
        self._session.headers.update(headers)

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def close(self):
        self._session.close()

    def search(self, query, media_type="movie", limit=5):
        self._validate_media_type(media_type)
        clean_query = query.strip()
        
        if not clean_query or limit <= 0:
            return []

        payload = self._request(
            f"/search/{media_type}",
            {"query": clean_query},
        )
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        items = []
        for record in results:
            if len(items) >= limit:
                break
            if not isinstance(record, dict):
                continue
            item = self._normalize(record, media_type)
            if item is not None:
                items.append(item)
        return items

    def _request(self, path, params=None):
        request_params = dict(params or {})
        request_params.update(self._auth_params)

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
                    raise TMDBError("Could not reach TMDB. Check your connection and try again.") from error
                self._wait_before_retry(attempt)
                continue

            status_code = response.status_code
            if status_code in (401, 403):
                raise TMDBError("TMDB authentication failed. Verify TMDB_API_KEY.")
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
                try:
                    delay = min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_SECONDS)
                except ValueError:
                    pass
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

        release_date = payload.get("release_date") if media_type == "movie" else payload.get("first_air_date")
        return MediaItem(
            tmdb_id=tmdb_id,
            media_type=media_type,
            title=title.strip(),
            year=self._parse_year(release_date),
            overview=self._clean_overview(payload.get("overview")),
            genres=self._extract_genres(payload, media_type),
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
        return list(dict.fromkeys(mapping[genre_id] for genre_id in genre_ids if genre_id in mapping))
