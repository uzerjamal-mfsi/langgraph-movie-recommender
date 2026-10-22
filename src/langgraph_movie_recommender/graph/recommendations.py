import re

from langgraph_movie_recommender.models import watched_id
from langgraph_movie_recommender.tmdb import TMDBClient, genre_ids_from_names

RESULT_LIMIT = 10


def _matches_constraints(item, filters):
    media_type = filters.get("media_type")
    if media_type in {"movie", "tv"} and item.media_type != media_type:
        return False

    year = filters.get("year")
    if year is not None and item.year != year:
        return False

    year_from = filters.get("year_from")
    if year_from is not None and (item.year is None or item.year < year_from):
        return False

    year_to = filters.get("year_to")
    if year_to is not None and (item.year is None or item.year > year_to):
        return False

    genres = {genre.casefold() for genre in item.genres}
    requested_genres = {genre.casefold() for genre in filters.get("genres", [])}
    return not requested_genres or requested_genres.issubset(genres)


def find_candidates(filters):
    filters = filters or {}
    media_type = filters.get("media_type", "movie")

    if media_type not in ("movie", "tv", "both"):
        media_type = "movie"

    media_types = ("movie", "tv") if media_type == "both" else (media_type,)

    genres = filters.get("genres") or []

    if isinstance(genres, str):
        genres = [genres]

    candidates = []

    with TMDBClient() as client:
        for kind in media_types:
            query = filters.get("query")

            if query:
                items = client.search(query, media_type=kind, limit=RESULT_LIMIT)
            else:
                items = client.discover(
                    media_type=kind,
                    genre_ids=genre_ids_from_names(genres, kind),
                    max_runtime=filters.get("max_runtime") if kind == "movie" else None,
                    year=filters.get("year"),
                    year_from=filters.get("year_from"),
                    year_to=filters.get("year_to"),
                    limit=RESULT_LIMIT,
                )

            candidates.extend(items)

    candidates.sort(
        key=lambda item: (item.vote_average, item.popularity),
        reverse=True,
    )

    unique = []
    seen = set()

    for item in candidates:
        key = (item.media_type, item.tmdb_id)

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:RESULT_LIMIT]


def format_recommendations(items):
    if not items:
        return "I couldn't find matching titles. Try different genres or filters."

    lines = ["Here are some titles you might enjoy:"]

    for index, item in enumerate(items, start=1):
        year = f" ({item.year})" if item.year else ""
        media_type = "Movie" if item.media_type == "movie" else "TV show"

        lines.append(f"\n{index}. **{item.title}{year}** — {media_type}")
        lines.append(f"   TMDB: {item.tmdb_url}")

        if item.overview:
            lines.append(f"   {item.overview}")

    return "\n".join(lines)


def filter_recommendations(items, filters, watched_ids):
    allow_rewatch = filters.get("allow_rewatch", False)
    watched = set(watched_ids or [])

    return [
        item
        for item in items
        if _matches_constraints(item, filters)
        and (allow_rewatch or watched_id(item.media_type, item.tmdb_id) not in watched)
    ]


def _words(text):
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) > 2}


def _taste_score(item, likes, dislikes):
    description = " ".join(
        [
            item.title,
            item.overview or "",
            " ".join(item.genres),
        ]
    )
    words = _words(description)

    score = sum(len(words & _words(preference)) for preference in likes)
    score -= sum(len(words & _words(preference)) for preference in dislikes)

    return score


def personalize_recommendations(items, likes, dislikes):
    return sorted(
        items,
        key=lambda item: (
            _taste_score(item, likes, dislikes),
            item.vote_average or 0,
            item.popularity or 0,
        ),
        reverse=True,
    )
