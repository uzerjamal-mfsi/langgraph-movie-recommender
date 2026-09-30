from langgraph_movie_recommender.tmdb import TMDBClient, genre_ids_from_names

RESULT_LIMIT = 10


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
