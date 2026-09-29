from langgraph_movie_recommender.tmdb import resolve_media_item


def save_library_item(
    repository,
    title,
    media_type,
    *,
    watched=False,
    rating=None,
    review=None,
    year=None,
):
    media_item = resolve_media_item(title, media_type, year=year)
    return repository.upsert(
        media_item,
        watched=watched,
        rating=rating,
        review=review,
    )
