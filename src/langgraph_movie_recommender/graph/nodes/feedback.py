import logging
from uuid import uuid4

from langchain_core.messages import AIMessage

from langgraph_movie_recommender.db.repository import (
    find_library_item,
    save_feedback_and_preferences,
)
from langgraph_movie_recommender.graph.nodes.common import SAVE_FEEDBACK_ERROR, error_result
from langgraph_movie_recommender.memory.vector_store import index_preference
from langgraph_movie_recommender.models import LibraryItem, Preference, utc_now

logger = logging.getLogger(__name__)


def _build_preferences(feedback, saved_item):
    preferences = []

    for kind, aspects in (
        ("like", feedback["positive_aspects"]),
        ("dislike", feedback["negative_aspects"]),
    ):
        for aspect in aspects:
            preferences.append(
                Preference(
                    id=str(uuid4()),
                    text=aspect,
                    kind=kind,
                    source="inferred",
                    source_title=saved_item.title,
                    source_media_type=saved_item.media_type,
                    source_tmdb_id=saved_item.tmdb_id,
                    updated_at=utc_now(),
                )
            )

    return preferences


def save_feedback(state):
    if state.get("error"):
        return {"messages": [AIMessage(content=state["error"])]}

    feedback = state.get("feedback")
    resolved = state.get("resolved_title")

    if not feedback or not resolved:
        return {"messages": [AIMessage(content="I couldn't identify the title")]}

    try:
        media_type = resolved["media_type"]
        tmdb_id = resolved["tmdb_id"]
        existing = find_library_item(media_type, tmdb_id)

        watched = feedback.get("watched")
        rating = feedback.get("rating")
        review = feedback.get("review")

        item = LibraryItem(
            media_type=media_type,
            tmdb_id=tmdb_id,
            title=resolved["title"],
            year=resolved.get("year"),
            watched=watched if watched is not None else existing.watched if existing else False,
            rating=rating if rating is not None else existing.rating if existing else None,
            review=review if review else existing.review if existing else None,
            updated_at=utc_now(),
        )
        preferences = _build_preferences(feedback, item)

        saved_item, saved_preferences = save_feedback_and_preferences(item, preferences)

        for preference in saved_preferences:
            if not index_preference(preference):
                logger.warning(
                    "save_feedback: could not index preference %s",
                    preference.id,
                )

        details = [f"Saved feedback for {saved_item.title}"]

        if watched is True:
            details.append("marked as watched")
        if rating is not None:
            details.append(f"rating {rating}/10")
        if review:
            details.append("saved your review")
        if saved_preferences:
            details.append(f"saved {len(saved_preferences)} title-specific aspect(s)")
    except Exception:
        logger.exception("save_feedback: failed to save feedback")
        return error_result(SAVE_FEEDBACK_ERROR)

    return {
        "messages": [AIMessage(content="I have " + ", ".join(details) + ".")],
        "saved_summary": ", ".join(details),
        "error": None,
    }
