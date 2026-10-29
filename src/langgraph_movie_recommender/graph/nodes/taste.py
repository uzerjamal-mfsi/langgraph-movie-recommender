import logging

from langgraph_movie_recommender.db.repository import (
    get_preferences,
    get_watched_ids,
)
from langgraph_movie_recommender.graph.nodes.common import get_latest_user_message
from langgraph_movie_recommender.memory.vector_store import search_relevant_taste
from langgraph_movie_recommender.models import watched_id

logger = logging.getLogger(__name__)


def load_taste(state):
    try:
        watched_ids = [
            watched_id(media_type, tmdb_id) for media_type, tmdb_id in sorted(get_watched_ids())
        ]
    except Exception:
        logger.exception("load_taste: failed to load taste profile")
        return {
            "taste_likes": [],
            "taste_dislikes": [],
            "watched_ids": [],
        }

    return {
        "taste_likes": [],
        "taste_dislikes": [],
        "watched_ids": watched_ids,
    }


def retrieve_taste_memories(state):
    message = get_latest_user_message(state)
    likes = []
    dislikes = []

    if message:
        try:
            memories = search_relevant_taste(message, limit=5)
            for memory in memories:
                text = memory.metadata.get("preference_text", memory.page_content)
                kind = memory.metadata.get("preference_kind")

                if kind == "like" and text not in likes:
                    likes.append(text)
                elif kind == "dislike" and text not in dislikes:
                    dislikes.append(text)
        except Exception:
            logger.exception("retrieve_taste_memories: failed to retrieve taste memories")

    if not likes and not dislikes:
        try:
            preferences = get_preferences()
            for pref in preferences:
                if pref.kind == "like" and len(likes) < 5:
                    likes.append(pref.text)
                elif pref.kind == "dislike" and len(dislikes) < 5:
                    dislikes.append(pref.text)
        except Exception:
            logger.exception("retrieve_taste_memories: failed to load fallback preferences")

    return {
        "taste_likes": likes,
        "taste_dislikes": dislikes,
    }
