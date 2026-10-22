import logging
from uuid import uuid4

from langchain_core.messages import AIMessage

from langgraph_movie_recommender.db.repository import save_preferences as persist_preferences
from langgraph_movie_recommender.graph.nodes.common import SAVE_PREFERENCES_ERROR, error_result
from langgraph_movie_recommender.memory.vector_store import index_preference
from langgraph_movie_recommender.models import Preference, utc_now

logger = logging.getLogger(__name__)


def save_preferences(state):
    if state.get("error"):
        return {"messages": [AIMessage(content=state["error"])]}

    preferences = state.get("extracted_preferences", [])

    if not preferences:
        return {
            "messages": [
                AIMessage(content="I didn't find any explicit general preferences to save.")
            ]
        }

    try:
        records = [
            Preference(
                id=str(uuid4()),
                text=extracted["text"],
                kind=extracted["kind"],
                source="explicit",
                updated_at=utc_now(),
            )
            for extracted in preferences
        ]
        saved = persist_preferences(records)

        for preference in saved:
            if not index_preference(preference):
                logger.warning(
                    "save_preferences: could not index preference %s",
                    preference.id,
                )
    except Exception:
        logger.exception("save_preferences: failed to save preferences")
        return error_result(SAVE_PREFERENCES_ERROR)

    texts = [preference.text for preference in saved]

    return {
        "messages": [AIMessage(content="Saved your preferences: " + "; ".join(texts) + ".")],
        "saved_summary": "; ".join(texts),
        "error": None,
    }
