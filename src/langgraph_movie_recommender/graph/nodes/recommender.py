import logging

from langchain_core.messages import AIMessage

from langgraph_movie_recommender.graph.nodes.common import GENERAL_ERROR, error_result
from langgraph_movie_recommender.graph.recommendations import (
    filter_recommendations,
    find_candidates,
    format_recommendations,
    personalize_recommendations,
)
from langgraph_movie_recommender.graph.schemas import normalize_filters
from langgraph_movie_recommender.tmdb import TMDBError

logger = logging.getLogger(__name__)


def recommend(state):
    logger.debug("recommend: start, filters=%s", state.get("filters"))

    try:
        filters = normalize_filters(state.get("filters"))
        items = find_candidates(filters)
        logger.debug("recommend: found %d candidate(s)", len(items))

        items = filter_recommendations(
            items,
            filters,
            state.get("watched_ids", []),
        )
        logger.debug("recommend: %d candidate(s) after filtering", len(items))

        items = personalize_recommendations(
            items,
            state.get("taste_likes", []),
            state.get("taste_dislikes", []),
        )

        if not items:
            response = "I couldn't find recommendations matching your request and watch history."
        else:
            response = format_recommendations(items[:5])

        logger.debug("recommend: formatted response with %d characters", len(response))
    except TMDBError as error:
        logger.warning("recommend: TMDB error: %s", error)
        return error_result(str(error))
    except Exception:
        logger.exception("recommend: failed to build recommendations")
        return error_result(GENERAL_ERROR)

    return {
        "messages": [AIMessage(content=response)],
        "error": None,
    }
