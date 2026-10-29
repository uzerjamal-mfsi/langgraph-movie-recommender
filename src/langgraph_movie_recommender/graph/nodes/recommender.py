import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_movie_recommender.graph.llm import get_chat_model
from langgraph_movie_recommender.graph.nodes.common import GENERAL_ERROR, error_result
from langgraph_movie_recommender.graph.prompts import RECOMMENDER_PROMPT
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

        top_items = items[:5]
        if not top_items:
            response = "I couldn't find recommendations matching your request and watch history."
        else:
            candidate_context = "\n\n".join(
                f"Title: {item.title} ({item.year or 'N/A'})\n"
                f"Type: {'Movie' if item.media_type == 'movie' else 'TV show'}\n"
                f"Genres: {', '.join(item.genres) if item.genres else 'None'}\n"
                f"Rating: {item.vote_average}/10\n"
                f"Overview: {item.overview or 'No synopsis available'}\n"
                f"TMDB URL: {item.tmdb_url}"
                for item in top_items
            )
            likes = state.get("taste_likes", [])
            dislikes = state.get("taste_dislikes", [])
            taste_context = (
                f"Retrieved User Likes: {', '.join(likes) if likes else 'None'}\n"
                f"Retrieved User Dislikes: {', '.join(dislikes) if dislikes else 'None'}"
            )
            latest_msg = ""
            for m in reversed(state.get("messages", [])):
                if getattr(m, "content", None):
                    latest_msg = m.content
                    break

            prompt_content = (
                f"User Request: {latest_msg}\n\n"
                f"User Taste Profile (from vector memory):\n{taste_context}\n\n"
                f"Candidate Titles (from TMDB retrieval):\n{candidate_context}"
            )

            try:
                llm_response = get_chat_model().invoke(
                    [
                        SystemMessage(content=RECOMMENDER_PROMPT),
                        HumanMessage(content=prompt_content),
                    ]
                )
                response = llm_response.content
            except Exception:
                logger.warning(
                    "recommend: LLM synthesis failed, falling back to template",
                    exc_info=True,
                )
                response = format_recommendations(top_items)

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
