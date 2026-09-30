import logging

from langchain_core.messages import AIMessage, SystemMessage

from langgraph_movie_recommender.graph.llm import get_chat_model
from langgraph_movie_recommender.graph.prompts import CLASSIFIER_PROMPT
from langgraph_movie_recommender.graph.recommendations import find_candidates, format_recommendations
from langgraph_movie_recommender.graph.schemas import IntentClassification
from langgraph_movie_recommender.graph.state import GraphState
from langgraph_movie_recommender.tmdb import TMDBError

logger = logging.getLogger(__name__)


def classify_intent(state: GraphState, config):
    logger.debug("classify_intent: start, message_count=%d", len(state.get("messages", [])))

    try:
        llm = get_chat_model()
        classifier = llm.with_structured_output(IntentClassification)

        result = classifier.invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            *state["messages"],
        ])

        logger.debug(
            "classify_intent: classified intent=%s filters=%s",
            result.intent,
            result.filters,
        )

        return {
            "intent": result.intent,
            "filters": result.filters,
            "error": None
        }
    except Exception:
        logger.exception("classify_intent: failed to classify intent")
        return {
            "messages": [AIMessage(content="I couldn't understand your request. Please try again later.")],
            "error": "I couldn't understand your request. Please try again later."
        }

def route_intent(state: GraphState):
    if state.get("error"):
        logger.debug("route_intent: routing to error, error=%s", state.get("error"))
        return "error"

    route = state.get("intent", "general")
    logger.debug("route_intent: routing to %s", route)
    return route

def answer_general(state: GraphState):
    logger.debug("answer_general: start, message_count=%d", len(state.get("messages", [])))

    try:
        llm = get_chat_model()
        response = llm.invoke(state["messages"])
        message = response.content
        logger.debug("answer_general: generated response with %d characters", len(message or ""))
    except Exception:
        logger.exception("answer_general: failed to generate response")
        message = "Something went wrong please try again."

    return {
        "messages": [AIMessage(content=message)],
        "error": None
    }

def recommend(state: GraphState):
    logger.debug("recommend: start, filters=%s", state.get("filters"))

    try:
        items = find_candidates(state.get("filters"))
        logger.debug("recommend: found %d candidate(s)", len(items))
        response = format_recommendations(items)
        logger.debug("recommend: formatted response with %d characters", len(response))
    except TMDBError as e:
        logger.warning("recommend: TMDB error: %s", e)
        response = str(e)
    except Exception:
        logger.exception("recommend: failed to build recommendations")
        response = "Something went wrong please try again."

    return {
        "messages": [AIMessage(content=response)],
        "error": None,
    }
