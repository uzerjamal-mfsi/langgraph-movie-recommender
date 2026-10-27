import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langgraph_movie_recommender.graph.llm import get_chat_model
from langgraph_movie_recommender.graph.nodes.common import (
    FEEDBACK_EXTRACTION_ERROR,
    GENERAL_ERROR,
    INTENT_ERROR,
    PREFERENCE_EXTRACTION_ERROR,
    error_result,
    failed,
    get_latest_user_message,
)
from langgraph_movie_recommender.graph.prompts import (
    CLASSIFIER_PROMPT,
    FEEDBACK_EXTRACTION_PROMPT,
    GENERAL_ASSISTANT_PROMPT,
    PREFERENCE_EXTRACTION_PROMPT,
)
from langgraph_movie_recommender.graph.schemas import (
    FeedbackExtraction,
    IntentClassification,
    PreferenceExtraction,
)
from langgraph_movie_recommender.graph.structured import invoke_structured

logger = logging.getLogger(__name__)


def classify_intent(state, config):
    logger.debug("classify_intent: start, message_count=%d", len(state.get("messages", [])))

    try:
        result = invoke_structured(
            get_chat_model(),
            IntentClassification,
            [
                SystemMessage(content=CLASSIFIER_PROMPT),
                *state["messages"],
            ],
            operation="classify_intent",
        )
    except Exception:
        logger.exception("classify_intent: failed to classify intent")
        return error_result(INTENT_ERROR)

    if result is None:
        return error_result(INTENT_ERROR)

    filters = result.filters.model_dump(exclude_none=True) if result.filters else None
    logger.debug(
        "classify_intent: classified intent=%s filters=%s",
        result.intent,
        filters,
    )

    return {"intent": result.intent, "filters": filters, "error": None}


def answer_general(state):
    logger.debug("answer_general: start, message_count=%d", len(state.get("messages", [])))

    try:
        response = get_chat_model().invoke(
            [
                SystemMessage(content=GENERAL_ASSISTANT_PROMPT),
                *state["messages"],
            ]
        )
        message = response.content
    except Exception:
        logger.exception("answer_general: failed to generate response")
        return error_result(GENERAL_ERROR)

    logger.debug("answer_general: generated response with %d characters", len(message or ""))
    return {"messages": [AIMessage(content=message)], "error": None}


def extract_feedback(state):
    message = get_latest_user_message(state)

    if not message:
        return failed("I couldn't find your latest message", feedback=None)

    try:
        result = invoke_structured(
            get_chat_model(),
            FeedbackExtraction,
            [
                SystemMessage(content=FEEDBACK_EXTRACTION_PROMPT),
                HumanMessage(content=message),
            ],
            operation="extract_feedback",
        )
    except Exception:
        logger.exception("extract_feedback: failed to extract feedback")
        return failed(FEEDBACK_EXTRACTION_ERROR, feedback=None)

    if result is None:
        return failed(FEEDBACK_EXTRACTION_ERROR, feedback=None)

    feedback = result.model_dump()

    if not feedback["title"]:
        return failed("I couldn't identify a title", feedback=feedback)

    return {"feedback": feedback, "error": None}


def extract_preferences(state):
    message = get_latest_user_message(state)

    if not message:
        return failed(
            "I couldn't find your latest message to extract preferences from.",
            extracted_preferences=[],
        )

    try:
        result = invoke_structured(
            get_chat_model(),
            PreferenceExtraction,
            [
                SystemMessage(content=PREFERENCE_EXTRACTION_PROMPT),
                HumanMessage(content=message),
            ],
            operation="extract_preferences",
        )
    except Exception:
        logger.exception("extract_preferences: failed to extract preferences")
        return failed(PREFERENCE_EXTRACTION_ERROR, extracted_preferences=[])

    if result is None:
        return failed(PREFERENCE_EXTRACTION_ERROR, extracted_preferences=[])

    return {
        "extracted_preferences": [preference.model_dump() for preference in result.preferences],
        "error": None,
    }
