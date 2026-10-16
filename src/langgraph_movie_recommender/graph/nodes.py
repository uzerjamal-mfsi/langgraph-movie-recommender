import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from langgraph_movie_recommender.db.repository import (
    DEFAULT_PROFILE_ID,
    find_library_item,
    save_library_item,
    save_preference,
)
from langgraph_movie_recommender.graph.llm import get_chat_model
from langgraph_movie_recommender.graph.prompts import (
    CLASSIFIER_PROMPT,
    FEEDBACK_EXTRACTION_PROMPT,
    PREFERENCE_EXTRACTION_PROMPT,
)
from langgraph_movie_recommender.graph.recommendations import (
    find_candidates,
    format_recommendations,
)
from langgraph_movie_recommender.graph.schemas import (
    FeedbackExtraction,
    IntentClassification,
    PreferenceExtraction,
)
from langgraph_movie_recommender.graph.state import GraphState
from langgraph_movie_recommender.models import LibraryItem, Preference, utc_now
from langgraph_movie_recommender.tmdb import TitleResolutionError, TMDBError, resolve_media_item

logger = logging.getLogger(__name__)


def _get_latest_user_message(state: GraphState):
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content

    return ""


def route_after_title_resolution(state):
    if state.get("error"):
        return "error"

    if state.get("needs_clarification"):
        return "clarify"

    return "save"


def route_after_clarification(state):
    if state.get("error"):
        return "error"

    if state.get("needs_clarification"):
        return "clarify"

    return "save"


def classify_intent(state: GraphState, config):
    logger.debug("classify_intent: start, message_count=%d", len(state.get("messages", [])))

    try:
        llm = get_chat_model()
        classifier = llm.with_structured_output(IntentClassification)

        result = classifier.invoke(
            [
                SystemMessage(content=CLASSIFIER_PROMPT),
                *state["messages"],
            ]
        )

        logger.debug(
            "classify_intent: classified intent=%s filters=%s",
            result.intent,
            result.filters,
        )

        return {"intent": result.intent, "filters": result.filters, "error": None}
    except Exception:
        logger.exception("classify_intent: failed to classify intent")
        return {
            "messages": [
                AIMessage(content="I couldn't understand your request. Please try again later.")
            ],
            "error": "I couldn't understand your request. Please try again later.",
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

    return {"messages": [AIMessage(content=message)], "error": None}


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


def extract_feedback(state: GraphState):
    message = _get_latest_user_message(state)

    if not message:
        return {
            "error": "I couldn't find your latest message",
            "feedback": None,
        }

    result = (
        get_chat_model()
        .with_structured_output(FeedbackExtraction)
        .invoke([SystemMessage(content=FEEDBACK_EXTRACTION_PROMPT), HumanMessage(content=message)])
    )

    feedback = result.model_dump()

    if feedback["rating"] is not None and not 1 <= feedback["rating"] <= 10:
        feedback["rating"] = None

    if not feedback["title"]:
        return {
            "feedback": feedback,
            "error": "I couldn't identify a title",
        }

    return {"feedback": feedback, "error": None}


def resolve_title(state: GraphState):
    feedback = state.get("feedback")

    if not feedback or state.get("error"):
        return {}

    title = feedback.get("title")
    media_type = feedback.get("media_type")
    year = feedback.get("year")

    if not title:
        return {"error": "I couldn't identify the title to save"}

    if media_type not in {"movie", "tv"}:
        return {
            "error": (
                f"Is '{title}' a movie or a TV show? "
                "I need that detail to identify the correct title."
            )
        }

    try:
        item = resolve_media_item(title, media_type, year)
    except TitleResolutionError:
        return {
            "error": f"I couldn't match '{title}' to a TMDB title. Please provide the exact title and year"
        }

    return {
        "resolved_title": item.model_dump(),
        "error": None,
    }


def save_feedback(state: GraphState):
    if state.get("error"):
        return {"messages": [AIMessage(content=state["error"])]}

    feedback = state.get("feedback")
    resolved = state.get("resolved_title")

    if not feedback or not resolved:
        return {"messages": [AIMessage(content="I couldn't identify the title")]}

    profile_id = DEFAULT_PROFILE_ID
    media_type = resolved["media_type"]
    tmdb_id = resolved["tmdb_id"]

    existing = find_library_item(profile_id, media_type, tmdb_id)

    watched = feedback.get("watched")
    rating = feedback.get("rating")
    review = feedback.get("review")

    item = LibraryItem(
        media_type=media_type,
        tmdb_id=tmdb_id,
        title=resolved["title"],
        year=resolved.get("year"),
        watched=(watched if watched is not None else existing.watched if existing else False),
        rating=(rating if rating is not None else existing.rating if existing else None),
        review=(review if review else existing.review if existing else None),
        updated_at=utc_now(),
    )

    saved_item = save_library_item(profile_id, item)

    saved_preferences = []

    for aspect in feedback["positive_aspects"]:
        saved_preferences.append(
            Preference(
                id=str(uuid4()),
                text=aspect,
                kind="like",
                source="inferred",
                source_title=saved_item.title,
                source_media_type=media_type,
                source_tmdb_id=tmdb_id,
                updated_at=utc_now(),
            )
        )

    for aspect in feedback["negative_aspects"]:
        saved_preferences.append(
            Preference(
                id=str(uuid4()),
                text=aspect,
                kind="dislike",
                source="inferred",
                source_title=saved_item.title,
                source_media_type=media_type,
                source_tmdb_id=tmdb_id,
                updated_at=utc_now(),
            )
        )

    for preference in saved_preferences:
        save_preference(profile_id, preference)

    details = [f"Saved feedback for {saved_item.title}"]

    if watched is True:
        details.append("marked as watched")
    if rating is not None:
        details.append(f"rating {rating}/10")
    if review:
        details.append("saved your review")
    if saved_preferences:
        details.append(f"saved {len(saved_preferences)} title-specific aspect(s)")

    return {
        "messages": [AIMessage(content="I have " + ", ".join(details) + ".")],
        "saved_summary": ", ".join(details),
        "error": None,
    }


def extract_preferences(state: GraphState):
    message = _get_latest_user_message(state)

    if not message:
        return {
            "extracted_preferences": [],
            "error": "I couldn't find your latest message to extract preferences from.",
        }

    result = (
        get_chat_model()
        .with_structured_output(PreferenceExtraction)
        .invoke(
            [
                SystemMessage(content=PREFERENCE_EXTRACTION_PROMPT),
                HumanMessage(content=message),
            ]
        )
    )

    return {
        "extracted_preferences": [preference.model_dump() for preference in result.preferences],
        "error": None,
    }


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

    saved = []

    for extracted in preferences:
        preference = Preference(
            id=str(uuid4()),
            text=extracted["text"],
            kind=extracted["kind"],
            source="explicit",
            updated_at=utc_now(),
        )

        save_preference(DEFAULT_PROFILE_ID, preference)
        saved.append(preference.text)

    return {
        "messages": [AIMessage(content="Saved your preferences: " + "; ".join(saved) + ".")],
        "saved_summary": "; ".join(saved),
        "error": None,
    }


def ask_clarifying_question(state: GraphState):
    question = state["clarification_question"]
    candidates = state.get("clarification_candidates", [])

    if candidates:
        choices = "\n".join(
            f"{i}. {item['title']} ({item.get('year') or 'unknown year'})"
            for i, item in enumerate(candidates, start=1)
        )
        question = f"{question}\n{choices}\nPlease choose a title"

    answer = interrupt(question)

    return {"clasification_answer": str(answer)}


def resolve_classification(state):
    answer = state.get("clarification_answer", "").strip()
    candidates = state.get("clasification_candidates", [])

    if not answer:
        return {"error": "I didn't receive a title choice. Please try again."}

    if answer.isdigit():
        choice = int(answer)

        if 1 <= choice <= len(candidates):
            selected = candidates[choice - 1]
            return {
                "resolved_title": selected,
                "needs_clarification": False,
                "error": None,
            }

    matches = [item for item in candidates if answer.casefold() == item["title"].casefold()]

    if len(matches) == 1:
        return {
            "resolved_title": matches[0],
            "needs_clarification": False,
            "error": None,
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    "I couldn't match that answer to one of the options. "
                    "Please choose an option by number or give the exact title."
                )
            )
        ],
        "needs_clarification": True,
        "error": None,
    }
