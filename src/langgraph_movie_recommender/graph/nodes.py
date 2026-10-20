import logging
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from langgraph_movie_recommender.db.repository import (
    DEFAULT_PROFILE_ID,
    find_library_item,
    get_library,
    get_preferences,
    get_watched_ids,
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
    filter_recommendations,
    find_candidates,
    format_recommendations,
    personalize_recommendations,
)
from langgraph_movie_recommender.graph.schemas import (
    FeedbackExtraction,
    IntentClassification,
    PreferenceExtraction,
)
from langgraph_movie_recommender.graph.state import GraphState
from langgraph_movie_recommender.memory.vector_store import index_preference, search_relevant_taste
from langgraph_movie_recommender.models import LibraryItem, Preference, utc_now
from langgraph_movie_recommender.tmdb import TMDBError, find_media_candidates

logger = logging.getLogger(__name__)


def _get_latest_user_message(state: GraphState):
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content

    return ""


def _index_saved_preference(profile_id, preference):
    saved = next(
        (
            item
            for item in get_preferences(profile_id)
            if item.text == preference.text and item.kind == preference.kind
        ),
        None,
    )

    if saved:
        index_preference(saved, profile_id)


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

    filters = state.get("filters") or {}

    try:
        items = find_candidates(filters)
        logger.debug("recommend: found %d candidate(s)", len(items))

        items = filter_recommendations(
            items,
            filters,
            state.get("watched_ids", set()),
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

    try:
        result = (
            get_chat_model()
            .with_structured_output(FeedbackExtraction)
            .invoke(
                [
                    SystemMessage(content=FEEDBACK_EXTRACTION_PROMPT),
                    HumanMessage(content=message),
                ]
            )
        )

        feedback = result.model_dump()
    except Exception:
        logger.exception("extract_feedback: failed to extract feedback")
        return {
            "feedback": None,
            "error": "I couldn't process your feedback. Please try again later.",
        }

    if feedback["rating"] is not None and not 1 <= feedback["rating"] <= 10:
        feedback["rating"] = None

    if not feedback["title"]:
        return {
            "feedback": feedback,
            "error": "I couldn't identify a title",
        }

    return {"feedback": feedback, "error": None}


def _resolve_title_candidates(title, media_type, year):
    try:
        matches = find_media_candidates(title, media_type, year)
    except (TMDBError, ValueError) as error:
        return {"error": str(error), "needs_clarification": False}
    except Exception:
        logger.exception("resolve_title: failed to resolve title")
        return {
            "error": "I couldn't look up that title right now. Please try again later.",
            "needs_clarification": False,
        }

    if not matches:
        year_label = f" from {year}" if year is not None else ""
        return {
            "error": (
                f"I couldn't match '{title}' to a TMDB {media_type}{year_label}. "
                "Please provide the exact title and year."
            ),
            "needs_clarification": False,
        }

    if len(matches) > 1:
        return {
            "clarification_question": (
                f"I found multiple {media_type} titles matching '{title}'. Which one did you mean?"
            ),
            "clarification_candidates": [item.model_dump() for item in matches],
            "needs_clarification": True,
            "error": None,
        }

    return {
        "resolved_title": matches[0].model_dump(),
        "needs_clarification": False,
        "error": None,
    }


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
            "clarification_question": (
                f"Is '{title}' a movie or a TV show? Please reply with 'movie' or 'tv'."
            ),
            "clarification_candidates": [],
            "needs_clarification": True,
            "error": None,
        }

    return _resolve_title_candidates(title, media_type, year)


def save_feedback(state: GraphState):
    if state.get("error"):
        return {"messages": [AIMessage(content=state["error"])]}

    feedback = state.get("feedback")
    resolved = state.get("resolved_title")

    if not feedback or not resolved:
        return {"messages": [AIMessage(content="I couldn't identify the title")]}

    try:
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
            _index_saved_preference(profile_id, preference)

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
    except Exception:
        logger.exception("save_feedback: failed to save feedback")
        message = "I couldn't save your feedback. Please try again later."
        return {
            "messages": [AIMessage(content=message)],
            "error": message,
        }


def extract_preferences(state: GraphState):
    message = _get_latest_user_message(state)

    if not message:
        return {
            "extracted_preferences": [],
            "error": "I couldn't find your latest message to extract preferences from.",
        }

    try:
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
    except Exception:
        logger.exception("extract_preferences: failed to extract preferences")
        return {
            "extracted_preferences": [],
            "error": "I couldn't process your preferences. Please try again later.",
        }

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

    try:
        for extracted in preferences:
            preference = Preference(
                id=str(uuid4()),
                text=extracted["text"],
                kind=extracted["kind"],
                source="explicit",
                updated_at=utc_now(),
            )

            save_preference(DEFAULT_PROFILE_ID, preference)
            _index_saved_preference(DEFAULT_PROFILE_ID, preference)
            saved.append(preference.text)
    except Exception:
        logger.exception("save_preferences: failed to save preferences")
        message = "I couldn't save your preferences. Please try again later."
        return {
            "messages": [AIMessage(content=message)],
            "error": message,
        }

    return {
        "messages": [AIMessage(content="Saved your preferences: " + "; ".join(saved) + ".")],
        "saved_summary": "; ".join(saved),
        "error": None,
    }


def ask_clarifying_question(state: GraphState):
    question = state.get("clarification_question") or "Could you clarify what you meant?"
    candidates = state.get("clarification_candidates") or []

    if candidates:
        choices = "\n".join(
            f"{i}. {item['title']} ({item.get('year') or 'unknown year'})"
            for i, item in enumerate(candidates, start=1)
        )
        question = f"{question}\n{choices}\nPlease choose a title"

    answer = interrupt(question)

    return {
        "clarification_answer": str(answer),
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=str(answer)),
        ],
    }


def resolve_classification(state):
    raw_answer = state.get("clarification_answer")
    answer = str(raw_answer).strip() if raw_answer is not None else ""
    candidates = state.get("clarification_candidates") or []

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

    feedback = state.get("feedback") or {}
    media_type_answer = answer.casefold()

    if media_type_answer in {"movie", "tv"}:
        title = feedback.get("title")
        if title:
            return _resolve_title_candidates(title, media_type_answer, feedback.get("year"))

    matches = [item for item in candidates if answer.casefold() == item["title"].casefold()]

    if len(matches) == 1:
        return {
            "resolved_title": matches[0],
            "needs_clarification": False,
            "error": None,
        }

    return {
        "clarification_question": (
            "I couldn't match that answer to one of the options. "
            "Please choose an option by number or give the exact title."
        ),
        "clarification_candidates": candidates,
        "needs_clarification": True,
        "error": None,
    }


def load_taste(state: GraphState):
    try:
        library = get_library(DEFAULT_PROFILE_ID)
        preferences = get_preferences(DEFAULT_PROFILE_ID)
        watched_ids = get_watched_ids(DEFAULT_PROFILE_ID)

        ratings = [
            f"{item.title}: {item.rating}/10" + (f" — {item.review}" if item.review else "")
            for item in library
            if item.watched and (item.rating is not None or item.review)
        ]

        likes = [
            preference.text
            for preference in preferences
            if str(getattr(preference.kind, "value", preference.kind)) == "like"
        ]

        dislikes = [
            preference.text
            for preference in preferences
            if str(getattr(preference.kind, "value", preference.kind)) == "dislike"
        ]

        context = []

        if likes:
            context.append("Likes: " + "; ".join(likes[:10]))

        if dislikes:
            context.append("Dislikes: " + "; ".join(dislikes[:10]))

        if ratings:
            context.append("Watched titles and feedback: " + "; ".join(ratings[:10]))

        return {
            "taste_context": "\n".join(context),
            "taste_likes": likes,
            "taste_dislikes": dislikes,
            "watched_ids": watched_ids,
        }
    except Exception:
        logger.exception("load_taste: failed to load taste profile")
        return {
            "taste_context": "",
            "taste_likes": [],
            "taste_dislikes": [],
            "watched_ids": set(),
        }


def retrieve_taste_memories(state: GraphState):
    message = _get_latest_user_message(state)

    if not message:
        return {}

    try:
        memories = search_relevant_taste(message, limit=5)
    except Exception:
        logger.exception("retrieve_taste_memories: failed to retrieve taste memories")
        return {}

    likes = list(state.get("taste_likes", []))
    dislikes = list(state.get("taste_dislikes", []))
    relevant = []

    for memory in memories:
        text = memory.metadata.get("preference_text", memory.page_content)
        kind = memory.metadata.get("preference_kind")
        relevant.append(text)

        if kind == "like" and text not in likes:
            likes.append(text)
        elif kind == "dislike" and text not in dislikes:
            dislikes.append(text)

    return {
        "taste_likes": likes,
        "taste_dislikes": dislikes,
        "relevant_taste_memories": relevant,
    }
