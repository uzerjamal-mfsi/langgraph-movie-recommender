import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from langgraph_movie_recommender.graph.nodes.common import TITLE_LOOKUP_ERROR, failed
from langgraph_movie_recommender.tmdb import TMDBError, find_media_candidates

logger = logging.getLogger(__name__)


def _resolved(item):
    return {"resolved_title": item, "needs_clarification": False, "error": None}


def _resolve_title_candidates(title, media_type, year):
    try:
        matches = find_media_candidates(title, media_type, year)
    except (TMDBError, ValueError) as error:
        return failed(str(error), needs_clarification=False)
    except Exception:
        logger.exception("resolve_title: failed to resolve title")
        return failed(TITLE_LOOKUP_ERROR, needs_clarification=False)

    if not matches:
        year_label = f" from {year}" if year is not None else ""
        return failed(
            f"I couldn't match '{title}' to a TMDB {media_type}{year_label}. "
            "Please provide the exact title and year.",
            needs_clarification=False,
        )

    if len(matches) > 1:
        return {
            "clarification_question": (
                f"I found multiple {media_type} titles matching '{title}'. Which one did you mean?"
            ),
            "clarification_candidates": [item.model_dump() for item in matches],
            "needs_clarification": True,
            "error": None,
        }

    return _resolved(matches[0].model_dump())


def resolve_title(state):
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


def ask_clarifying_question(state):
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
            return _resolved(candidates[choice - 1])

    feedback = state.get("feedback") or {}
    media_type_answer = answer.casefold()

    if media_type_answer in {"movie", "tv"}:
        title = feedback.get("title")
        if title:
            return _resolve_title_candidates(title, media_type_answer, feedback.get("year"))

    matches = [item for item in candidates if answer.casefold() == item["title"].casefold()]

    if len(matches) == 1:
        return _resolved(matches[0])

    return {
        "clarification_question": (
            "I couldn't match that answer to one of the options. "
            "Please choose an option by number or give the exact title."
        ),
        "clarification_candidates": candidates,
        "needs_clarification": True,
        "error": None,
    }
