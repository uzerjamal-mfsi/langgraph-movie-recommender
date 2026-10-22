from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

Intent = Literal[
    "recommend",
    "title_question",
    "general",
    "record_feedback",
    "update_preferences",
]


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: NotRequired[Intent]
    filters: NotRequired[dict[str, Any] | None]
    error: NotRequired[str | None]
    feedback: NotRequired[dict[str, Any] | None]
    resolved_title: NotRequired[dict[str, Any] | None]
    extracted_preferences: NotRequired[list[dict[str, Any]]]
    saved_summary: NotRequired[str | None]
    clarification_question: NotRequired[str]
    clarification_candidates: NotRequired[list[dict[str, Any]]]
    clarification_answer: NotRequired[str]
    needs_clarification: NotRequired[bool]
    taste_context: NotRequired[str]
    taste_likes: NotRequired[list[str]]
    taste_dislikes: NotRequired[list[str]]
    relevant_taste_memories: NotRequired[list[str]]
    watched_ids: NotRequired[list[str]]
