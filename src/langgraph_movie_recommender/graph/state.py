from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

Intent = Literal["recommend", "title_question", "general"]


class ConversationMetadata(TypedDict, total=False):
    thread_id: str
    turn: int


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: NotRequired[Intent]
    filters: NotRequired[dict[str, Any] | None]
    error: NotRequired[str | None]
