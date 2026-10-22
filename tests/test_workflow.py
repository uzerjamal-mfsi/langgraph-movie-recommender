from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_movie_recommender.graph import nodes
from langgraph_movie_recommender.graph.schemas import (
    FeedbackExtraction,
    IntentClassification,
)
from langgraph_movie_recommender.graph.workflow import build_workflow
from langgraph_movie_recommender.models import MediaItem


class FakeStructuredLLM:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class FakeWorkflowLLM:
    def __init__(self, classification, feedback):
        self.classification = classification
        self.feedback = feedback

    def with_structured_output(self, schema):
        if schema is IntentClassification:
            return FakeStructuredLLM(self.classification)
        if schema is FeedbackExtraction:
            return FakeStructuredLLM(self.feedback)
        raise AssertionError(f"Unexpected schema: {schema}")


def make_item(tmdb_id, year):
    return MediaItem(
        tmdb_id=tmdb_id,
        media_type="movie",
        title="Heat",
        year=year,
        tmdb_url=f"https://www.themoviedb.org/movie/{tmdb_id}",
    )


def test_clarification_interrupts_and_resumes(monkeypatch):
    classification = IntentClassification(intent="record_feedback", filters=None)
    feedback = FeedbackExtraction(
        title="Heat",
        media_type="movie",
        year=None,
        watched=True,
        rating=9,
        review="Great",
        positive_aspects=[],
        negative_aspects=[],
    )
    monkeypatch.setattr(
        nodes.classification,
        "get_chat_model",
        lambda: FakeWorkflowLLM(classification, feedback),
    )
    monkeypatch.setattr(
        nodes.titles,
        "find_media_candidates",
        lambda title, media_type, year: [make_item(1, 1995), make_item(2, 1986)],
    )
    monkeypatch.setattr(
        nodes.feedback,
        "find_library_item",
        lambda media_type, tmdb_id: None,
    )
    saved = {}

    def save_feedback_and_preferences(item, preferences):
        saved["item"] = item
        return item, preferences

    monkeypatch.setattr(
        nodes.feedback,
        "save_feedback_and_preferences",
        save_feedback_and_preferences,
    )
    monkeypatch.setattr(nodes.feedback, "index_preference", lambda preference: True)

    workflow = build_workflow(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "clarification-test"}}

    interrupted = workflow.invoke(
        {"messages": [HumanMessage(content="I watched Heat. 9/10.")]},
        config=config,
    )

    assert interrupted["__interrupt__"]

    resumed = workflow.invoke(Command(resume="1"), config=config)

    assert saved["item"].tmdb_id == 1
    assert "Saved feedback for Heat" in resumed["messages"][-1].content
