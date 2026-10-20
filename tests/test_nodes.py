import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_movie_recommender.graph import nodes
from langgraph_movie_recommender.graph.schemas import (
    IntentClassification,
    PreferenceExtraction,
    PreferenceStatement,
)
from langgraph_movie_recommender.models import LibraryItem, MediaItem, Preference
from langgraph_movie_recommender.tmdb import TMDBError


class FakeClassifier:
    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class FakeStructuredLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return FakeClassifier(self.result)


class FakeLLM:
    def __init__(self, message=None, error=None):
        self.message = message
        self.error = error

    def invoke(self, messages):
        if self.error is not None:
            raise self.error
        return self.message


def make_item():
    return MediaItem(
        tmdb_id=1,
        media_type="movie",
        title="Heat",
        year=1995,
        tmdb_url="https://www.themoviedb.org/movie/1",
    )


def test_classify_intent_success(monkeypatch):
    expected = IntentClassification(intent="recommend", filters={"genres": ["Action"]})
    monkeypatch.setattr(nodes, "get_chat_model", lambda: FakeStructuredLLM(expected))

    result = nodes.classify_intent({"messages": [HumanMessage(content="action movies")]}, {})

    assert result == {"intent": "recommend", "filters": {"genres": ["Action"]}, "error": None}


def test_classify_intent_failure(monkeypatch):
    def raise_error():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(nodes, "get_chat_model", raise_error)

    result = nodes.classify_intent({"messages": [HumanMessage(content="hi")]}, {})

    assert result["error"] == "I couldn't understand your request. Please try again later."
    assert isinstance(result["messages"][0], AIMessage)


def test_route_intent_success():
    assert nodes.route_intent({"intent": "recommend"}) == "recommend"
    assert nodes.route_intent({}) == "general"


def test_route_intent_failure_returns_error_route():
    assert nodes.route_intent({"error": "boom"}) == "error"


def test_answer_general_success(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_chat_model", lambda: FakeLLM(message=AIMessage(content="hello"))
    )

    result = nodes.answer_general({"messages": [HumanMessage(content="hi")]})

    assert result["messages"][0].content == "hello"
    assert result["error"] is None


def test_answer_general_failure(monkeypatch):
    monkeypatch.setattr(
        nodes, "get_chat_model", lambda: FakeLLM(error=RuntimeError("model unavailable"))
    )

    result = nodes.answer_general({"messages": [HumanMessage(content="hi")]})

    assert result["messages"][0].content == "Something went wrong please try again."
    assert result["error"] is None


def test_recommend_success(monkeypatch):
    monkeypatch.setattr(nodes, "find_candidates", lambda filters: [make_item()])
    monkeypatch.setattr(nodes, "format_recommendations", lambda items: "Here are titles")

    result = nodes.recommend({"filters": {}})

    assert result["messages"][0].content == "Here are titles"
    assert result["error"] is None


def test_recommend_failure(monkeypatch):
    def raise_tmdb_error(filters):
        raise TMDBError("TMDB is down")

    monkeypatch.setattr(nodes, "find_candidates", raise_tmdb_error)

    result = nodes.recommend({"filters": None})

    assert result["messages"][0].content == "TMDB is down"
    assert result["error"] is None


def make_library_item(**overrides):
    data = {
        "media_type": "movie",
        "tmdb_id": 1,
        "title": "Heat",
        "year": 1995,
        "watched": True,
        "rating": 9,
    }
    data.update(overrides)
    return LibraryItem(**data)


def test_route_after_title_resolution_success():
    assert nodes.route_after_title_resolution({"error": None}) == "save"


def test_route_after_title_resolution_failure():
    assert nodes.route_after_title_resolution({"error": "boom"}) == "error"


def test_route_after_clarification_success():
    assert nodes.route_after_clarification({"needs_clarification": False}) == "save"


def test_route_after_clarification_failure():
    assert nodes.route_after_clarification({"error": "boom"}) == "error"


def test_extract_feedback_failure():
    result = nodes.extract_feedback({"messages": []})

    assert result["feedback"] is None
    assert result["error"] == "I couldn't find your latest message"


def test_save_feedback_failure():
    result = nodes.save_feedback({"error": "boom"})

    assert result["messages"][0].content == "boom"


def test_extract_preferences_success(monkeypatch):
    extraction = PreferenceExtraction(
        preferences=[PreferenceStatement(text="likes sci-fi", kind="like")]
    )
    monkeypatch.setattr(nodes, "get_chat_model", lambda: FakeStructuredLLM(extraction))

    result = nodes.extract_preferences({"messages": [HumanMessage(content="I like sci-fi")]})

    assert result["extracted_preferences"] == [{"text": "likes sci-fi", "kind": "like"}]
    assert result["error"] is None


def test_extract_preferences_failure():
    result = nodes.extract_preferences({"messages": []})

    assert result["extracted_preferences"] == []
    assert result["error"] is not None


def test_save_preferences_success(monkeypatch):
    monkeypatch.setattr(nodes, "save_preference", lambda profile, preference: preference)
    monkeypatch.setattr(nodes, "get_preferences", lambda profile: [])

    result = nodes.save_preferences(
        {"extracted_preferences": [{"text": "likes sci-fi", "kind": "like"}]}
    )

    assert result["messages"][0].content == "Saved your preferences: likes sci-fi."
    assert result["error"] is None


def test_save_preferences_failure():
    result = nodes.save_preferences({"error": "boom"})

    assert result["messages"][0].content == "boom"


def test_ask_clarifying_question_success(monkeypatch):
    monkeypatch.setattr(nodes, "interrupt", lambda question: "2")

    result = nodes.ask_clarifying_question({"clarification_question": "Which one?"})

    assert result["clarification_answer"] == "2"
    assert [message.content for message in result["messages"]] == ["Which one?", "2"]


def test_ask_clarifying_question_failure(monkeypatch):
    def raise_error(question):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(nodes, "interrupt", raise_error)

    with pytest.raises(RuntimeError):
        nodes.ask_clarifying_question({"clarification_question": "Which one?"})


def test_resolve_classification_success():
    result = nodes.resolve_classification(
        {
            "clarification_answer": "1",
            "clarification_candidates": [make_item().model_dump()],
        }
    )

    assert result["resolved_title"]["title"] == "Heat"
    assert result["needs_clarification"] is False


def test_resolve_classification_failure():
    result = nodes.resolve_classification({"clarification_answer": ""})

    assert result["error"] == "I didn't receive a title choice. Please try again."


def test_load_taste_success(monkeypatch):
    monkeypatch.setattr(nodes, "get_library", lambda profile: [make_library_item()])
    monkeypatch.setattr(
        nodes,
        "get_preferences",
        lambda profile: [
            Preference(id="p1", text="likes sci-fi", kind="like"),
            Preference(id="p2", text="dislikes horror", kind="dislike"),
        ],
    )
    monkeypatch.setattr(nodes, "get_watched_ids", lambda profile: {("movie", 1)})

    result = nodes.load_taste({})

    assert result["taste_likes"] == ["likes sci-fi"]
    assert result["taste_dislikes"] == ["dislikes horror"]
    assert result["watched_ids"] == {("movie", 1)}


def test_load_taste_failure(monkeypatch):
    def raise_error(profile):
        raise RuntimeError("db down")

    monkeypatch.setattr(nodes, "get_library", raise_error)

    with pytest.raises(RuntimeError):
        nodes.load_taste({})


def test_retrieve_taste_memories_success(monkeypatch):
    document = Document(
        page_content="like: likes sci-fi",
        metadata={"preference_text": "likes sci-fi", "preference_kind": "like"},
    )
    monkeypatch.setattr(nodes, "search_relevant_taste", lambda message, limit: [document])

    result = nodes.retrieve_taste_memories(
        {"messages": [HumanMessage(content="sci-fi")], "taste_likes": []}
    )

    assert result["relevant_taste_memories"] == ["likes sci-fi"]
    assert result["taste_likes"] == ["likes sci-fi"]


def test_retrieve_taste_memories_failure(monkeypatch):
    def raise_error(message, limit):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(nodes, "search_relevant_taste", raise_error)

    with pytest.raises(RuntimeError):
        nodes.retrieve_taste_memories({"messages": [HumanMessage(content="sci-fi")]})
