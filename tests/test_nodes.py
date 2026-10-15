from langchain_core.messages import AIMessage, HumanMessage

from langgraph_movie_recommender.graph import nodes
from langgraph_movie_recommender.graph.schemas import IntentClassification
from langgraph_movie_recommender.models import MediaItem
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

    result = nodes.recommend({"filters": {"genres": ["Action"]}})

    assert result["messages"][0].content == "Here are titles"
    assert result["error"] is None


def test_recommend_failure(monkeypatch):
    def raise_tmdb_error(filters):
        raise TMDBError("TMDB is down")

    monkeypatch.setattr(nodes, "find_candidates", raise_tmdb_error)

    result = nodes.recommend({"filters": None})

    assert result["messages"][0].content == "TMDB is down"
    assert result["error"] is None
