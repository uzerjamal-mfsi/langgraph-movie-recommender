from langchain_core.messages import SystemMessage

from langgraph_movie_recommender.graph.llm import get_chat_model
from langgraph_movie_recommender.graph.prompts import CLASSIFIER_PROMPT
from langgraph_movie_recommender.graph.schemas import IntentClassification
from langgraph_movie_recommender.graph.state import GraphState


def classify_intent(state: GraphState, config):
    try:
        llm = get_chat_model()
        classifier = llm.with_structured_output(IntentClassification)

        result = classifier.invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            *state["messages"],
        ])

        return {
            "intent": result.intent,
            "fitlers": result.filters,
            "error": None
        }
    except Exception:
        return {
            "error": "I couldn't understand your request. Please try again later."
        }