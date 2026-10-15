from langgraph.graph import END, START, StateGraph

from langgraph_movie_recommender.graph.nodes import (
    answer_general,
    classify_intent,
    extract_feedback,
    extract_preferences,
    recommend,
    resolve_title,
    route_intent,
    save_feedback,
    save_preferences,
)
from langgraph_movie_recommender.graph.state import GraphState


def build_workflow(checkpointer=None):
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("recommend", recommend)
    workflow.add_node("answer_general", answer_general)
    workflow.add_node("extract_feedback", extract_feedback)
    workflow.add_node("resolve_title", resolve_title)
    workflow.add_node("save_feedback", save_feedback)
    workflow.add_node("extract_preferences", extract_preferences)
    workflow.add_node("save_preferences", save_preferences)

    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "recommend": "recommend",
            "title_question": "answer_general",
            "general": "answer_general",
            "record_feedback": "extract_feedback",
            "update_preferences": "extract_preferences",
            "error": END,
        },
    )

    workflow.add_edge("recommend", END)
    workflow.add_edge("answer_general", END)

    workflow.add_edge("extract_feedback", "resolve_title")
    workflow.add_edge("resolve_title", "save_feedback")
    workflow.add_edge("save_feedback", END)

    workflow.add_edge("extract_preferences", "save_preferences")
    workflow.add_edge("save_preferences", END)

    return workflow.compile(checkpointer=checkpointer)
