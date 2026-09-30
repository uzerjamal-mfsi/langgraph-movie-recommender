from langgraph.graph import START, END, StateGraph

from langgraph_movie_recommender.graph.nodes import answer_general, classify_intent, recommend, route_intent
from langgraph_movie_recommender.graph.state import GraphState


def build_workflow(checkpointer=None):
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("recommend", recommend)
    workflow.add_node("answer_general", answer_general)

    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges("classify_intent", route_intent, {
        "recommend": "recommend",
        "title_question": "answer_general",
        "general": "answer_general",
        "error": END,
    })

    workflow.add_edge("recommend", END)
    workflow.add_edge("answer_general", END)

    return workflow.compile(checkpointer=checkpointer)