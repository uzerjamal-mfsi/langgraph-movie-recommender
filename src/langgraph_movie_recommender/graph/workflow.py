from langgraph.graph import START, END, StateGraph

from langgraph_movie_recommender.graph.nodes import classify_intent
from langgraph_movie_recommender.graph.state import GraphState


def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_edge(START, "classify_intent")
    workflow.add_edge("classify_intent", END)

    return workflow.compile()