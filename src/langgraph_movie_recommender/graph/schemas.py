from typing import Any

from pydantic import BaseModel, Field

from langgraph_movie_recommender.graph.state import Intent


class IntentClassification(BaseModel):
    intent: Intent
    filters: dict[str, Any] | None = Field(
        description="Relevant movie or TV show preferences and filters extracted from the user's message or None if there are none."
    )