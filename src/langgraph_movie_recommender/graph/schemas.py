from typing import Any

from pydantic import BaseModel, Field

from langgraph_movie_recommender.graph.state import Intent


class IntentClassification(BaseModel):
    intent: Intent
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Recommendation filters, when relevant. Supported keys include media_type (movie, tv, or both), genres (list of genre names), year, year_from, year_to, max_runtime, and query for a specific title or search phrase. Use null when there are no filters."
    )