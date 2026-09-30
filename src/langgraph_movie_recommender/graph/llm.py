from functools import lru_cache

from langchain_openrouter import ChatOpenRouter

from langgraph_movie_recommender.config import get_settings

MODEL_TIMEOUT_MILLISECONDS = 45000


@lru_cache
def get_chat_model():
    settings = get_settings()
    return ChatOpenRouter(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        temperature=0,
        max_retries=2,
        request_timeout=MODEL_TIMEOUT_MILLISECONDS,
    )
