import logging
from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.repository import DEFAULT_PROFILE_ID, get_preferences

logger = logging.getLogger(__name__)


@lru_cache
def _get_client():
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)

    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    return client


@lru_cache
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_vector_store():
    return QdrantVectorStore(
        client=_get_client(),
        collection_name=get_settings().qdrant_collection,
        embeddings=_get_embeddings(),
    )


def index_preference(preference, profile_id=DEFAULT_PROFILE_ID):
    try:
        kind = str(getattr(preference.kind, "value", preference.kind))
        record_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{profile_id}:{preference.id}",
            )
        )

        document = Document(
            page_content=f"{kind}: {preference.text}",
            metadata={
                "profile_id": profile_id,
                "source_record_id": preference.id,
                "preference_kind": kind,
                "preference_text": preference.text,
                "source": preference.source,
                "source_title": preference.source_title or "",
                "source_tmdb_id": preference.source_tmdb_id or 0,
            },
        )

        get_vector_store().add_documents(
            documents=[document],
            ids=[record_id],
        )
        return True

    except Exception:
        logger.warning("Could not index preferences", exc_info=True)
        return False


def search_relevant_taste(query, profile_id=DEFAULT_PROFILE_ID, limit=5):
    if not query.strip() or limit < 1:
        return []

    try:
        return get_vector_store().similarity_search(
            query,
            k=limit,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="profile_id",
                        match=models.MatchValue(value=profile_id),
                    )
                ]
            ),
        )
    except Exception:
        logger.warning("Could not retrieve taste memories from Qdrant", exc_info=True)
        return []


def rebuild_preference_index():
    try:
        client = _get_client()
        collection = get_settings().qdrant_collection

        client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=get_settings().embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

        get_vector_store.cache_clear()

        count = 0
        for preference in get_preferences(DEFAULT_PROFILE_ID):
            if index_preference(preference):
                count += 1

        return count

    except Exception:
        logger.warning("Could not rebuild the Qdrant preference index", exc_info=True)
        return 0
