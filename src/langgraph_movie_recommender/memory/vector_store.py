import logging
from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.repository import get_preferences

logger = logging.getLogger(__name__)


def _probe_embedding_dimensions(embeddings):
    """Return the vector size produced by an embeddings implementation."""
    client = getattr(embeddings, "_client", None)
    getter = getattr(client, "get_embedding_dimension", None) or getattr(
        client, "get_sentence_embedding_dimension", None
    )

    if callable(getter):
        dimensions = getter()
        if isinstance(dimensions, int) and dimensions > 0:
            return dimensions

    return len(embeddings.embed_query("dimension probe"))


@lru_cache
def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=get_settings().embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def _resolve_embedding_dimensions():
    settings = get_settings()
    dimensions = _probe_embedding_dimensions(_get_embeddings())
    configured = settings.embedding_dimensions

    if configured is not None and configured != dimensions:
        raise ValueError(
            f"EMBEDDING_DIMENSIONS={configured} does not match model "
            f"'{settings.embedding_model}', which produces {dimensions}-dimensional vectors. "
            "Unset EMBEDDING_DIMENSIONS to auto-detect it, or set it to the model's dimension."
        )

    return dimensions


def _collection_vector_size(client, collection_name):
    info = client.get_collection(collection_name)
    vectors = info.config.params.vectors

    if isinstance(vectors, dict):
        vectors = next(iter(vectors.values()), None)

    return getattr(vectors, "size", None)


def _ensure_collection(client, settings):
    exists = client.collection_exists(settings.qdrant_collection)
    dimensions = _resolve_embedding_dimensions()

    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection %s with %d dimensions",
            settings.qdrant_collection,
            dimensions,
        )
        return

    stored_dimensions = _collection_vector_size(client, settings.qdrant_collection)

    if stored_dimensions is not None and stored_dimensions != dimensions:
        raise ValueError(
            f"Qdrant collection '{settings.qdrant_collection}' stores {stored_dimensions}-"
            f"dimensional vectors, but model '{settings.embedding_model}' produces "
            f"{dimensions}. Align EMBEDDING_MODEL/EMBEDDING_DIMENSIONS and rebuild the "
            "index with 'movie-recommender --rebuild-index'."
        )


@lru_cache
def _get_client():
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url)
    _ensure_collection(client, settings)
    return client


@lru_cache
def get_vector_store():
    return QdrantVectorStore(
        client=_get_client(),
        collection_name=get_settings().qdrant_collection,
        embedding=_get_embeddings(),
    )


def index_preference(preference):
    try:
        record_id = str(uuid5(NAMESPACE_URL, preference.id))

        document = Document(
            page_content=f"{preference.kind}: {preference.text}",
            metadata={
                "source_record_id": preference.id,
                "preference_kind": preference.kind,
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


def search_relevant_taste(query, limit=5):
    if not query.strip() or limit < 1:
        return []

    try:
        return get_vector_store().similarity_search(query, k=limit)
    except Exception:
        logger.warning("Could not retrieve taste memories from Qdrant", exc_info=True)
        return []


def rebuild_preference_index():
    settings = get_settings()
    collection = settings.qdrant_collection
    dimensions = _resolve_embedding_dimensions()
    client = QdrantClient(url=settings.qdrant_url)

    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)

        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimensions,
                distance=models.Distance.COSINE,
            ),
        )
    finally:
        client.close()

    _get_client.cache_clear()
    get_vector_store.cache_clear()

    count = 0
    for preference in get_preferences():
        if not index_preference(preference):
            raise RuntimeError(f"Failed to index preference {preference.id} in Qdrant.")
        count += 1

    return count
