__version__ = "0.2.0"

from ._client import Client
from ._vector_ops import VectorOperations
from ._embed_ops import EmbeddingOperations
from ._types import (
    VectorRecord,
    SearchResult,
    SearchResults,
    BatchSearchResults,
    SearchParams,
    UpsertParams,
    FilterParams
)


async def create_client(
        host: str,
        timeout: int = 120,
        debug: bool = False,
        use_http2: bool = True,
        cache_size: int = 1000,
        connection_pool_size: int = 0,
        thread_pool_size: int = 0
) -> Client:
    """
    Create Antarys client

    Args:
        host: Antarys server host address
        timeout: Request timeout in seconds
        debug: Enable debug mode for additional logging
        use_http2: Use HTTP/2 for improved performance
        cache_size: Size of client-side query result cache (0 to disable)
        connection_pool_size: Connection pool size for concurrent requests (0 = auto)
        thread_pool_size: Size of thread pool for CPU-bound tasks (0 = auto)

    Returns:
        Configured Antarys client
    """
    return Client(
        host=host,
        timeout=timeout,
        debug=debug,
        use_http2=use_http2,
        cache_size=cache_size,
        connection_pool_size=connection_pool_size,
        thread_pool_size=thread_pool_size
    )


async def create_collection(
        client: Client,
        name: str,
        dimensions: int,  # Now required parameter
        enable_hnsw: bool = True,
        shards: int = 16,
        m: int = 16,
        ef_construction: int = 200,
) -> dict:
    """
    Create a new vector collection with specified dimensions

    Args:
        client: Antarys client instance
        name: Collection name
        dimensions: Vector dimensions for this collection (required)
        enable_hnsw: Enable HNSW indexing
        shards: Number of shards (for performance)
        m: HNSW M parameter
        ef_construction: HNSW ef construction parameter

    Returns:
        Response with creation confirmation
    """
    return await client.create_collection(
        name=name,
        dimensions=dimensions,
        enable_hnsw=enable_hnsw,
        shards=shards,
        m=m,
        ef_construction=ef_construction
    )


async def embed(
        client: Client,
        texts,
        batch_size: int = 256,
):
    """
    Generate embeddings for text(s) using the Antarys embedding API

    Args:
        client: Antarys client instance
        texts: Single text string or list of text strings
        batch_size: Batch size for processing (default: 256)

    Returns:
        Single embedding vector if input is string, or list of embedding vectors if input is list

    Example:
        # Single text
        embedding = await embed(client, "Hello, World!")

        # Multiple texts
        embeddings = await embed(client, ["First doc", "Second doc"])
    """
    embed_ops = client.embedding_operations()
    return await embed_ops.embed(texts, batch_size=batch_size)


async def embed_query(client: Client, query: str):
    """
    Generate embedding for a search query with "query: " prefix

    Args:
        client: Antarys client instance
        query: Query text

    Returns:
        Embedding vector

    Example:
        query_embedding = await embed_query(client, "What is machine learning?")
    """
    embed_ops = client.embedding_operations()
    return await embed_ops.embed_query(query)


async def embed_documents(
        client: Client,
        documents,
        batch_size: int = 256,
        show_progress: bool = False,
):
    """
    Generate embeddings for documents with "passage: " prefix

    Args:
        client: Antarys client instance
        documents: List of document texts
        batch_size: Batch size for API requests
        show_progress: Show progress bar during processing

    Returns:
        List of embedding vectors

    Example:
        doc_embeddings = await embed_documents(
            client,
            documents=["Python is a language", "JavaScript for web"],
            show_progress=True
        )
    """
    embed_ops = client.embedding_operations()
    return await embed_ops.embed_documents(
        documents,
        batch_size=batch_size,
        show_progress=show_progress
    )


async def text_similarity(
        client: Client,
        text1: str,
        text2: str,
) -> float:
    """
    Calculate cosine similarity between two texts

    Args:
        client: Antarys client instance
        text1: First text
        text2: Second text

    Returns:
        Cosine similarity score (0 to 1)

    Example:
        score = await text_similarity(
            client,
            "machine learning",
            "artificial intelligence"
        )
    """
    embed_ops = client.embedding_operations()
    return await embed_ops.similarity(text1, text2)
