__version__ = "0.1.0"

from ._client import Client
from ._vector_ops import VectorOperations
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
