import asyncio
import time
from typing import Dict, List, Any, Optional
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import orjson
from cachetools import LRUCache, TTLCache
import platform
import os

from ._vector_ops import VectorOperations


class Client:
    """Antarys client for vector database operations"""

    def __init__(
            self,
            host: str,
            timeout: int = 120,
            connection_pool_size: int = 0,
            retry_attempts: int = 5,
            compression: bool = True,
            debug: bool = False,
            use_http2: bool = True,
            cache_size: int = 1000,
            thread_pool_size: int = 0,
            cache_ttl: int = 300,
    ):
        """
        Initialise Antarys client with performance optimisations

        Args:
            host: API host address (e.g., "http://localhost:8080")
            timeout: Request timeout in seconds
            connection_pool_size: Connection pool size for concurrent requests (0 = auto)
            retry_attempts: Number of retry attempts for failed requests
            compression: Use compression for requests/responses
            debug: Enable debug mode for additional logging
            use_http2: Use HTTP/2 transport for multiplexing and performance
            cache_size: Size of client-side query result cache (0 to disable)
            thread_pool_size: Size of thread pool for CPU-bound tasks (0 = auto)
            cache_ttl: Time-to-live for cached items in seconds
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.compression = compression
        self.debug = debug
        self.use_http2 = use_http2
        cpu_count = os.cpu_count() or 4

        if connection_pool_size <= 0:
            connection_pool_size = max(cpu_count * 5, 20)
        self.connection_pool_size = connection_pool_size

        if thread_pool_size <= 0:
            thread_pool_size = max(cpu_count * 2, 8)
        self.thread_pool_size = thread_pool_size

        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_pool_size,
            thread_name_prefix="antarys_worker"
        )

        headers = {"Content-Type": "application/json"}
        if compression:
            headers["Accept-Encoding"] = "gzip, deflate, br"

        transport_args = {}
        if use_http2:
            transport_args = {
                "http2": True
            }

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=connection_pool_size,
                max_keepalive_connections=connection_pool_size // 2,
                keepalive_expiry=60.0
            ),
            headers=headers,
            **transport_args
        )

        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        if cache_size > 0:
            self._setup_cache(cache_size, cache_ttl)
            self._cache_hits = 0
            self._cache_misses = 0

        self._collection_cache = {}
        self._request_semaphore = asyncio.Semaphore(connection_pool_size // 2)
        self._requests_total = 0
        self._request_times = []
        self._start_time = time.time()

        if debug:
            print(f"Antarys client initialized with {thread_pool_size} threads, "
                  f"{connection_pool_size} connections, HTTP/2: {use_http2}")
            print(f"Running on {platform.system()} {platform.release()}, "
                  f"Python {platform.python_version()}, {cpu_count} CPUs")

    def _setup_cache(self, cache_size: int, cache_ttl: int):
        """Set up efficient client-side caching"""
        self._query_cache = TTLCache(maxsize=cache_size, ttl=cache_ttl)
        self._metadata_cache = LRUCache(maxsize=cache_size)
        self._cache_lock = asyncio.Lock()

    @staticmethod
    def _compute_query_cache_key(collection: str, vector: np.ndarray,
                                 top_k: int, **params) -> str:
        """Compute an efficient cache key for vector queries"""

        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        vector_hash = hash(vector.tobytes())

        params_bytes = orjson.dumps({
            "collection": collection,
            "top_k": top_k,
            **{k: v for k, v in params.items() if k not in ['vector']}
        })

        key = f"{collection}:{vector_hash}:{top_k}:{hash(params_bytes)}"
        return key

    async def create_collection(
            self,
            name: str,
            dimensions: int = 1536,
            enable_hnsw: bool = True,
            shards: int = 16,
            m: int = 16,
            ef_construction: int = 200,
    ) -> Dict:
        """
        Create a new vector collection with specified dimensions

        Args:
            name: Collection name
            dimensions: Vector dimensions for this collection (required)
            enable_hnsw: Enable HNSW indexing
            shards: Number of shards (for performance)
            m: HNSW M parameter
            ef_construction: HNSW ef construction parameter
        """
        payload = {
            "name": name,
            "dimensions": dimensions,
            "enable_hnsw": enable_hnsw,
            "shards": shards,
            "m": m,
            "ef_construction": ef_construction
        }

        payload_bytes = orjson.dumps(payload)

        async with self._request_semaphore:
            start_time = time.time()
            self._requests_total += 1

            try:
                response = await self.client.post(
                    f"{self.host}/collections",
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                    timeout=httpx.Timeout(60.0)
                )

                if response.status_code >= 400:
                    error_msg = f"Failed to create collection: HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg += f" - {error_data['error']}"
                    except Exception:
                        error_msg += f" - {response.text[:100]}"

                    raise Exception(error_msg)

                self._request_times.append(time.time() - start_time)

                result = orjson.loads(response.content)

                if result.get("success"):
                    self._collection_cache[name] = {
                        "dimensions": dimensions,
                        "created_at": time.time()
                    }

                return result
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    async def list_collections(self) -> List[Dict]:
        """List all collections with their metadata including dimensions"""

        cache_key = "list_collections"
        if hasattr(self, '_metadata_cache') and cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        async with self._request_semaphore:
            self._requests_total += 1
            start_time = time.time()

            try:
                response = await self.client.get(
                    f"{self.host}/collections",
                    timeout=httpx.Timeout(30.0)
                )

                if response.status_code >= 400:
                    raise Exception(f"Failed to list collections: HTTP {response.status_code}")

                self._request_times.append(time.time() - start_time)

                result = orjson.loads(response.content)
                collections = result.get("collections", [])

                for collection in collections:
                    if isinstance(collection, dict) and 'name' in collection:
                        collection_name = collection['name']

                        try:
                            collection_info = await self.describe_collection(collection_name)
                            if collection_info.get('dimensions'):
                                self._collection_cache[collection_name] = {
                                    "dimensions": collection_info['dimensions'],
                                    "updated_at": time.time()
                                }
                        except Exception:
                            pass

                if hasattr(self, '_metadata_cache'):
                    self._metadata_cache[cache_key] = collections

                return collections
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    async def describe_collection(self, name: str) -> Dict:
        """Get collection information including dimensions"""

        cache_key = f"describe_{name}"
        if hasattr(self, '_metadata_cache') and cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        async with self._request_semaphore:
            self._requests_total += 1
            start_time = time.time()

            try:
                response = await self.client.get(
                    f"{self.host}/collections/{name}",
                    timeout=httpx.Timeout(30.0)
                )

                if response.status_code >= 400:
                    raise Exception(f"Failed to describe collection: HTTP {response.status_code}")

                self._request_times.append(time.time() - start_time)

                result = orjson.loads(response.content)

                if result.get('dimensions'):
                    self._collection_cache[name] = {
                        "dimensions": result['dimensions'],
                        "updated_at": time.time()
                    }

                if hasattr(self, '_metadata_cache'):
                    self._metadata_cache[cache_key] = result

                return result
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    async def get_collection_dimensions(self, collection_name: str) -> Optional[int]:
        """
        Get the dimensions for a specific collection

        Args:
            collection_name: Name of the collection

        Returns:
            Number of dimensions for the collection, or None if not found
        """

        if collection_name in self._collection_cache:
            return self._collection_cache[collection_name].get('dimensions')

        try:

            collection_info = await self.describe_collection(collection_name)
            return collection_info.get('dimensions')
        except Exception:
            return None

    async def validate_vector_dimensions(self, collection_name: str, vector: List[float]) -> bool:
        """
        Validate that a vector has the correct dimensions for a collection

        Args:
            collection_name: Name of the collection
            vector: Vector to validate

        Returns:
            True if dimensions match, False otherwise
        """
        expected_dims = await self.get_collection_dimensions(collection_name)
        if expected_dims is None:
            return True

        return len(vector) == expected_dims

    async def delete_collection(self, name: str) -> Dict:
        """Delete a collection and clear related caches"""
        async with self._request_semaphore:
            self._requests_total += 1
            start_time = time.time()

            try:
                response = await self.client.delete(
                    f"{self.host}/collections/{name}",
                    timeout=httpx.Timeout(30.0)
                )

                if response.status_code >= 400:
                    raise Exception(f"Failed to delete collection: HTTP {response.status_code}")

                self._request_times.append(time.time() - start_time)

                if name in self._collection_cache:
                    del self._collection_cache[name]

                if hasattr(self, '_query_cache'):
                    self._query_cache.clear()

                if hasattr(self, '_metadata_cache'):
                    cache_keys_to_remove = [
                        "list_collections",
                        f"describe_{name}"
                    ]
                    for key in cache_keys_to_remove:
                        if key in self._metadata_cache:
                            del self._metadata_cache[key]

                return orjson.loads(response.content)
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    async def commit(self) -> Dict:
        """Force a commit to disk"""
        async with self._request_semaphore:
            self._requests_total += 1
            start_time = time.time()

            try:
                response = await self.client.post(
                    f"{self.host}/admin/commit",
                    timeout=httpx.Timeout(30.0)
                )

                if response.status_code >= 400:
                    raise Exception(f"Failed to commit: HTTP {response.status_code}")

                self._request_times.append(time.time() - start_time)

                return orjson.loads(response.content)
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    def vector_operations(self, collection_name: str = "default") -> VectorOperations:
        """Get vector operations interface for a collection with optimized settings"""
        vector = VectorOperations(
            host=self.host,
            collection_name=collection_name,
            client=self.client,
            batch_size=5000,
            thread_pool=self._thread_pool,
            query_cache=getattr(self, '_query_cache', None) if hasattr(self,
                                                                       'cache_size') and self.cache_size > 0 else None,
            cache_lock=getattr(self, '_cache_lock', None) if hasattr(self,
                                                                     'cache_size') and self.cache_size > 0 else None,
            compute_cache_key=self._compute_query_cache_key if hasattr(self,
                                                                       'cache_size') and self.cache_size > 0 else None,
            request_semaphore=self._request_semaphore,
            debug=self.debug,
            collection_cache=self._collection_cache
        )

        if self.debug:
            print(f"Created VectorOperations for collection '{collection_name}'")
            original_upsert_batch = vector.upsert_batch

            async def debug_upsert_batch(batch):
                """Debug wrapper for _upsert_batch"""
                print(f"DEBUG: Upserting batch of {len(batch)} vectors")
                if batch:
                    first_vec_sample = {k: v if k != "values" and k != "vector"
                    else f"[{len(v)} values]" for k, v in batch[0].items()}
                    print(f"DEBUG: First vector sample: {orjson.dumps(first_vec_sample).decode()}")
                try:
                    result = await original_upsert_batch(batch)
                    print(f"DEBUG: Upsert success: {orjson.dumps(result).decode()}")
                    return result
                except Exception as e:
                    print(f"DEBUG: Upsert failed: {str(e)}")
                    raise

            vector.upsert_batch = debug_upsert_batch

        return vector

    async def health(self) -> Dict:
        """Check server health"""
        async with self._request_semaphore:
            try:
                response = await self.client.get(
                    f"{self.host}/health",
                    timeout=httpx.Timeout(10.0)
                )

                if response.status_code >= 400:
                    raise Exception(f"Health check failed: HTTP {response.status_code}")

                return orjson.loads(response.content)
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    async def info(self) -> Dict:
        """Get server information with caching"""

        cache_key = "server_info"
        if hasattr(self, '_metadata_cache') and cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        async with self._request_semaphore:
            self._requests_total += 1
            start_time = time.time()

            try:
                response = await self.client.get(
                    f"{self.host}/info",
                    timeout=httpx.Timeout(10.0)
                )

                if response.status_code >= 400:
                    raise Exception(f"Failed to get info: HTTP {response.status_code}")

                self._request_times.append(time.time() - start_time)

                result = orjson.loads(response.content)

                if hasattr(self, '_metadata_cache'):
                    self._metadata_cache[cache_key] = result

                return result
            except httpx.RequestError as e:
                raise Exception(f"Request error: {str(e)}")

    async def batch_insert(self, records: List[Dict], collection_name: str = "default",
                           batch_size: int = 5000, show_progress: bool = True,
                           parallelism: int = 0, validate_dimensions: bool = True) -> Dict:
        """
        Batch insertion with parallel processing and dimension validation

        Args:
            records: List of vector records with id, values, and optional metadata
            collection_name: Target collection name
            batch_size: Size of batches for insertion
            show_progress: Show progress during insertion
            parallelism: Number of parallel insertion workers (0 = auto)
            validate_dimensions: Whether to validate vector dimensions before insertion

        Returns:
            Result with count of inserted vectors
        """

        if validate_dimensions and records:
            expected_dims = await self.get_collection_dimensions(collection_name)
            if expected_dims is not None:
                for i, record in enumerate(records[:10]):
                    vector = record.get('values') or record.get('vector', [])
                    if len(vector) != expected_dims:
                        raise ValueError(f"Vector dimension mismatch in record {i}: "
                                         f"got {len(vector)}, expected {expected_dims} "
                                         f"for collection '{collection_name}'")

        if parallelism <= 0:
            parallelism = min(os.cpu_count() or 4, 8)

        vector = self.vector_operations(collection_name)

        return await vector.upsert(
            records,
            batch_size=batch_size,
            show_progress=show_progress,
            parallel_workers=parallelism
        )

    async def batch_insert_with_gpu(self, collection_name: str, records: List[Dict]) -> Dict:
        """
        Alias for batch_insert to maintain API compatibility with Go client

        Args:
            collection_name: Target collection name
            records: List of vector records with id, values, and optional metadata

        Returns:
            Result with count of inserted vectors
        """
        return await self.batch_insert(records, collection_name)

    def get_query_interface(self, collection_name: str = "default"):
        """
        Get a query interface for a collection

        This is a more intuitive alternative to vector_operations().

        Args:
            collection_name: Target collection name

        Returns:
            VectorOperations instance for queries
        """
        return self.vector_operations(collection_name)

    async def clear_cache(self):
        """Clear all client-side caches"""
        if hasattr(self, '_query_cache'):
            self._query_cache.clear()
        if hasattr(self, '_metadata_cache'):
            self._metadata_cache.clear()
        self._collection_cache.clear()

        if hasattr(self, '_cache_hits'):
            self._cache_hits = 0
        if hasattr(self, '_cache_misses'):
            self._cache_misses = 0

        return {"success": True, "message": "All caches cleared"}

    async def close(self):
        """Properly close the client and release all resources"""

        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)

        if self.client:
            await self.client.aclose()

        if hasattr(self, '_query_cache'):
            self._query_cache.clear()
        if hasattr(self, '_metadata_cache'):
            self._metadata_cache.clear()
        self._collection_cache.clear()
