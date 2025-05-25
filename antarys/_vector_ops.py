import os
import asyncio
import numpy as np
import orjson
import httpx
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
from typing import Dict, List, Optional, Any, Union, Callable

try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class VectorOperations:
    """Vector operation interface for collections"""

    def __init__(
            self,
            host: str,
            collection_name: str,
            client: httpx.AsyncClient,
            batch_size: int = 5000,
            thread_pool: Optional[ThreadPoolExecutor] = None,
            query_cache: Optional[Dict] = None,
            cache_lock: Optional[asyncio.Lock] = None,
            compute_cache_key: Optional[Callable] = None,
            request_semaphore: Optional[asyncio.Semaphore] = None,
            debug: bool = False,
            collection_cache: Optional[Dict] = None,  # New parametwer for collection metadata caching
    ):
        """
        Initialize vector operations with performance optimizations

        Args:
            host: API host address
            collection_name: Collection name to operate on
            client: Shared HTTP client
            batch_size: Default batch size for operations (optimized)
            thread_pool: Thread pool for CPU-bound operations
            query_cache: Optional shared query cache
            cache_lock: Lock for cache access
            compute_cache_key: Function to compute cache keys
            request_semaphore: Semaphore for limiting concurrent requests
            debug: Enable debug mode for additional logging
            collection_cache: Cache for collection metadata including dimensions
        """
        self.host = host.rstrip("/")
        self.collection_name = collection_name
        self.client = client
        self.batch_size = batch_size
        self.debug = debug
        self._query_cache = query_cache
        self._cache_lock = cache_lock
        self._compute_cache_key = compute_cache_key
        self._cache_hits = 0
        self._cache_misses = 0
        self._collection_cache = collection_cache or {}

        if thread_pool is None:
            cpu_count = os.cpu_count() or 4
            self._thread_pool = ThreadPoolExecutor(
                max_workers=cpu_count * 2,
                thread_name_prefix=f"vector_ops_{collection_name}"
            )
            self._owns_thread_pool = True
        else:
            self._thread_pool = thread_pool
            self._owns_thread_pool = False

        if request_semaphore is None:
            cpu_count = os.cpu_count() or 4
            self._request_semaphore = asyncio.Semaphore(cpu_count * 5)
        else:
            self._request_semaphore = request_semaphore

        if NUMBA_AVAILABLE:
            self._setup_optimized_functions()
            if debug:
                print(f"Using Numba for accelerated vector operations")

        self._setup_reusable_buffers()

    def _setup_optimized_functions(self):
        """Set up Numba-optimized functions for vector operations"""
        if not NUMBA_AVAILABLE:
            return

        # Numba-optimized cosine similarity computation
        @numba.njit(parallel=True, fastmath=True)
        def cosine_similarity_batch(query_vec, matrix):
            """Compute cosine similarity between a query vector and a matrix of vectors"""
            query_norm = np.sqrt(np.sum(query_vec * query_vec))
            if query_norm == 0:
                query_norm = 1.0
            query_vec = query_vec / query_norm
            n_vectors = matrix.shape[0]
            similarities = np.zeros(n_vectors, dtype=np.float32)

            for i in numba.prange(n_vectors):
                vec = matrix[i]
                vec_norm = np.sqrt(np.sum(vec * vec))
                if vec_norm == 0:
                    vec_norm = 1.0
                similarities[i] = np.sum(query_vec * (vec / vec_norm))

            return similarities

        self._cosine_similarity_batch = cosine_similarity_batch

        @numba.njit(parallel=True, fastmath=True)
        def euclidean_distance_batch(query_vec, matrix):
            """Compute Euclidean distance between a query vector and a matrix of vectors"""
            n_vectors = matrix.shape[0]
            distances = np.zeros(n_vectors, dtype=np.float32)

            for i in numba.prange(n_vectors):
                vec = matrix[i]
                diff = query_vec - vec
                distances[i] = np.sqrt(np.sum(diff * diff))

            return distances

        self._euclidean_distance_batch = euclidean_distance_batch

        @numba.njit(fastmath=True)
        def normalize_vector(vector):
            """Normalize a vector to unit length"""
            norm = np.sqrt(np.sum(vector * vector))
            if norm == 0:
                return vector
            return vector / norm

        self._normalize_vector = normalize_vector

    def _setup_reusable_buffers(self):
        """Set up reusable buffers for better memory efficiency"""
        self._vector_array_pool = []
        self._vector_array_pool_lock = asyncio.Lock()

        self._query_payload_template = {
            "collection": self.collection_name,
            "include_vectors": False,
            "include_metadata": True,
            "use_ann": True,
            "threshold": 0.0
        }

        self._batch_results_buffer = []

    async def get_collection_dimensions(self) -> Optional[int]:
        """
        Get the expected dimensions for this collection

        Returns:
            Number of dimensions expected for vectors in this collection
        """

        if self.collection_name in self._collection_cache:
            cache_entry = self._collection_cache[self.collection_name]
            if isinstance(cache_entry, dict) and 'dimensions' in cache_entry:
                return cache_entry['dimensions']

        try:
            response = await self.client.get(
                f"{self.host}/collections/{self.collection_name}",
                timeout=httpx.Timeout(10.0)
            )

            if response.status_code == 200:
                collection_info = orjson.loads(response.content)
                dimensions = collection_info.get('dimensions')

                if dimensions is not None:
                    self._collection_cache[self.collection_name] = {
                        'dimensions': dimensions,
                        'updated_at': asyncio.get_event_loop().time()
                    }

                return dimensions
        except Exception as e:
            if self.debug:
                print(f"Warning: Could not fetch collection dimensions: {e}")

        return None

    async def validate_vector_dimensions(self, vector: Union[List[float], np.ndarray]) -> bool:
        """
        Validate that a vector has the correct dimensions for this collection

        Args:
            vector: Vector to validate

        Returns:
            True if dimensions are valid, False otherwise
        """
        expected_dims = await self.get_collection_dimensions()
        if expected_dims is None:
            return True

        vector_length = len(vector)
        return vector_length == expected_dims

    async def _validate_batch_dimensions(self, vectors: List[Dict[str, Any]]) -> List[str]:
        """
        Validate dimensions for a batch of vectors

        Args:
            vectors: List of vector records

        Returns:
            List of validation error messages (empty if all valid)
        """
        expected_dims = await self.get_collection_dimensions()
        if expected_dims is None:
            return []  # Can't validate without expected dimensions

        errors = []
        for i, vec in enumerate(vectors):
            # Get vector values
            vector_values = vec.get("vector") or vec.get("values")
            if vector_values is None:
                errors.append(f"Vector {i} missing vector data")
                continue

            if len(vector_values) != expected_dims:
                errors.append(f"Vector {i} dimension mismatch: got {len(vector_values)}, expected {expected_dims}")

        return errors

    async def _get_vector_array(self, shape):
        """Get a reusable numpy array from the pool or create a new one"""
        async with self._vector_array_pool_lock:
            for i, (arr_shape, arr) in enumerate(self._vector_array_pool):
                if arr_shape == shape:
                    # Found a matching array, remove from pool and return
                    self._vector_array_pool.pop(i)
                    return arr

        # Create a new array if none found in pool
        return np.zeros(shape, dtype=np.float32)

    async def _return_vector_array(self, array):
        """Return a numpy array to the pool for reuse"""
        # Limit pool size to avoid excessive memory usage
        async with self._vector_array_pool_lock:
            if len(self._vector_array_pool) < 10:  # Limit pool size
                self._vector_array_pool.append((array.shape, array))

    async def upsert(
            self,
            vectors: List[Dict[str, Any]],
            batch_size: int = None,
            show_progress: bool = False,
            parallel_workers: int = 0,
            validate_dimensions: bool = True,
    ) -> Dict:
        """
        Vector insertion with parallel processing and dimension validation

        Args:
            vectors: List of vectors to upsert
            batch_size: Batch size (overrides default)
            show_progress: Show progress during operation
            parallel_workers: Number of parallel workers (0 = auto)
            validate_dimensions: Whether to validate vector dimensions

        Returns:
            Response with upsert count
        """
        if not vectors:
            return {"upserted_count": 0}

        # Validate dimensions if requested
        if validate_dimensions:
            validation_errors = await self._validate_batch_dimensions(vectors)
            if validation_errors:
                # Return first few errors
                max_errors = min(5, len(validation_errors))
                error_msg = "; ".join(validation_errors[:max_errors])
                if len(validation_errors) > max_errors:
                    error_msg += f" (and {len(validation_errors) - max_errors} more errors)"
                raise ValueError(f"Dimension validation failed: {error_msg}")

        # Auto-determine parallelism level if not specified
        if parallel_workers <= 0:
            parallel_workers = min(os.cpu_count() or 4, 8)

        batch_size = batch_size or self.batch_size
        total_upserted = 0

        # Optimize vectors data first using thread pool for CPU-bound work
        if self.debug:
            print(f"Preprocessing {len(vectors)} vectors with {parallel_workers} workers")

        # Create batches of vectors optimally sized for network transmission
        batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]

        # For very large batches, use parallel preprocessing
        if len(vectors) > 10000:
            # Define preprocessing function to run in thread pool
            def preprocess_batch(batch_vectors):
                # Convert to numpy arrays for numeric vectors if possible
                processed_batch = []
                for vec in batch_vectors:
                    # Handle vector values
                    vector_values = None
                    if "values" in vec:
                        vector_values = vec["values"]
                    elif "vector" in vec:
                        vector_values = vec["vector"]

                    # Format vector consistently
                    if vector_values is not None:
                        # Convert to numpy array if it's not already
                        if isinstance(vector_values, np.ndarray):
                            # Ensure correct data type
                            if vector_values.dtype != np.float32:
                                vector_values = vector_values.astype(np.float32)
                        else:
                            # Convert to numpy for faster processing
                            vector_values = np.array(vector_values, dtype=np.float32)

                        # Convert back to list for JSON serialization
                        vector_values = vector_values.tolist()

                    # Create new record with consistent format
                    record = {
                        "id": str(vec["id"]),
                        "vector": vector_values
                    }

                    # Add metadata if present
                    if "metadata" in vec and vec["metadata"] is not None:
                        record["metadata"] = vec["metadata"]

                    processed_batch.append(record)
                return processed_batch

            # Use thread pool for parallel preprocessing
            preprocessing_tasks = []
            for batch in batches:
                preprocessing_tasks.append(
                    self._thread_pool.submit(preprocess_batch, batch)
                )

            # Collect processed batches
            processed_batches = []
            for task in preprocessing_tasks:
                processed_batches.append(task.result())

            # Replace original batches with processed ones
            batches = processed_batches

        # Create async tasks for batch insertion with concurrency control
        semaphore = asyncio.Semaphore(parallel_workers)

        async def process_batch_with_semaphore(batch, batch_idx):
            """Process a batch with semaphore for concurrency control"""
            async with semaphore:
                try:
                    # Insert batch and return number of inserted items
                    result = await self.upsert_batch(batch)
                    return batch_idx, result.get("count", len(batch)), None
                except Exception as e:
                    if self.debug:
                        print(f"Error in batch {batch_idx}: {str(e)}")
                    return batch_idx, 0, e

        # Create tasks for all batches
        tasks = []
        for i, batch in enumerate(batches):
            task = asyncio.create_task(process_batch_with_semaphore(batch, i))
            tasks.append(task)

        # Process all batches with progress reporting if requested
        if show_progress:
            # Setup progress bar
            pbar = tqdm(total=len(vectors), desc="Upserting vectors")

            # Process batches and update progress
            for future in asyncio.as_completed(tasks):
                batch_idx, count, error = await future
                total_upserted += count
                pbar.update(len(batches[batch_idx]))
                pbar.set_postfix({"inserted": total_upserted})

                # If error occurred, display warning but continue
                if error:
                    pbar.write(f"Warning: Batch {batch_idx} failed: {str(error)}")

            pbar.close()
        else:
            # Process without progress bar
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for batch_idx, count, error in results:
                if error is None:
                    total_upserted += count

        if self.debug:
            print(f"Upsert complete: {total_upserted}/{len(vectors)} vectors inserted")

        return {"upserted_count": total_upserted}

    async def upsert_batch(self, batch: List[Dict[str, Any]]) -> Dict | None:
        """Batch vector insertion with optimized serialization and dimension validation"""
        # Format vectors consistently with the Go server's expectations
        formatted_vectors = []
        for vec in batch:
            # Get vector values
            vector_values = vec.get("vector", None)
            if vector_values is None and "values" in vec:
                vector_values = vec.get("values", None)

            if vector_values is None:
                raise ValueError(f"Vector missing 'values' or 'vector' field")

            # Convert numpy arrays to lists if needed
            if isinstance(vector_values, np.ndarray):
                vector_values = vector_values.tolist()

            # Ensure all values are Python floats (no numpy types for JSON compatibility)
            if any(not isinstance(v, (int, float)) for v in vector_values):
                vector_values = [float(v) for v in vector_values]

            # Format vector for Go server
            formatted_vec = {
                "id": str(vec["id"]),
                "vector": vector_values  # Use "vector" field name for Go server compatibility
            }

            # Add metadata if it exists
            if "metadata" in vec and vec["metadata"] is not None:
                formatted_vec["metadata"] = vec["metadata"]

            formatted_vectors.append(formatted_vec)

        # Create payload in the format expected by the Go server
        payload = {
            "collection": self.collection_name,
            "vectors": formatted_vectors
        }

        # Serialize with orjson for maximum performance
        try:
            payload_bytes = orjson.dumps(payload)
        except Exception as e:
            raise ValueError(f"Failed to serialize payload to JSON: {str(e)}")

        # Implement retry logic with exponential backoff
        max_retries = 3
        retry_delay = 1.0  # Starting delay in seconds

        for attempt in range(max_retries):
            try:
                # Use semaphore for concurrency control
                async with self._request_semaphore:
                    # Make request with optimized settings
                    url = f"{self.host}/vectors/upsert"

                    response = await self.client.post(
                        url,
                        content=payload_bytes,
                        headers={"Content-Type": "application/json"},
                        timeout=120.0  # Increased timeout for large batches
                    )

                    if response.status_code >= 400:
                        error_msg = f"API error: HTTP {response.status_code}"
                        try:
                            error_data = orjson.loads(response.content)
                            if "error" in error_data:
                                error_msg += f" - {error_data['error']}"
                        except:
                            error_msg += f" - {response.text[:100]}"

                        raise Exception(error_msg)

                    # Parse response with orjson for performance
                    return orjson.loads(response.content)

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    jitter = 0.1 * retry_delay * np.random.random()
                    wait_time = retry_delay * (2 ** attempt) + jitter

                    if self.debug:
                        print(f"Retrying after error: {e}, waiting {wait_time:.1f}s...")

                    await asyncio.sleep(wait_time)
                else:
                    # Last attempt failed
                    raise

    async def delete(self, ids: List[str]) -> Dict:
        """
        Delete vectors by ID with optimized performance

        Args:
            ids: List of vector IDs to delete

        Returns:
            Response with deletion results
        """
        if not ids:
            return {"deleted": [], "failed": []}

        payload = {
            "collection": self.collection_name,
            "ids": ids
        }

        # Serialize with orjson for performance
        payload_bytes = orjson.dumps(payload)

        try:
            # Use semaphore for concurrency control
            async with self._request_semaphore:
                response = await self.client.post(
                    f"{self.host}/vectors/delete",
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )

                if response.status_code >= 400:
                    raise Exception(f"API error: HTTP {response.status_code} - {response.text[:100]}")

                # Invalidate cache entries for these IDs if caching is enabled
                if self._query_cache is not None and self._cache_lock is not None:
                    # Simple approach: just clear the entire cache since we can't
                    # easily identify which cache entries might include these vectors
                    if len(ids) > 10:  # Only clear cache for bulk operations
                        async with self._cache_lock:
                            self._query_cache.clear()

                return orjson.loads(response.content)
        except Exception as e:
            if self.debug:
                print(f"Error deleting vectors: {e}")
            return {"deleted": [], "failed": ids}

    async def query(
            self,
            vector: Union[List[float], np.ndarray] = None,
            queries: List[Dict] = None,
            top_k: int = 10,
            include_values: bool = False,
            include_metadata: bool = True,
            filter: Dict = None,
            use_ann: bool = True,
            ef_search: int = 100,
            threshold: float = 0.0,
            skip_cache: bool = False,
            validate_dimensions: bool = True,
    ) -> Dict | None:
        """
        Vector similarity search with caching and dimension validation

        Args:
            vector: Query vector
            queries: Multiple queries (for batch search)
            top_k: Number of results to return
            include_values: Include vector values in results
            include_metadata: Include metadata in results
            filter: Metadata filter
            use_ann: Use approximate nearest neighbors (HNSW)
            ef_search: HNSW ef search parameter
            threshold: Similarity threshold
            skip_cache: Skip cache lookup even if caching is enabled
            validate_dimensions: Whether to validate query vector dimensions

        Returns:
            Search results
        """
        # Handle batch of queries
        if queries is not None:
            batch_results = await self.batch_query(
                queries=queries,
                top_k=top_k,
                include_values=include_values,
                include_metadata=include_metadata,
                filter=filter,
                use_ann=use_ann,
                ef_search=ef_search,
                threshold=threshold,
                skip_cache=skip_cache,
                validate_dimensions=validate_dimensions
            )
            return batch_results

        # Single vector query
        if vector is None:
            raise ValueError("Query vector is required")

        # Validate dimensions if requested
        if validate_dimensions:
            is_valid = await self.validate_vector_dimensions(vector)
            if not is_valid:
                expected_dims = await self.get_collection_dimensions()
                raise ValueError(
                    f"Query vector dimension mismatch: got {len(vector)}, expected {expected_dims} for collection '{self.collection_name}'")

        # Check cache if enabled and not skipped
        cache_key = None
        if self._query_cache is not None and self._compute_cache_key is not None and not skip_cache:
            cache_key = self._compute_cache_key(
                self.collection_name, vector, top_k,
                include_values=include_values,
                include_metadata=include_metadata,
                use_ann=use_ann,
                threshold=threshold,
                filter=str(filter) if filter else None  # Convert filter to string for caching
            )

            # Look up in cache
            async with self._cache_lock:
                if cache_key in self._query_cache:
                    self._cache_hits = getattr(self, '_cache_hits', 0) + 1
                    return self._query_cache[cache_key]
                self._cache_misses = getattr(self, '_cache_misses', 0) + 1

        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        # Ensure all values are native Python float type
        if any(not isinstance(v, (int, float)) for v in vector):
            vector = [float(v) for v in vector]

        # Build payload using template for efficiency
        payload = {
            "collection": self.collection_name,
            "vector": vector,
            "top_k": top_k,
            "include_vectors": include_values,
            "include_metadata": include_metadata,
            "use_ann": use_ann,
            "threshold": float(threshold)  # Ensure threshold is a native Python float
        }

        if filter:
            payload["filter"] = filter

        # Serialize with orjson for maximum performance
        payload_bytes = orjson.dumps(payload)

        # Make request with retry logic for important queries
        max_retries = 3
        retry_delay = 0.5  # Starting delay in seconds

        for attempt in range(max_retries):
            try:
                # Use semaphore for concurrency control
                async with self._request_semaphore:
                    # Execute query with timeout
                    response = await self.client.post(
                        f"{self.host}/vectors/query",
                        content=payload_bytes,
                        headers={"Content-Type": "application/json"},
                        timeout=60.0
                    )

                    if response.status_code >= 400:
                        raise Exception(f"API error: HTTP {response.status_code} - {response.text[:100]}")

                    # Parse with orjson for performance
                    api_results = orjson.loads(response.content)

                    # Format results to match expected return format
                    matches = []
                    for match in api_results.get("results", []):
                        match_item = {
                            "id": match["id"],
                            "score": match["score"],
                        }

                        if include_values and "vector" in match:
                            match_item["values"] = match["vector"]

                        if include_metadata and "metadata" in match:
                            match_item["metadata"] = match["metadata"]

                        matches.append(match_item)

                    result = {"matches": matches}

                    # Store in cache if enabled
                    if self._query_cache is not None and self._compute_cache_key is not None and not skip_cache:
                        async with self._cache_lock:
                            if cache_key is not None:
                                self._query_cache[cache_key] = result

                    return result

            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    jitter = 0.1 * retry_delay * np.random.random()
                    wait_time = retry_delay * (2 ** attempt) + jitter
                    await asyncio.sleep(wait_time)
                else:
                    # Last attempt failed
                    raise Exception(f"Query failed after {max_retries} attempts: {str(e)}")

    async def batch_query(
            self,
            vectors: List[Union[List[float], np.ndarray]] = None,
            queries: List[Dict] = None,
            top_k: int = 10,
            include_values: bool = False,
            include_metadata: bool = True,
            filter: Dict = None,
            use_ann: bool = True,
            ef_search: int = 100,
            threshold: float = 0.0,
            skip_cache: bool = False,
            validate_dimensions: bool = True,
    ) -> Dict | None:
        """
        Batch query for multiple vectors with dimension validation

        Args:
            vectors: List of query vectors
            queries: Multiple queries with parameters
            top_k: Number of results to return
            include_values: Include vector values in results
            include_metadata: Include metadata in results
            filter: Metadata filter
            use_ann: Use approximate nearest neighbors (HNSW)
            ef_search: HNSW ef search parameter
            threshold: Similarity threshold
            skip_cache: Skip cache lookup even if caching is enabled
            validate_dimensions: Whether to validate query vector dimensions

        Returns:
            Batch search results
        """
        # Convert queries to vector list if provided
        if queries and not vectors:
            vectors = [q.get("vector") for q in queries if "vector" in q]

            # Extract other parameters from first query
            if queries and len(queries) > 0:
                first_query = queries[0]
                top_k = first_query.get("top_k", top_k)
                include_values = first_query.get("include_values", include_values)
                include_metadata = first_query.get("include_metadata", include_metadata)
                filter = first_query.get("filter", filter)
                use_ann = first_query.get("use_ann", use_ann)
                ef_search = first_query.get("ef_search", ef_search)
                threshold = first_query.get("threshold", threshold)

        if not vectors:
            raise ValueError("Query vectors are required")

        # Validate dimensions if requested
        if validate_dimensions:
            expected_dims = await self.get_collection_dimensions()
            if expected_dims is not None:
                for i, vec in enumerate(vectors):
                    if len(vec) != expected_dims:
                        raise ValueError(
                            f"Query vector {i} dimension mismatch: got {len(vec)}, expected {expected_dims} for collection '{self.collection_name}'")

        # Process vectors in parallel using thread pool for CPU-bound conversions
        def process_vectors(vecs):
            processed = []
            for vec in vecs:
                # Convert numpy arrays to lists if needed
                if isinstance(vec, np.ndarray):
                    vec_list = vec.tolist()
                else:
                    vec_list = vec

                # Ensure all values are native Python float type
                processed.append([float(v) for v in vec_list])
            return processed

        # Use thread pool for parallel vector processing
        processed_vectors = await asyncio.get_event_loop().run_in_executor(
            self._thread_pool, process_vectors, vectors
        )

        # For very large batches, split them up to avoid overwhelming the server
        max_batch_size = 50
        if len(processed_vectors) > max_batch_size:
            results = {"results": []}

            # Process in smaller batches
            for i in range(0, len(processed_vectors), max_batch_size):
                batch = processed_vectors[i:i + max_batch_size]
                batch_result = await self.batch_query(
                    vectors=batch,
                    top_k=top_k,
                    include_values=include_values,
                    include_metadata=include_metadata,
                    filter=filter,
                    use_ann=use_ann,
                    ef_search=ef_search,
                    threshold=threshold,
                    skip_cache=skip_cache,
                    validate_dimensions=False  # Already validated above
                )

                if "results" in batch_result:
                    results["results"].extend(batch_result["results"])

            return results

        # Create batch query payload
        batch_queries = []
        for i, vec in enumerate(processed_vectors):
            query = {
                "vector": vec,
                "top_k": top_k,
                "include_values": include_values,
                "include_metadata": include_metadata,
                "use_ann": use_ann,
                "ef_search": ef_search,
                "threshold": threshold,
                "query_id": f"query_{i}"
            }
            if filter:
                query["filter"] = filter
            batch_queries.append(query)

        payload = {
            "collection": self.collection_name,
            "queries": batch_queries,
            "include_vectors": include_values,
            "include_metadata": include_metadata
        }

        # Serialize with orjson for performance
        payload_bytes = orjson.dumps(payload)

        # Make batch query request
        try:
            async with self._request_semaphore:
                response = await self.client.post(
                    f"{self.host}/vectors/batch_query",
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                    timeout=120.0
                )

                if response.status_code >= 400:
                    raise Exception(f"API error: HTTP {response.status_code} - {response.text[:100]}")

                # Parse response
                api_results = orjson.loads(response.content)

                # Format results to match expected structure
                formatted_results = []
                for query_result in api_results.get("results", []):
                    matches = []
                    for match in query_result.get("results", []):
                        match_item = {
                            "id": match["id"],
                            "score": match["score"],
                        }

                        if include_values and "vector" in match:
                            match_item["values"] = match["vector"]

                        if include_metadata and "metadata" in match:
                            match_item["metadata"] = match["metadata"]

                        matches.append(match_item)

                    formatted_results.append({"matches": matches})

                return {"results": formatted_results}

        except Exception as e:
            raise Exception(f"Batch query failed: {str(e)}")

    async def get_vector(self, vector_id: str) -> Dict | None:
        """
        Retrieve a specific vector by ID

        Args:
            vector_id: ID of the vector to retrieve

        Returns:
            Vector data or None if not found
        """
        try:
            async with self._request_semaphore:
                response = await self.client.get(
                    f"{self.host}/vectors/{vector_id}",
                    params={"collection": self.collection_name},
                    timeout=30.0
                )

                if response.status_code == 404:
                    return None
                elif response.status_code >= 400:
                    raise Exception(f"API error: HTTP {response.status_code} - {response.text[:100]}")

                return orjson.loads(response.content)

        except Exception as e:
            if self.debug:
                print(f"Error retrieving vector {vector_id}: {e}")
            return None

    async def count_vectors(self) -> int:
        """
        Get the count of vectors in this collection

        Returns:
            Number of vectors in the collection
        """
        try:
            # Get collection info which should include count
            response = await self.client.get(
                f"{self.host}/collections/{self.collection_name}",
                timeout=30.0
            )

            if response.status_code >= 400:
                raise Exception(f"API error: HTTP {response.status_code}")

            collection_info = orjson.loads(response.content)
            return collection_info.get("vector_count", 0)

        except Exception as e:
            if self.debug:
                print(f"Error getting vector count: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics

        Returns:
            Dictionary with cache statistics
        """
        if not hasattr(self, '_cache_hits'):
            return {"cache_enabled": False}

        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_enabled": True,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._query_cache) if self._query_cache else 0
        }

    async def clear_cache(self):
        """Clear the query cache for this collection"""
        if self._query_cache is not None and self._cache_lock is not None:
            async with self._cache_lock:
                # Clear only cache entries for this collection
                # Since we can't easily identify collection-specific keys,
                # we clear the entire cache for simplicity
                self._query_cache.clear()

                # Reset statistics
                self._cache_hits = 0
                self._cache_misses = 0

        return {"success": True, "message": f"Cache cleared for collection '{self.collection_name}'"}

    async def close(self):
        """Close vector operations and clean up resources"""
        # Shutdown thread pool if we own it
        if hasattr(self, '_owns_thread_pool') and self._owns_thread_pool:
            if hasattr(self, '_thread_pool'):
                self._thread_pool.shutdown(wait=True)

        # Clear caches
        if hasattr(self, '_query_cache') and self._query_cache:
            self._query_cache.clear()

        # Clean up vector array pool
        if hasattr(self, '_vector_array_pool'):
            self._vector_array_pool.clear()

    def __repr__(self):
        return f"VectorOperations(collection='{self.collection_name}', host='{self.host}')"

    def __str__(self):
        return f"VectorOperations for collection '{self.collection_name}'"
