"""
Embedding operations for Antarys client
Provides high-level functions for generating embeddings using the Antarys embedding API
"""

import asyncio
import httpx
import orjson
from tqdm.asyncio import tqdm
from typing import List, Dict, Any, Optional, Union


class EmbeddingOperations:
    """High-level embedding operations interface"""

    def __init__(
            self,
            host: str,
            client: httpx.AsyncClient,
            request_semaphore: Optional[asyncio.Semaphore] = None,
            debug: bool = False,
    ):
        """
        Initialize embedding operations

        Args:
            host: API host address
            client: Shared HTTP client
            request_semaphore: Semaphore for limiting concurrent requests
            debug: Enable debug mode for additional logging
        """
        self.host = host.rstrip("/")
        self.client = client
        self.debug = debug

        if request_semaphore is None:
            import os
            cpu_count = os.cpu_count() or 4
            self._request_semaphore = asyncio.Semaphore(cpu_count * 5)
        else:
            self._request_semaphore = request_semaphore

    async def embed(
            self,
            texts: Union[str, List[str]],
            batch_size: int = 256,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)

        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing (default: 256)

        Returns:
            Single embedding vector if input is string, or list of embedding vectors if input is list

        Example:
            # Single text
            embedding = await embed_ops.embed("Hello, World!")

            # Multiple texts
            embeddings = await embed_ops.embed([
                "First document",
                "Second document",
                "Third document"
            ])
        """

        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        if not texts:
            raise ValueError("Input texts cannot be empty")

        payload = {
            "input": texts,
            "batchSize": batch_size
        }

        payload_bytes = orjson.dumps(payload)

        try:
            async with self._request_semaphore:
                response = await self.client.post(
                    f"{self.host}/embedding/generate",
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                    timeout=120.0
                )

                if response.status_code >= 400:
                    error_msg = f"Embedding generation failed: HTTP {response.status_code}"
                    try:
                        error_data = orjson.loads(response.content)
                        if "error" in error_data:
                            error_msg += f" - {error_data['error']}"
                    except:
                        error_msg += f" - {response.text[:100]}"
                    raise Exception(error_msg)

                result = orjson.loads(response.content)
                embeddings = result.get("embeddings", [])

                if self.debug:
                    model = result.get("model", "unknown")
                    dimensions = result.get("dimensions", "unknown")
                    print(f"Generated {len(embeddings)} embeddings using {model} (dim: {dimensions})")

                if single_input:
                    return embeddings[0] if embeddings else []

                return embeddings

        except httpx.RequestError as e:
            raise Exception(f"Request error: {str(e)}")

    async def embed_batch(
            self,
            texts: List[str],
            batch_size: int = 256,
            show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for a large batch of texts with progress tracking

        Args:
            texts: List of text strings
            batch_size: Batch size for API requests (default: 256)
            show_progress: Show progress bar during processing

        Returns:
            List of embedding vectors

        Example:
            embeddings = await embed_ops.embed_batch(
                texts=documents,
                batch_size=100,
                show_progress=True
            )
        """
        if not texts:
            return []

        if len(texts) <= batch_size:
            return await self.embed(texts, batch_size=batch_size)

        chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        all_embeddings = []

        if show_progress:
            pbar = tqdm(total=len(texts), desc="Generating embeddings")

            for chunk in chunks:
                chunk_embeddings = await self.embed(chunk, batch_size=batch_size)
                all_embeddings.extend(chunk_embeddings)
                pbar.update(len(chunk))

            pbar.close()
        else:
            for chunk in chunks:
                chunk_embeddings = await self.embed(chunk, batch_size=batch_size)
                all_embeddings.extend(chunk_embeddings)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query

        Convenience method that prefixes the query with "query: " for better retrieval results.
        Use this when generating embeddings for search queries.

        Args:
            query: Query text

        Returns:
            Embedding vector

        Example:
            query_embedding = await embed_ops.embed_query("What is machine learning?")
        """
        prefixed_query = f"query: {query}"
        return await self.embed(prefixed_query)

    async def embed_documents(
            self,
            documents: List[str],
            batch_size: int = 256,
            show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for documents/passages

        Prefixes each document with "passage: " for better retrieval results.
        Use this when generating embeddings for documents to be searched.

        Args:
            documents: List of document texts
            batch_size: Batch size for API requests
            show_progress: Show progress bar during processing

        Returns:
            List of embedding vectors

        Example:
            doc_embeddings = await embed_ops.embed_documents(
                documents=[
                    "Python is a programming language",
                    "JavaScript is used for web development"
                ],
                show_progress=True
            )
        """
        # Prefix each document with "passage: "
        prefixed_docs = [f"passage: {doc}" for doc in documents]
        return await self.embed_batch(prefixed_docs, batch_size=batch_size, show_progress=show_progress)

    async def embed_with_metadata(
            self,
            texts: List[str],
            batch_size: int = 256,
            show_progress: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings and return with metadata

        Returns embeddings along with model information and dimensions.

        Args:
            texts: List of text strings
            batch_size: Batch size for API requests
            show_progress: Show progress bar during processing

        Returns:
            List of dictionaries containing text, embedding, and metadata

        Example:
            results = await embed_ops.embed_with_metadata(["Hello", "World"])
            # Returns: [
            #   {"text": "Hello", "embedding": [...], "model": "...", "dimensions": 384},
            #   {"text": "World", "embedding": [...], "model": "...", "dimensions": 384}
            # ]
        """
        embeddings = await self.embed_batch(texts, batch_size=batch_size, show_progress=show_progress)

        payload = {"input": [texts[0]], "batchSize": batch_size}
        payload_bytes = orjson.dumps(payload)

        async with self._request_semaphore:
            response = await self.client.post(
                f"{self.host}/embedding/generate",
                content=payload_bytes,
                headers={"Content-Type": "application/json"},
                timeout=120.0
            )
            result = orjson.loads(response.content)
            model = result.get("model", "unknown")
            dimensions = result.get("dimensions", 0)

        return [
            {
                "text": text,
                "embedding": embedding,
                "model": model,
                "dimensions": dimensions,
            }
            for text, embedding in zip(texts, embeddings)
        ]

    async def similarity(
            self,
            text1: str,
            text2: str,
    ) -> float:
        """
        Calculate cosine similarity between two texts

        Generates embeddings for both texts and computes their cosine similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score (0 to 1)

        Example:
            score = await embed_ops.similarity(
                "machine learning",
                "artificial intelligence"
            )
        """
        import numpy as np

        embeddings = await self.embed([text1, text2])

        vec1 = np.array(embeddings[0], dtype=np.float32)
        vec2 = np.array(embeddings[1], dtype=np.float32)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def __repr__(self):
        return f"EmbeddingOperations(host='{self.host}')"

    def __str__(self):
        return f"EmbeddingOperations for {self.host}"
