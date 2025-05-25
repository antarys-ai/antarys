# Antarys Vector Database Python Client

Python client for Antarys vector database, optimized for large-scale vector operations with built-in caching, parallel
processing, and dimension validation.

## Installation

```bash
pip install antarys-client
```

Optional dependencies for accelerated performance:

```bash
pip install numba lz4
```

## Quick Start

```python
import asyncio
from antarys import Client


async def main():
    # Initialize client with performance optimizations
    client = Client(
        host="http://localhost:8080",
        connection_pool_size=100,  # Auto-sized based on CPU count
        use_http2=True,
        cache_size=1000,
        thread_pool_size=16
    )

    # Create collection
    await client.create_collection(
        name="my_vectors",
        dimensions=1536,
        enable_hnsw=True,
        shards=16
    )

    vectors = client.vector_operations("my_vectors")

    # Upsert vectors
    await vectors.upsert([
        {"id": "1", "values": [0.1] * 1536, "metadata": {"category": "A"}},
        {"id": "2", "values": [0.2] * 1536, "metadata": {"category": "B"}}
    ])

    # Query similar vectors
    results = await vectors.query(
        vector=[0.1] * 1536,
        top_k=10,
        include_metadata=True
    )

    await client.close()


asyncio.run(main())
```

## Core Concepts

### Collections

```python
# Create collection with optimized parameters
await client.create_collection(
    name="vectors",
    dimensions=1536,  # Required: vector dimensions
    enable_hnsw=True,  # Enable HNSW indexing for fast ANN
    shards=16,  # Parallel processing shards
    m=16,  # HNSW connectivity parameter
    ef_construction=200  # HNSW construction quality
)

# List collections
collections = await client.list_collections()

# Get collection info
info = await client.describe_collection("vectors")

# Delete collection
await client.delete_collection("vectors")
```

### Vector Operations

#### Single Vector Upsert

```python
vectors = client.vector_operations("my_collection")

data = [
    {
        "id": "1",
        "values": [0.1, 0.2, 0.3],  # Must match collection dimensions
        "metadata": {"category": "example", "timestamp": 1234567890}
    },
    {
        "id": "2",
        "values": [0.4, 0.5, 0.6],  # Must match collection dimensions
        "metadata": {"category": "example", "timestamp": 1234567891}
    }
]

# Upsert single vector
await vectors.upsert(data)
```

#### Batch Upsert For Large Scale Data

```python
# Upload multiple vectors in batches for large scale
batch = []
for i in range(1000):
    vector_record = {
        "id": f"vector_{i}",
        "vector": [random.random() for _ in range(1536)],  # Use "vector" key
        "metadata": {
            "category": f"category_{i % 5}",
            "timestamp": int(time.time()),
            "batch_id": 1
        }
    }
    batch.append(vector_record)

result = await vectors.upsert_batch(batch)
```

#### Vector Query

```python
# Single vector similarity search
results = await vectors.query(
    vector=[0.1] * 1536,
    top_k=10,
    include_values=False,  # Exclude vector values for faster response
    include_metadata=True,  # Include metadata in results
    filter={"category": "A"},  # Metadata filtering
    use_ann=True,  # Use approximate nearest neighbors (HNSW)
    threshold=0.7  # Minimum similarity filter (0.0 for all results)
)

for match in results["matches"]:
    print(f"ID: {match['id']}, Score: {match['score']}")
```

#### Batch Query

```python
# Multiple vector queries in parallel
query_vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]

batch_results = await vectors.batch_query(
    vectors=query_vectors,
    top_k=5,
    include_metadata=True,
    validate_dimensions=True
)

for i, result in enumerate(batch_results["results"]):
    print(f"Query {i}: {len(result['matches'])} matches")
```

#### Delete Vectors

```python
# Delete vectors by ID
await vectors.delete(["vector_1", "vector_2", "vector_3"])

# Get vector by ID
vector_data = await vectors.get_vector("vector_1")

# Count vectors in collection
count = await vectors.count_vectors()
```

## Performance Optimization

### Client Configuration

```python
client = Client(
    host="http://localhost:8080",

    # Connection Pool Optimization
    connection_pool_size=100,  # High concurrency (auto: CPU_COUNT * 5)
    timeout=120,  # Extended timeout for large operations

    # HTTP/2 and Compression
    use_http2=True,  # Enable HTTP/2 multiplexing
    compression=True,  # Enable response compression

    # Caching Configuration
    cache_size=1000,  # Client-side query cache
    cache_ttl=300,  # Cache TTL in seconds

    # Threading and Parallelism
    thread_pool_size=16,  # CPU-bound operations (auto: CPU_COUNT * 2)

    # Retry Configuration
    retry_attempts=5,  # Network resilience

    # Debug Mode
    debug=True  # Performance monitoring
)
```

### Batch Operation Tuning

```python
# Optimal batch upsert parameters
await vectors.upsert(
    vectors=large_dataset,
    batch_size=5000,  # Optimal for network efficiency
    parallel_workers=8,  # Match server capability
    validate_dimensions=True,  # Prevent dimension errors
    show_progress=True
)

# High-throughput query configuration
results = await vectors.query(
    vector=query_vector,
    top_k=100,
    include_values=False,  # Reduce response size
    include_metadata=True,
    use_ann=True,  # Fast approximate search
    ef_search=200,  # Higher quality (vs speed)
    skip_cache=False  # Leverage cache
)
```

### Server-Side Optimization

#### HNSW Index Parameters

```python
await client.create_collection(
    name="high_performance",
    dimensions=1536,
    enable_hnsw=True,

    # HNSW Tuning
    m=16,  # Connectivity (16-64 for high recall)
    ef_construction=200,  # Graph construction quality (200-800)
    shards=32,  # Parallel processing (match CPU cores)
)

# Query-time HNSW parameters
results = await vectors.query(
    vector=query_vector,
    ef_search=200,  # Search quality (100-800) | Higher means accuracy over speed and ram consumption 
    use_ann=True  # Enable HNSW acceleration
)
```

#### Memory and Resource Management

```python
# Force commit for persistence
await client.commit()

# Clear client-side caches
await client.clear_cache()
await vectors.clear_cache()

# Proper resource cleanup
await client.close()
```

## Advanced Features

### Dimension Validation

```python
# Automatic dimension validation
is_valid = await vectors.validate_vector_dimensions([0.1] * 1536)

# Get collection dimensions
dims = await vectors.get_collection_dimensions()
```

### Cache Performance Monitoring

```python
# Get cache statistics
stats = vectors.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Cache size: {stats['cache_size']}")
```

## Performance Benchmarks

### Recommended Settings by Scale

#### Small Scale (< 1M vectors)

```python
client = Client(
    connection_pool_size=20,
    cache_size=500,
    thread_pool_size=4
)

batch_size = 1000
parallel_workers = 2
```

#### Medium Scale (1M - 10M vectors)

```python
client = Client(
    connection_pool_size=50,
    cache_size=2000,
    thread_pool_size=8
)

batch_size = 3000
parallel_workers = 4
```

#### Large Scale (10M+ vectors)

```python
client = Client(
    connection_pool_size=100,
    cache_size=5000,
    thread_pool_size=16
)

batch_size = 5000
parallel_workers = 8
```

## Data Types

The client uses strongly typed interfaces:

```python
from antarys.types import VectorRecord, SearchResult, SearchParams

# Type-safe vector record
record: VectorRecord = {
    "id": "example",
    "values": [0.1, 0.2, 0.3],
    "metadata": {"key": "value"}
}

# Search parameters
params = SearchParams(
    vector=[0.1] * 1536,
    top_k=10,
    include_metadata=True,
    threshold=0.8
)
```

## Health Monitoring

```python
# Check server health
health = await client.health()

# Get server information
info = await client.info()

# Collection statistics
collection_info = await client.describe_collection("vectors")
print(f"Vector count: {collection_info.get('vector_count', 0)}")
```
