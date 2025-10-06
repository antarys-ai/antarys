# Antarys Vector Database

<div align="center">

<picture>
  <img src="./media/logo.jpg" alt="Antarys Logo" width="60%" />
</picture>

<h3>Blazingly Fast Vector Database for Everyone</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/antarys-ai/antarys)](https://github.com/antarys-ai/antarys/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Package](https://img.shields.io/pypi/v/antarys)](https://pypi.org/project/antarys/)

<h3>

[Documentation](https://docs.antarys.ai) | [Discord](https://discord.gg/cvcBA3CgwX) | [Twitter](https://x.com/antarys_ai)

</h3>

</div>

---

Antarys is a high-performance vector database engineered for production-scale AI applications. Built from the ground up
for speed, it delivers **2-4.5x faster write throughput** and **30-130x faster query performance** compared to leading
alternatives, while maintaining superior recall accuracy.

<div align="center">
  <video width="100%" src="./media/intro.mp4" controls></video>
</div>

## Performance Benchmarks

Benchmarked against leading vector databases using
the [OpenAI-compatible DBpedia Dataset](https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M) (1M
vectors, 1536 dimensions).

### Write Performance

| Database    | Throughput (vectors/sec) | Performance vs Antarys |
|-------------|--------------------------|------------------------|
| **Antarys** | **2,017**                | **Baseline**           |
| Chroma      | 1,234                    | 1.6x slower            |
| Qdrant      | 892                      | 2.3x slower            |
| Milvus      | 445                      | 4.5x slower            |

### Batch Operations

| Database    | Avg Batch Time (ms) | P99 Latency (ms) |
|-------------|---------------------|------------------|
| **Antarys** | **495.7**           | **570.3**        |
| Chroma      | 810.4               | 890.2            |
| Qdrant      | 1,121.6             | 1,456.8          |
| Milvus      | 2,247.3             | 3,102.5          |

### Query Performance

| Database    | Throughput (queries/sec) | Avg Query Time (ms) | P99 Latency (ms) |
|-------------|--------------------------|---------------------|------------------|
| **Antarys** | **602.4**                | **1.66**            | **6.9**          |
| Chroma      | 340.1                    | 2.94                | 14.2             |
| Qdrant      | 19.4                     | 51.47               | 186.3            |
| Milvus      | 4.5                      | 220.46              | 892.1            |

### Search Quality & Recall

| Database    | Recall@100 (%) | Standard Deviation |
|-------------|----------------|--------------------|
| **Antarys** | **98.47%**     | **0.0023**         |
| Chroma      | 97.12%         | 0.0034             |
| Qdrant      | 96.83%         | 0.0041             |
| Milvus      | 95.67%         | 0.0056             |

[View full benchmark repository →](https://github.com/antarys-ai/benchmark)

## Installation

### Download Antarys Database

Install Antarys with our one-line installer (macOS ARM and Linux x64):

```bash
curl -fsSL http://antarys.ai/start.sh | bash
```

### Install Python Client

```bash
pip install antarys
```

For accelerated performance, install optional dependencies:

```bash
pip install antarys[performance]
# or individually
pip install numba lz4
```

### Alternative: Node.js Client

```bash
npm install @antarys/client
```

[View Node.js documentation →](https://github.com/antarys-ai/antarys-node)

## Quick Start

```python
import asyncio
from antarys import Client


async def main():
    # Initialize client
    client = Client(host="http://localhost:8080")

    # Create collection
    await client.create_collection(
        name="my_vectors",
        dimensions=1536,
        enable_hnsw=True
    )

    vectors = client.vector_operations("my_vectors")

    # Upsert vectors
    await vectors.upsert([
        {
            "id": "doc1",
            "values": [0.1] * 1536,
            "metadata": {"category": "AI", "source": "research"}
        },
        {
            "id": "doc2",
            "values": [0.2] * 1536,
            "metadata": {"category": "ML", "source": "tutorial"}
        }
    ])

    # Query similar vectors
    results = await vectors.query(
        vector=[0.15] * 1536,
        top_k=5,
        include_metadata=True,
        filter={"category": "AI"}
    )

    for match in results["matches"]:
        print(f"ID: {match['id']}, Score: {match['score']:.4f}")

    await client.close()


asyncio.run(main())
```

## Core Features

### Collections

Create and manage vector collections with optimized parameters:

```python
# Create collection with HNSW indexing
await client.create_collection(
    name="documents",
    dimensions=1536,
    enable_hnsw=True,
    shards=16,
    m=16,
    ef_construction=200
)

# List all collections
collections = await client.list_collections()

# Get collection details
info = await client.describe_collection("documents")

# Delete collection
await client.delete_collection("documents")
```

### Built-in Text Embeddings

Generate embeddings without external API calls:

```python
# Simple embedding
embedding = await client.embed("Hello, World!")

# Batch embeddings
embeddings = await client.embed([
    "First document",
    "Second document",
    "Third document"
])

# Query-optimized embeddings
query_emb = await client.embed_query("What is artificial intelligence?")

# Document embeddings with progress
doc_embs = await client.embed_documents(
    documents=["Doc 1", "Doc 2", "Doc 3"],
    show_progress=True
)

# Text similarity comparison
score = await client.text_similarity(
    "machine learning",
    "artificial intelligence"
)
```

### Vector Operations

#### Upsert Vectors

```python
vectors = client.vector_operations("my_collection")

# Single vector upsert
await vectors.upsert([
    {
        "id": "vec1",
        "values": [0.1, 0.2, 0.3],
        "metadata": {"type": "document", "timestamp": 1234567890}
    }
])

# Batch upsert for large-scale operations
batch = []
for i in range(10000):
    batch.append({
        "id": f"vector_{i}",
        "values": [random.random() for _ in range(1536)],
        "metadata": {"category": f"cat_{i % 5}"}
    })

await vectors.upsert_batch(
    batch,
    batch_size=5000,
    parallel_workers=8,
    show_progress=True
)
```

#### Query Vectors

```python
# Semantic search with filters
results = await vectors.query(
    vector=[0.1] * 1536,
    top_k=10,
    include_metadata=True,
    filter={"category": "research"},
    threshold=0.7,
    use_ann=True
)

# Batch queries
query_vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
batch_results = await vectors.batch_query(
    vectors=query_vectors,
    top_k=5,
    include_metadata=True
)

# Get specific vector
vector_data = await vectors.get_vector("vec1")

# Count vectors
count = await vectors.count_vectors()
```

#### Delete Vectors

```python
# Delete by IDs
await vectors.delete(["vec1", "vec2", "vec3"])
```

## Performance Optimization

### Client Configuration

Configure the client for optimal performance based on your workload:

```python
client = Client(
    host="http://localhost:8080",

    # Connection pooling
    connection_pool_size=100,

    # HTTP/2 and compression
    use_http2=True,
    compression=True,

    # Client-side caching
    cache_size=1000,
    cache_ttl=300,

    # Threading
    thread_pool_size=16,

    # Reliability
    retry_attempts=5,
    timeout=120
)
```

### Scale-Based Recommendations

#### Small Scale (< 1M vectors)

```python
client = Client(
    connection_pool_size=20,
    cache_size=500,
    thread_pool_size=4
)

# Batch operations
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

# Batch operations
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

# Batch operations
batch_size = 5000
parallel_workers = 8
```

### HNSW Index Tuning

Optimize HNSW parameters for your accuracy/speed requirements:

```python
# Collection creation
await client.create_collection(
    name="optimized",
    dimensions=1536,
    enable_hnsw=True,
    m=16,  # Connectivity (16-64 for high recall)
    ef_construction=200,  # Construction quality (200-800)
    shards=32  # Parallel processing
)

# Query-time tuning
results = await vectors.query(
    vector=query_vector,
    ef_search=200,  # Search quality (100-800)
    use_ann=True  # Enable HNSW acceleration
)
```

## Advanced Features

### Dimension Validation

```python
# Validate vector dimensions
is_valid = await vectors.validate_vector_dimensions([0.1] * 1536)

# Get collection dimensions
dims = await vectors.get_collection_dimensions()
```

### Cache Management

```python
# Get cache statistics
stats = vectors.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")

# Clear caches
await client.clear_cache()
await vectors.clear_cache()
```

### Health Monitoring

```python
# Check server health
health = await client.health()

# Get server information
info = await client.info()

# Collection statistics
collection_info = await client.describe_collection("vectors")
print(f"Vector count: {collection_info.get('vector_count', 0)}")
```

## Type Safety

Antarys includes comprehensive type definitions:

```python
from antarys.types import VectorRecord, SearchResult, SearchParams

# Type-safe vector record
record: VectorRecord = {
    "id": "example",
    "values": [0.1, 0.2, 0.3],
    "metadata": {"key": "value"}
}

# Type-safe search parameters
params = SearchParams(
    vector=[0.1] * 1536,
    top_k=10,
    include_metadata=True,
    threshold=0.8
)
```

## Resources

- **[Full Documentation](https://docs.antarys.ai)** - Complete API reference and guides
- **[Performance Report](https://docs.antarys.ai/docs/python/performance)** - Detailed benchmark analysis
- **[Benchmark Repository](https://github.com/antarys-ai/benchmark)** - Reproduce performance tests
- **[Node.js Client](https://github.com/antarys-ai/antarys-node)** - TypeScript/JavaScript SDK

## License

Antarys is released under the [MIT License](LICENSE).

## Community

- **[Discord](https://discord.gg/cvcBA3CgwX)** - Get help and discuss features
- **[GitHub Issues](https://github.com/antarys-ai/antarys/issues)** - Report bugs or request features
- **[Twitter](https://x.com/antarys_ai)** - Follow for updates

---

<div align="center">

⭐ **Star this repo to help more developers discover Antarys!**

</div>