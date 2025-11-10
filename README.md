## [Antarys](https://antarys.ai)

High-Performance Embeddable Vector Database. [WIP]

## Shards

Everything lives in shards:

```go
type ShardedCollectionData struct {
    Vectors      map[string][]float32  // Your actual vectors
    Metadata     map[string]any        // embedding metadata you give us
    HNSWGraph    *hnswGraph           // The search index

    // async channels
    pendingInserts   chan *pendingInsert
    hnswUpdateChan   chan *hnswUpdate
}
```

HNSW Graphs are responsible for creating connections

```go
type hnswGraph struct {
    Nodes      map[uint32]*hnswNode  // Map of node ID to actual node
    EntryPoint uint32                // Where to start searches
    IDToNodeID map[string]uint32     // Your ID -> our internal ID
}

type hnswNode struct {
    ID          string
    VectorID    string
    Connections map[int][]uint32      // level -> list of connected nodes
}
```

## Async Insertion

Here's where it gets interesting. When you insert a vector:

1. We immediately store it in the `Vectors` map
2. Return success to you right away
3. Separately queue an HNSW update

```go
func (shard *ShardedCollectionData) processInsertLockless(insert *pendingInsert) error {
    // Store the vector right away
    shard.Mutex.Lock()
    shard.Vectors[insert.id] = insert.vector
    shard.Metadata[insert.id] = insert.metadata
    shard.Mutex.Unlock()

    // Try to queue the HNSW update
    select {
    case shard.hnswUpdateChan <- &hnswUpdate{...}:
        // Great, it's queued
    default:
        // channel full
    }

    return nil
}
```

There's a worker constantly processing these updates:

```go
func (shard *ShardedCollectionData) asyncHNSWWorker() {
    updateBuffer := make([]*hnswUpdate, 0, 100)
    ticker := time.NewTicker(10 * time.Millisecond)

    for {
        select {
        case update := <-shard.hnswUpdateChan:
            updateBuffer = append(updateBuffer, update)

            // Process in batches of 100
            if len(updateBuffer) >= 100 {
                shard.processBatchHNSWUpdates(updateBuffer)
                updateBuffer = updateBuffer[:0]
            }

        case <-ticker.C:
            // Or every 10ms, whichever comes first
            if len(updateBuffer) > 0 {
                shard.processBatchHNSWUpdates(updateBuffer)
                updateBuffer = updateBuffer[:0]
            }
        }
    }
}
```

### Current Limitations

This approach has some issues with the channel being full, HNSW updates might get lost resulting in search quality loss when search operations kick in. Also a goroutine dump can be triggered if we are reading and writing too much information all at the same time.

## The Future: Contiguous Arrays

The current approach uses tons of pointers and maps. Every time we search, we're jumping around memory. Not great for performance.

The plan is to switch to structure-of-arrays - basically flatten everything into contiguous chunks of memory.

### Instead of This

```go
// Current: nodes scattered in memory
type hnswNode struct {
    ID          string
    Connections map[int][]uint32  // Pointer to another map
}

nodes := map[uint32]*hnswNode{  // Pointers everywhere
    1: &hnswNode{...},
    2: &hnswNode{...},
}
```

### We'll Do This

```go
type Graph struct {
    // All vectors in one big chunk
    Vectors     [][]float32    // [nodeID][dimension]

    // All edges flattened
    Edges       []uint32       // [1,2,5,3,7,9,...]
    EdgeStart   []uint32       // [0,3,6,...] - where each node's edges begin
    EdgeCount   []uint16       // [3,3,2,...] - how many edges each node has

    // Node info
    Levels      []uint8        // [2,1,3,...] - max level for each node
    EntryPoint  uint32
}
```

### Why This Is Better

**Cache Friendly**: Instead of chasing pointers, we do sequential reads through arrays.

```go
// Current: pointer hopping
for _, neighbor := range node.Connections[level] {
    neighborNode := graph.Nodes[neighbor]  // Cache miss
    distance := similarity(query, neighborNode.Vector)  // Another cache miss
}

// Future: array walking
for i := uint32(0); i < graph.NodeCount; i++ {
    vector := graph.Vectors[i]           // Sequential access
    distance := similarity(query, vector) // Cache hit
}
```

**SIMD Friendly**: Can process multiple vectors at once at compile time:

```go
// Process 8 vectors simultaneously
for i := 0; i < len(nodeIDs); i += 8 {
    batch := nodeIDs[i:i+8]
    similarities := dotProductBatch8(query, graph.Vectors, batch)
}
```

### Copy-on-Write for Consistency

To fix the consistency issues, we'll use immutable snapshots:

```go
type Database struct {
    currentSnapshot  *Graph    // Read from this
    mutableBuffer    *WriteBuffer // Write to this
}

// Reads are always consistent
func (db *Database) Search(query []float32) []Result {
    snapshot := atomic.LoadPointer(&db.currentSnapshot)
    return searchGraph(snapshot, query)
}

// Writes accumulate in buffer
func (db *Database) Insert(vector []float32) error {
    db.buffer.Add(vector)

    // Rebuild when buffer gets big
    if db.buffer.Size() > threshold {
        go db.rebuildSnapshot()
    }
}

// Atomic swap when ready
func (db *Database) rebuildSnapshot() {
    newGraph := merge(db.currentSnapshot, db.buffer)
    atomic.StorePointer(&db.currentSnapshot, newGraph)
    db.buffer.Clear()
}
```

## Memory Layout Example

Say you have 3 vectors with these connections:

- Node 0: connects to [1, 2]
- Node 1: connects to [0, 2]
- Node 2: connects to [0, 1]

**Current storage**:

```
Node0 -> {connections: map[0:[1,2]]} -> malloc'd somewhere
Node1 -> {connections: map[0:[0,2]]} -> malloc'd somewhere else
Node2 -> {connections: map[0:[0,1]]} -> malloc'd somewhere else
```

**Array storage**:

```
Vectors:   [v0_data, v1_data, v2_data]        // Sequential
Edges:     [1, 2, 0, 2, 0, 1]                // All edges flattened
EdgeStart: [0, 2, 4]                         // Node 0 starts at 0, Node 1 at 2, etc
EdgeCount: [2, 2, 2]                         // Each node has 2 edges
```

## Planned Features

The following features are in development:

- Contiguous Arrays Architecture
- Filtered search and payload indexing (HTTP API)
- Distributed architecture with coordination
- gRPC support for high-performance RPC
- Comprehensive observability and monitoring
- Advanced resource management
- Hybrid and multi-modal search capabilities
- Multi-tenancy and security features
- Enhanced query optimization
- Advanced collection management
- Backup and recovery improvements
