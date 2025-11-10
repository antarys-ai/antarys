package antarys

import (
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/klauspost/compress/zstd"
	"golang.org/x/sync/errgroup"
	"golang.org/x/sync/semaphore"
)

const (
	DefaultDimension           = 1536
	DefaultMaxResults          = 100
	DefaultSimilarityThreshold = 0.0
	DefaultCommitInterval      = 30 * time.Second
	DefaultDataDirName         = "data"
	DefaultCacheSize           = 5000
	DefaultQueryConcurrency    = 4
	DefaultHNSWM               = 16
	DefaultHNSWEfConstruction  = 200
	DefaultHNSWEfSearch        = 64
	MagicHeader                = "ANTARYS"
	FormatVersion              = 1
	DefaultShards              = 8
	DefaultBatchSize           = 1000
	MinAccuracyCandidates      = 64
	MaxHNSWLevels              = 12
)

type SearchMethod int

const (
	SearchMethodBruteForce SearchMethod = iota
	SearchMethodHNSW
	SearchMethodVLQADC
)

var (
	candidatePool = sync.Pool{
		New: func() interface{} {
			return make([]queueItem, 0, 512)
		},
	}

	searchWorkerPool *WorkerPool
	insertWorkerPool *WorkerPool
)

func GetOptimalVector(size int) []float32 {
	switch size {
	case 1536:
		return vectorPool1536.Get().([]float32)
	case 768:
		return vectorPool768.Get().([]float32)
	case 512:
		return vectorPool512.Get().([]float32)
	default:
		v := vectorPoolGeneric.Get().([]float32)
		if cap(v) < size {
			return make([]float32, size)
		}
		return v[:size]
	}
}

type LRUCache struct {
	capacity int
	cache    map[uint64]float32
	order    []uint64
	mutex    sync.RWMutex
}

func (c *LRUCache) Get(key uint64) (float32, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	value, exists := c.cache[key]
	return value, exists
}

func (c *LRUCache) Put(key uint64, value float32) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if len(c.cache) >= c.capacity {

		oldest := c.order[0]
		delete(c.cache, oldest)
		c.order = c.order[1:]
	}

	c.cache[key] = value
	c.order = append(c.order, key)
}

type WorkerPool struct {
	workers   chan chan func()
	workQueue chan func()
	quit      chan bool
	size      int
}

func NewWorkerPool(size int) *WorkerPool {
	pool := &WorkerPool{
		workers:   make(chan chan func(), size),
		workQueue: make(chan func(), 1000),
		quit:      make(chan bool),
		size:      size,
	}

	for i := 0; i < size; i++ {
		worker := &Worker{
			workerPool: pool.workers,
			jobChannel: make(chan func()),
			quit:       make(chan bool),
		}
		worker.start()
	}

	go pool.dispatch()
	return pool
}

type Worker struct {
	workerPool chan chan func()
	jobChannel chan func()
	quit       chan bool
}

func (w *Worker) start() {
	go func() {
		for {
			w.workerPool <- w.jobChannel
			select {
			case job := <-w.jobChannel:
				job()
			case <-w.quit:
				return
			}
		}
	}()
}

func (wp *WorkerPool) dispatch() {
	for {
		select {
		case job := <-wp.workQueue:
			worker := <-wp.workers
			worker <- job
		case <-wp.quit:
			return
		}
	}
}

func (wp *WorkerPool) Submit(job func()) {
	wp.workQueue <- job
}

type VectorRecord struct {
	ID         string    `json:"id"`
	Vector     []float32 `json:"vector"`
	Metadata   any       `json:"metadata,omitempty"`
	Created    time.Time `json:"created"`
	Updated    time.Time `json:"updated"`
	Collection string    `json:"collection"`
}

type SearchResult struct {
	Record   VectorRecord `json:"record"`
	Score    float32      `json:"score"`
	Distance float32      `json:"distance"`
}

type SearchOptions struct {
	MaxResults          int                `json:"max_results"`
	SimilarityThreshold float32            `json:"similarity_threshold"`
	MetadataFilter      func(any) bool     `json:"-"`
	SortByCreated       bool               `json:"sort_by_created"`
	SortByUpdated       bool               `json:"sort_by_updated"`
	Quantization        QuantizationMethod `json:"quantization"`
	UseANN              bool               `json:"use_ann"`
	EfSearch            int                `json:"ef_search"`
	BFSMode             bool               `json:"bfs_mode"`
	GreedyMode          bool               `json:"greedy_mode"`
	SearchContext       context.Context    `json:"-"`
}

type QuantizationMethod int

const (
	QuantizationNone       QuantizationMethod = 0
	QuantizationScalar8Bit QuantizationMethod = 1
	QuantizationPQ         QuantizationMethod = 2
	QuantizationOPQ        QuantizationMethod = 3
	QuantizationResidualPQ QuantizationMethod = 4
	DefaultQuantization                       = QuantizationNone
)

type IndexingOptions struct {
	UseHNSW              bool               `json:"use_hnsw"`
	M                    int                `json:"m"`
	N                    int                `json:"n"`
	EfConstruction       int                `json:"ef_construction"`
	Quantization         QuantizationMethod `json:"quantization"`
	NumShards            int                `json:"num_shards"`
	ParallelConstruction bool               `json:"parallel_construction"`
	EdgesPerCentroid     int                `json:"edges_per_centroid"`
	UsePQ                bool               `json:"use_pq"`
	NumPQSubspaces       int                `json:"num_pq_subspaces"`
	NumPQClusters        int                `json:"num_pq_clusters"`
	SubRegionPortion     float32            `json:"sub_region_portion"`
	VLQMaxResults        int                `json:"vlq_max_results"`
}

type Config struct {
	DataDir          string          `json:"data_dir"`
	CommitInterval   time.Duration   `json:"commit_interval"`
	CacheSize        int             `json:"cache_size"`
	QueryConcurrency int             `json:"query_concurrency"`
	IndexingOptions  IndexingOptions `json:"indexing_options"`
	EncryptionKey    []byte          `json:"-"`
	Compression      bool            `json:"compression"`
	EnableProfiling  bool            `json:"enable_profiling"`
}

type QueryRequest struct {
	Query    []float32
	Options  *SearchOptions
	ResultCh chan<- []SearchResult
	ErrorCh  chan<- error
}

type hnswNode struct {
	ID          string
	VectorID    string
	Connections map[int][]uint32
}

type MemoryoptimisedVector struct {
	Data       []float32
	Dimensions int
}

type QuantizedVector struct {
	Data       []byte
	Dimensions int
	Scale      float32
	Offset     float32
}

type ShardedCollectionData struct {
	Vectors      map[string][]float32
	Quantized    map[string]QuantizedVector
	Metadata     map[string]any
	Created      map[string]time.Time
	Updated      map[string]time.Time
	HNSWGraph    *hnswGraph
	Mutex        sync.RWMutex
	PQCentroids  [][]float32
	PQCodes      map[string][]byte
	NumSubspaces int

	pendingInserts   chan *pendingInsert
	hnswUpdateChan   chan *hnswUpdate
	batchInsertChan  chan []*BatchInsert
	commitChan       chan struct{}
	AsyncWorkerStop  chan struct{}
	AsyncWorkerGroup sync.WaitGroup
}

func (shard *ShardedCollectionData) initAsyncWorkers() {
	shard.pendingInserts = make(chan *pendingInsert, 10000)
	shard.hnswUpdateChan = make(chan *hnswUpdate, 10000)
	shard.batchInsertChan = make(chan []*BatchInsert, 1000)
	shard.commitChan = make(chan struct{}, 100)
	shard.AsyncWorkerStop = make(chan struct{})

	numWorkers := runtime.NumCPU()
	for i := 0; i < numWorkers; i++ {
		shard.AsyncWorkerGroup.Add(1)
		go shard.asyncInsertWorker()
	}

	shard.AsyncWorkerGroup.Add(1)
	go shard.asyncHNSWWorker()

	shard.AsyncWorkerGroup.Add(1)
	go shard.asyncBatchWorker()
}

func (shard *ShardedCollectionData) asyncInsertWorker() {
	defer shard.AsyncWorkerGroup.Done()

	for {
		select {
		case insert := <-shard.pendingInserts:
			err := shard.processInsertLockless(insert)
			if insert.done != nil {
				insert.done <- err
			}

		case <-shard.AsyncWorkerStop:
			return
		}
	}
}

func (shard *ShardedCollectionData) processInsertLockless(insert *pendingInsert) error {
	shard.Mutex.Lock()

	shard.Vectors[insert.id] = insert.vector
	shard.Metadata[insert.id] = insert.metadata
	shard.Created[insert.id] = insert.created
	shard.Updated[insert.id] = insert.updated

	shard.Mutex.Unlock()

	select {
	case shard.hnswUpdateChan <- &hnswUpdate{
		id:     insert.id,
		vector: insert.vector,
	}:
	default:

	}

	return nil
}

func (shard *ShardedCollectionData) asyncHNSWWorker() {
	defer shard.AsyncWorkerGroup.Done()

	updateBuffer := make([]*hnswUpdate, 0, 100)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case update := <-shard.hnswUpdateChan:
			updateBuffer = append(updateBuffer, update)

			if len(updateBuffer) >= 100 {
				shard.processBatchHNSWUpdates(updateBuffer)
				updateBuffer = updateBuffer[:0]
			}

		case <-ticker.C:

			if len(updateBuffer) > 0 {
				shard.processBatchHNSWUpdates(updateBuffer)
				updateBuffer = updateBuffer[:0]
			}

		case <-shard.AsyncWorkerStop:

			if len(updateBuffer) > 0 {
				shard.processBatchHNSWUpdates(updateBuffer)
			}
			return
		}
	}
}

func (shard *ShardedCollectionData) updateHNSWGraphFast(id string, vector []float32, nodeID uint32) {
	if shard.HNSWGraph == nil {
		return
	}

	graph := shard.HNSWGraph

	existingNodeID, exists := graph.IDToNodeID[id]

	if !exists {

		level := randomLevelFast(graph.M)

		node := &hnswNode{
			ID:          fmt.Sprintf("%d", nodeID),
			VectorID:    id,
			Connections: make(map[int][]uint32),
		}

		for l := 0; l <= level; l++ {
			node.Connections[l] = make([]uint32, 0, graph.M*2)
		}

		graph.Nodes[nodeID] = node
		graph.IDToNodeID[id] = nodeID

		if level > graph.MaxLevel || len(graph.Nodes) == 1 {
			atomic.StoreUint32(&graph.EntryPoint, nodeID)
			graph.MaxLevel = level
		}

		if len(graph.Nodes) > 1 {
			shard.insertNodeWithProperHNSW(nodeID, vector, level)
		}
	} else {

		if existingNodeID < uint32(len(graph.Nodes)) && graph.Nodes[existingNodeID] != nil {
			shard.rebuildNodeConnections(graph, existingNodeID, vector)
		}
	}
}

func (shard *ShardedCollectionData) insertNodeWithProperHNSW(nodeID uint32, vector []float32, targetLevel int) {
	graph := shard.HNSWGraph

	ep := atomic.LoadUint32(&graph.EntryPoint)
	if ep == 0 || graph.Nodes[ep] == nil {
		return
	}

	currentEntryPoints := []uint32{ep}

	for level := graph.MaxLevel; level > targetLevel; level-- {

		closest := shard.searchLayerProper(vector, currentEntryPoints, 1, level)
		if len(closest) > 0 {
			currentEntryPoints = []uint32{closest[0].nodeID}
		}
	}

	for level := min(targetLevel, graph.MaxLevel); level >= 0; level-- {

		efConstruction := max(graph.M*2, 200)
		candidates := shard.searchLayerProper(vector, currentEntryPoints, efConstruction, level)

		mLevel := graph.M
		if level == 0 {
			mLevel = graph.M * 2
		}

		selectedNeighbors := shard.selectNeighborsHeuristic(candidates, mLevel, vector, nodeID)

		shard.addBidirectionalConnections(nodeID, selectedNeighbors, level)

		for _, neighborID := range selectedNeighbors {
			shard.pruneConnectionsIfNeeded(neighborID, level, mLevel)
		}

		if len(selectedNeighbors) > 0 {
			currentEntryPoints = selectedNeighbors[:min(len(selectedNeighbors), 3)]
		}
	}
}

func (shard *ShardedCollectionData) searchLayerProper(query []float32, entryPoints []uint32, ef int, level int) []queueItem {
	visited := make(map[uint32]bool, ef*4)
	candidates := getCandidateSlice(ef * 2)
	defer putCandidateSlice(candidates)

	results := getCandidateSlice(ef)
	defer putCandidateSlice(results)

	for _, ep := range entryPoints {
		if ep == 0 || shard.HNSWGraph.Nodes[ep] == nil {
			continue
		}

		node := shard.HNSWGraph.Nodes[ep]
		if node.VectorID == "" {
			continue
		}

		epVector, exists := shard.Vectors[node.VectorID]
		if !exists {
			continue
		}

		similarity := cosineSimilarityForHNSW(query, epVector)
		if similarity < -0.5 {
			continue
		}

		candidate := queueItem{nodeID: ep, similarity: similarity}
		candidates = append(candidates, candidate)
		results = append(results, candidate)
		visited[ep] = true
	}

	for len(candidates) > 0 {

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].similarity > candidates[j].similarity
		})

		current := candidates[0]
		candidates = candidates[1:]

		if len(results) >= ef {
			sort.Slice(results, func(i, j int) bool {
				return results[i].similarity < results[j].similarity
			})
			worstResult := results[len(results)-1]

			if current.similarity < worstResult.similarity*0.98 {
				break
			}
		}

		currentNode := shard.HNSWGraph.Nodes[current.nodeID]
		if currentNode == nil {
			continue
		}

		connections, exists := currentNode.Connections[level]
		if !exists {
			continue
		}

		for _, neighborID := range connections {
			if visited[neighborID] {
				continue
			}

			visited[neighborID] = true

			neighborNode := shard.HNSWGraph.Nodes[neighborID]
			if neighborNode == nil || neighborNode.VectorID == "" {
				continue
			}

			neighborVector, exists := shard.Vectors[neighborNode.VectorID]
			if !exists {
				continue
			}

			similarity := cosineSimilarityForHNSW(query, neighborVector)
			if similarity < -0.5 {
				continue
			}

			candidate := queueItem{nodeID: neighborID, similarity: similarity}

			candidates = append(candidates, candidate)

			if len(results) < ef {
				results = append(results, candidate)
			} else {

				sort.Slice(results, func(i, j int) bool {
					return results[i].similarity < results[j].similarity
				})

				worstResult := results[len(results)-1]
				if similarity > worstResult.similarity {
					results[len(results)-1] = candidate
				}
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].similarity > results[j].similarity
	})

	return results
}

func (shard *ShardedCollectionData) selectNeighborsHeuristic(candidates []queueItem, m int, queryVector []float32, nodeID uint32) []uint32 {
	_ = queryVector
	_ = nodeID

	if len(candidates) == 0 {
		return []uint32{}
	}

	if len(candidates) <= m {

		result := make([]uint32, len(candidates))
		for i, candidate := range candidates {
			result[i] = candidate.nodeID
		}
		return result
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].similarity > candidates[j].similarity
	})

	selected := make([]uint32, 0, m)

	simpleCount := m / 2
	heuristicCount := m - simpleCount

	for i := 0; i < simpleCount && i < len(candidates); i++ {
		selected = append(selected, candidates[i].nodeID)
	}

	remaining := candidates[simpleCount:]
	heuristicSelected := 0

	for heuristicSelected < heuristicCount && len(remaining) > 0 {
		bestIdx := -1
		bestScore := float32(-2.0)

		for i, candidate := range remaining {
			candidateVector, exists := shard.Vectors[shard.HNSWGraph.Nodes[candidate.nodeID].VectorID]
			if !exists {
				continue
			}

			queryDist := 1.0 - candidate.similarity

			minDistToSelected := float32(2.0)
			diversityGood := true

			for _, selectedID := range selected {
				selectedVector, exists := shard.Vectors[shard.HNSWGraph.Nodes[selectedID].VectorID]
				if !exists {
					continue
				}

				selectedDist := 1.0 - cosineSimilarityForHNSW(candidateVector, selectedVector)
				if selectedDist < minDistToSelected {
					minDistToSelected = selectedDist
				}

				if selectedDist < queryDist*0.8 {
					diversityGood = false
				}
			}

			diversityBonus := float32(0.0)
			if diversityGood && len(selected) > 0 {

				diversityBonus = minDistToSelected * 0.1
			}

			combinedScore := candidate.similarity + diversityBonus

			if diversityGood && combinedScore > bestScore {
				bestScore = combinedScore
				bestIdx = i
			}
		}

		if bestIdx >= 0 {
			selected = append(selected, remaining[bestIdx].nodeID)
			heuristicSelected++

			remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
		} else {

			if len(remaining) > 0 {
				selected = append(selected, remaining[0].nodeID)
				heuristicSelected++
				remaining = remaining[1:]
			} else {
				break
			}
		}
	}

	for len(selected) < m && len(remaining) > 0 {
		selected = append(selected, remaining[0].nodeID)
		remaining = remaining[1:]
	}

	return selected
}

func (shard *ShardedCollectionData) addBidirectionalConnections(nodeID uint32, neighbors []uint32, level int) {
	node := shard.HNSWGraph.Nodes[nodeID]
	if node == nil {
		return
	}

	if node.Connections[level] == nil {
		node.Connections[level] = make([]uint32, 0, len(neighbors))
	}

	node.Connections[level] = neighbors

	for _, neighborID := range neighbors {
		neighborNode := shard.HNSWGraph.Nodes[neighborID]
		if neighborNode == nil {
			continue
		}

		if neighborNode.Connections[level] == nil {
			neighborNode.Connections[level] = make([]uint32, 0, shard.HNSWGraph.M*2)
		}

		exists := false
		for _, existingConn := range neighborNode.Connections[level] {
			if existingConn == nodeID {
				exists = true
				break
			}
		}

		if !exists {
			neighborNode.Connections[level] = append(neighborNode.Connections[level], nodeID)
		}
	}
}

func (shard *ShardedCollectionData) pruneConnectionsIfNeeded(nodeID uint32, level int, maxConnections int) {
	node := shard.HNSWGraph.Nodes[nodeID]
	if node == nil {
		return
	}

	connections := node.Connections[level]
	if len(connections) <= maxConnections {
		return
	}

	nodeVector, exists := shard.Vectors[node.VectorID]
	if !exists {
		return
	}

	candidates := make([]queueItem, 0, len(connections))
	for _, connID := range connections {
		connNode := shard.HNSWGraph.Nodes[connID]
		if connNode == nil {
			continue
		}

		connVector, exists := shard.Vectors[connNode.VectorID]
		if !exists {
			continue
		}

		similarity := cosineSimilarityForHNSW(nodeVector, connVector)
		candidates = append(candidates, queueItem{nodeID: connID, similarity: similarity})
	}

	selected := shard.selectNeighborsHeuristic(candidates, maxConnections, nodeVector, nodeID)

	node.Connections[level] = selected
}

func (shard *ShardedCollectionData) rebuildNodeConnections(graph *hnswGraph, nodeID uint32, vector []float32) {
	node := graph.Nodes[nodeID]
	if node == nil {
		return
	}

	maxLevel := 0
	for level := range node.Connections {
		if level > maxLevel {
			maxLevel = level
		}
	}

	for level := 0; level <= maxLevel; level++ {

		ep := atomic.LoadUint32(&graph.EntryPoint)
		if ep == 0 || graph.Nodes[ep] == nil {
			continue
		}

		currentEntryPoints := []uint32{ep}
		for searchLevel := graph.MaxLevel; searchLevel > level; searchLevel-- {
			if searchLevel <= maxLevel {
				closest := shard.searchLayerProper(vector, currentEntryPoints, 1, searchLevel)
				if len(closest) > 0 {
					currentEntryPoints = []uint32{closest[0].nodeID}
				}
			}
		}

		efConstruction := max(graph.M*2, 100)
		candidates := shard.searchLayerProper(vector, currentEntryPoints, efConstruction, level)

		filteredCandidates := make([]queueItem, 0, len(candidates))
		for _, candidate := range candidates {
			if candidate.nodeID != nodeID {
				filteredCandidates = append(filteredCandidates, candidate)
			}
		}

		mLevel := graph.M
		if level == 0 {
			mLevel = graph.M * 2
		}

		selectedNeighbors := shard.selectNeighborsHeuristic(filteredCandidates, mLevel, vector, nodeID)

		node.Connections[level] = selectedNeighbors

		shard.addBidirectionalConnections(nodeID, selectedNeighbors, level)
	}
}

func cosineSimilarityForHNSW(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return -1.0
	}

	result := cosineSimilarity(a, b)

	if math.IsNaN(float64(result)) || math.IsInf(float64(result), 0) {
		return -1.0
	}

	return max(-1.0, min(1.0, result))
}

func (shard *ShardedCollectionData) processBatchHNSWUpdates(updates []*hnswUpdate) {
	if shard.HNSWGraph == nil {
		return
	}

	shard.Mutex.Lock()
	defer shard.Mutex.Unlock()

	for _, update := range updates {

		shard.updateHNSWGraphFast(update.id, update.vector, update.nodeID)
	}
}

func (shard *ShardedCollectionData) asyncBatchWorker() {
	defer shard.AsyncWorkerGroup.Done()

	for {
		select {
		case batch := <-shard.batchInsertChan:
			shard.processBatchInsertLockless(batch)

		case <-shard.AsyncWorkerStop:
			return
		}
	}
}

func (shard *ShardedCollectionData) processBatchInsertLockless(batch []*BatchInsert) {
	if len(batch) == 0 {
		return
	}

	shard.Mutex.Lock()

	if len(shard.Vectors)+len(batch) > len(shard.Vectors)*2 {
		newSize := (len(shard.Vectors) + len(batch)) * 2
		newVectors := make(map[string][]float32, newSize)
		newMetadata := make(map[string]any, newSize)
		newCreated := make(map[string]time.Time, newSize)
		newUpdated := make(map[string]time.Time, newSize)

		for k, v := range shard.Vectors {
			newVectors[k] = v
		}
		for k, v := range shard.Metadata {
			newMetadata[k] = v
		}
		for k, v := range shard.Created {
			newCreated[k] = v
		}
		for k, v := range shard.Updated {
			newUpdated[k] = v
		}

		shard.Vectors = newVectors
		shard.Metadata = newMetadata
		shard.Created = newCreated
		shard.Updated = newUpdated
	}

	hnswUpdates := make([]*hnswUpdate, 0, len(batch))

	for _, insert := range batch {
		shard.Vectors[insert.Id] = insert.Vector
		shard.Metadata[insert.Id] = insert.Metadata
		shard.Created[insert.Id] = insert.Created
		shard.Updated[insert.Id] = insert.Updated

		hnswUpdates = append(hnswUpdates, &hnswUpdate{
			id:     insert.Id,
			vector: insert.Vector,
		})
	}

	shard.Mutex.Unlock()

	for _, update := range hnswUpdates {
		select {
		case shard.hnswUpdateChan <- update:
		default:

		}
	}
}

func (shard *ShardedCollectionData) StopAsyncWorkers() {
	close(shard.AsyncWorkerStop)
	shard.AsyncWorkerGroup.Wait()
}

type hnswGraph struct {
	Nodes      map[uint32]*hnswNode
	EntryPoint uint32
	MaxLevel   int
	M          int
	NextNodeID uint32
	IDToNodeID map[string]uint32
}

type pendingInsert struct {
	id       string
	vector   []float32
	metadata any
	created  time.Time
	updated  time.Time
	done     chan error
}

type hnswUpdate struct {
	id     string
	vector []float32
	nodeID uint32
}

type BatchInsert struct {
	Id       string
	Vector   []float32
	Metadata any
	Created  time.Time
	Updated  time.Time
}

type Collection struct {
	Name         string
	Dimensions   int `json:"dimensions"`
	VectorCount  int64
	Shards       []*ShardedCollectionData
	ShardMask    uint32
	Initialized  bool
	IndexOptions IndexingOptions
}

type AntarysDB struct {
	config         Config
	Collections    map[string]*Collection
	Mutex          sync.RWMutex
	dirty          atomic.Bool
	commitTicker   *time.Ticker
	stopCommit     chan struct{}
	queryQueue     chan QueryRequest
	workerPool     *errgroup.Group
	workerContext  context.Context
	workerCancel   context.CancelFunc
	resultCache    map[string][]SearchResult
	cacheKeys      []string
	cacheMutex     sync.RWMutex
	querySemaphore *semaphore.Weighted
	encoder        *zstd.Encoder
	decoder        *zstd.Decoder
	stats          *DBStats
}

type shardRecord struct {
	ID             string
	Vector         []float32
	Metadata       any
	NormalisedTime time.Time
}

type DBStats struct {
	Queries            int64
	CacheHits          int64
	InsertedVectors    int64
	TotalSearchTime    int64
	TotalInsertTime    int64
	LastCommitDuration time.Duration
	mutex              sync.RWMutex
}

func init() {
	numCPU := runtime.NumCPU()
	searchWorkerPool = NewWorkerPool(numCPU * 2)
	insertWorkerPool = NewWorkerPool(numCPU)
	detectCPUFeatures()
}

func NewAntarysDB(dataDir string) (*AntarysDB, error) {
	config := Config{
		DataDir:          dataDir,
		CommitInterval:   DefaultCommitInterval,
		CacheSize:        DefaultCacheSize,
		QueryConcurrency: DefaultQueryConcurrency,
		IndexingOptions: IndexingOptions{
			UseHNSW:              true,
			M:                    DefaultHNSWM,
			EfConstruction:       DefaultHNSWEfConstruction,
			Quantization:         DefaultQuantization,
			NumShards:            DefaultShards,
			ParallelConstruction: true,
		},
		Compression: true,
	}
	return NewAntarysDBWithConfig(config)
}

func (db *AntarysDB) StartAutoCommit(interval time.Duration) {
	db.commitTicker = time.NewTicker(interval)
	go func() {
		for {
			select {
			case <-db.commitTicker.C:
				if db.dirty.Load() {
					if err := db.Commit(); err != nil {
						fmt.Printf("Error in auto-commit: %v\n", err)
					}
				}
			case <-db.stopCommit:
				db.commitTicker.Stop()
				return
			}
		}
	}()
}

func (db *AntarysDB) Close() error {
	close(db.stopCommit)

	db.workerCancel()
	db.workerPool.Wait()

	if err := db.Commit(); err != nil {
		return err
	}

	if db.encoder != nil {
		db.encoder.Close()
	}
	if db.decoder != nil {
		db.decoder.Close()
	}

	return nil
}

func (db *AntarysDB) queryWorker(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return nil
		case req := <-db.queryQueue:
			results, err := db.performSearch(req.Query, req.Options)

			if err != nil {
				select {
				case req.ErrorCh <- err:
				default:

				}
			} else {
				select {
				case req.ResultCh <- results:
				default:

				}
			}
		}
	}
}

func (db *AntarysDB) performSearch(query []float32, opts *SearchOptions) ([]SearchResult, error) {
	startTime := time.Now()

	if opts == nil {
		opts = &SearchOptions{
			MaxResults:          DefaultMaxResults,
			SimilarityThreshold: DefaultSimilarityThreshold,
		}
	} else if opts.MaxResults <= 0 {
		opts.MaxResults = DefaultMaxResults
	}

	if opts.EfSearch <= 0 {
		opts.EfSearch = DefaultHNSWEfSearch
	}

	ctx := opts.SearchContext
	if ctx == nil {
		ctx = context.Background()
	}

	cacheKey := generateCacheKey(query, opts)

	db.cacheMutex.RLock()
	if cachedResults, ok := db.resultCache[cacheKey]; ok {
		db.cacheMutex.RUnlock()
		atomic.AddInt64(&db.stats.CacheHits, 1)
		return cachedResults, nil
	}
	db.cacheMutex.RUnlock()

	normalisedQuery := make([]float32, len(query))
	copy(normalisedQuery, query)
	NormaliseVector(normalisedQuery)

	var results []SearchResult

	db.Mutex.RLock()
	collections := make([]*Collection, 0, len(db.Collections))
	for _, coll := range db.Collections {
		collections = append(collections, coll)
	}
	db.Mutex.RUnlock()

	var wg sync.WaitGroup
	resultsCh := make(chan []SearchResult, len(collections))

	for _, coll := range collections {
		wg.Add(1)
		go func(c *Collection) {
			defer wg.Done()

			if ctx.Err() != nil {
				return
			}

			collResults, _ := db.SearchInCollection(c.Name, normalisedQuery, opts)
			resultsCh <- collResults
		}(coll)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-done:

		close(resultsCh)
	}

	for collResults := range resultsCh {
		results = append(results, collResults...)
	}

	sortResults(results, opts)

	if len(results) > opts.MaxResults {
		results = results[:opts.MaxResults]
	}

	db.cacheMutex.Lock()
	if len(db.cacheKeys) >= db.config.CacheSize {

		oldestKey := db.cacheKeys[0]
		delete(db.resultCache, oldestKey)
		db.cacheKeys = db.cacheKeys[1:]
	}
	db.resultCache[cacheKey] = results
	db.cacheKeys = append(db.cacheKeys, cacheKey)
	db.cacheMutex.Unlock()

	atomic.AddInt64(&db.stats.Queries, 1)
	atomic.AddInt64(&db.stats.TotalSearchTime, time.Since(startTime).Nanoseconds())

	return results, nil
}

func (db *AntarysDB) SearchInCollection(collectionName string, query []float32, opts *SearchOptions) ([]SearchResult, error) {
	if opts == nil {
		opts = &SearchOptions{
			MaxResults:          DefaultMaxResults,
			SimilarityThreshold: DefaultSimilarityThreshold,
		}
	}

	db.Mutex.RLock()
	coll, exists := db.Collections[collectionName]
	db.Mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("collection %s does not exist", collectionName)
	}

	normalisedQuery := getVector(len(query))
	defer putVector(normalisedQuery)
	copy(normalisedQuery, query)
	normaliseVector(normalisedQuery)

	totalCapacity := min(opts.MaxResults*len(coll.Shards), 1000)
	allResults := getResultSlice(totalCapacity)
	defer putResultSlice(allResults)

	resultsChan := make(chan []SearchResult, len(coll.Shards))
	var wg sync.WaitGroup

	for i, shard := range coll.Shards {
		wg.Add(1)
		searchWorkerPool.Submit(func(shardIdx int, s *ShardedCollectionData) func() {
			return func() {
				defer wg.Done()
				results := db.searchShard(s, normalisedQuery, opts)

				for i := range results {
					results[i].Record.Collection = collectionName
				}

				resultsChan <- results
			}
		}(i, shard))
	}

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	for shardResults := range resultsChan {
		allResults = append(allResults, shardResults...)
	}

	if len(allResults) > 1 {
		sortResultsFast(allResults)
	}

	if len(allResults) > opts.MaxResults {
		allResults = allResults[:opts.MaxResults]
	}

	finalResults := make([]SearchResult, len(allResults))
	copy(finalResults, allResults)

	return finalResults, nil
}

func (db *AntarysDB) searchShard(shard *ShardedCollectionData, query []float32, opts *SearchOptions) []SearchResult {
	shard.Mutex.RLock()
	vectorCount := len(shard.Vectors)
	useHNSW := opts.UseANN && shard.HNSWGraph != nil && vectorCount > 100
	shard.Mutex.RUnlock()

	if useHNSW {
		return db.hnswSearch(shard, query, opts)
	}

	return db.bruteForceSearch(shard, query, opts)
}

func (db *AntarysDB) hnswSearch(shard *ShardedCollectionData, query []float32, opts *SearchOptions) []SearchResult {
	shard.Mutex.RLock()
	graph := shard.HNSWGraph
	if graph == nil || len(graph.Nodes) == 0 {
		shard.Mutex.RUnlock()
		return db.bruteForceSearch(shard, query, opts)
	}

	vectors := shard.Vectors
	metadata := shard.Metadata
	created := shard.Created
	updated := shard.Updated
	shard.Mutex.RUnlock()

	ep := atomic.LoadUint32(&graph.EntryPoint)
	if ep == 0 || graph.Nodes[ep] == nil {
		return db.bruteForceSearch(shard, query, opts)
	}

	efSearch := opts.EfSearch
	if efSearch <= 0 {
		efSearch = max(opts.MaxResults*4, 100)
	}

	currentEP := ep
	for level := graph.MaxLevel; level > 0; level-- {
		closest := shard.searchLayerProper(query, []uint32{currentEP}, 1, level)
		if len(closest) > 0 {
			currentEP = closest[0].nodeID
		}
	}

	results := shard.searchLayerProper(query, []uint32{currentEP}, efSearch, 0)

	searchResults := make([]SearchResult, 0, len(results))
	for _, item := range results {
		if item.similarity < opts.SimilarityThreshold {
			continue
		}

		node := graph.Nodes[item.nodeID]
		if node == nil {
			continue
		}

		vectorID := node.VectorID
		vector, exists := vectors[vectorID]
		if !exists {
			continue
		}

		if opts.MetadataFilter != nil {
			if md, exists := metadata[vectorID]; exists && !opts.MetadataFilter(md) {
				continue
			}
		}

		searchResults = append(searchResults, SearchResult{
			Record: VectorRecord{
				ID:         vectorID,
				Vector:     vector,
				Metadata:   metadata[vectorID],
				Created:    created[vectorID],
				Updated:    updated[vectorID],
				Collection: "",
			},
			Score:    item.similarity,
			Distance: 1 - item.similarity,
		})
	}

	sort.Slice(searchResults, func(i, j int) bool {
		return searchResults[i].Score > searchResults[j].Score
	})

	if len(searchResults) > opts.MaxResults {
		searchResults = searchResults[:opts.MaxResults]
	}

	return searchResults
}

func (db *AntarysDB) searchLayerFast(graph *hnswGraph, vectors map[string][]float32, query []float32, ep uint32, ef int, level int, visited map[uint32]bool) uint32 {
	for k := range visited {
		delete(visited, k)
	}

	best := ep
	bestSim := cosineSimilarity(query, vectors[graph.Nodes[ep].VectorID])
	visited[ep] = true

	changed := true
	for changed {
		changed = false

		connections := graph.Nodes[best].Connections[level]
		for _, neighborID := range connections {
			if visited[neighborID] {
				continue
			}

			visited[neighborID] = true

			neighborNode := graph.Nodes[neighborID]
			if neighborNode == nil {
				continue
			}

			neighborSim := cosineSimilarity(query, vectors[neighborNode.VectorID])
			if neighborSim > bestSim {
				best = neighborID
				bestSim = neighborSim
				changed = true
			}
		}
	}

	return best
}

func (db *AntarysDB) bruteForceSearch(shard *ShardedCollectionData, query []float32, opts *SearchOptions) []SearchResult {
	shard.Mutex.RLock()
	vectors := shard.Vectors
	metadata := shard.Metadata
	created := shard.Created
	updated := shard.Updated
	shard.Mutex.RUnlock()

	results := getResultSlice(min(len(vectors), opts.MaxResults*2))
	defer putResultSlice(results)

	threshold := opts.SimilarityThreshold

	for id, vector := range vectors {

		if opts.MetadataFilter != nil {
			if md := metadata[id]; !opts.MetadataFilter(md) {
				continue
			}
		}

		similarity := cosineSimilarity(query, vector)

		if similarity < threshold {
			continue
		}

		results = append(results, SearchResult{
			Record: VectorRecord{
				ID:         id,
				Vector:     vector,
				Metadata:   metadata[id],
				Created:    created[id],
				Updated:    updated[id],
				Collection: "",
			},
			Score:    similarity,
			Distance: 1 - similarity,
		})

		if len(results) >= opts.MaxResults*3 {
			sortResultsFast(results)
			if len(results) > opts.MaxResults {
				results = results[:opts.MaxResults]
				threshold = results[len(results)-1].Score
			}
		}
	}

	finalResults := make([]SearchResult, len(results))
	copy(finalResults, results)
	return finalResults
}

func (db *AntarysDB) searchLayerAdvanced(shard *ShardedCollectionData, query []float32, ep uint32, ef int, level int, graph *hnswGraph) []queueItem {
	visited := make(map[uint32]bool, ef*2)
	candidates := priorityQueue{}
	results := priorityQueue{}

	epNode := graph.Nodes[ep]
	if epNode == nil {
		return []queueItem{}
	}

	epVector := shard.Vectors[epNode.VectorID]
	epSim := cosineSimilarity(query, epVector)

	candidates.push(queueItem{nodeID: ep, similarity: epSim})
	results.push(queueItem{nodeID: ep, similarity: epSim})
	visited[ep] = true

	exploredCount := 0
	maxExplored := ef * 2

	for candidates.len() > 0 && exploredCount < maxExplored {
		current := candidates.pop()
		exploredCount++

		if results.len() >= ef {
			worstInResults := results.bottom()
			if current.similarity < worstInResults.similarity*0.95 {
				continue
			}
		}

		connections, exists := graph.Nodes[current.nodeID].Connections[level]
		if !exists {
			continue
		}

		for _, neighborID := range connections {
			if visited[neighborID] {
				continue
			}

			visited[neighborID] = true

			neighborNode := graph.Nodes[neighborID]
			if neighborNode == nil {
				continue
			}

			neighborVector := shard.Vectors[neighborNode.VectorID]
			neighborSim := cosineSimilarity(query, neighborVector)

			if results.len() < ef {
				candidates.push(queueItem{nodeID: neighborID, similarity: neighborSim})
				results.push(queueItem{nodeID: neighborID, similarity: neighborSim})
			} else {
				worstInResults := results.bottom()
				if neighborSim > worstInResults.similarity {
					results.popBottom()
					results.push(queueItem{nodeID: neighborID, similarity: neighborSim})
					candidates.push(queueItem{nodeID: neighborID, similarity: neighborSim})
				}
			}
		}
	}

	result := make([]queueItem, 0, results.len())
	for results.len() > 0 {
		result = append(result, results.pop())
	}

	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result
}

func (db *AntarysDB) insertShardRecords(shard *ShardedCollectionData, records []shardRecord, indexOptions IndexingOptions) int64 {
	var newCount int64

	shard.Mutex.Lock()

	if shard.Vectors == nil {
		shard.Vectors = make(map[string][]float32, len(records))
		shard.Metadata = make(map[string]any, len(records))
		shard.Created = make(map[string]time.Time, len(records))
		shard.Updated = make(map[string]time.Time, len(records))
	}

	for _, rec := range records {
		isNew := false
		if _, exists := shard.Vectors[rec.ID]; !exists {
			isNew = true
			shard.Created[rec.ID] = rec.NormalisedTime
			newCount++
		}

		shard.Vectors[rec.ID] = rec.Vector
		shard.Metadata[rec.ID] = rec.Metadata
		shard.Updated[rec.ID] = rec.NormalisedTime

		if isNew && indexOptions.Quantization == QuantizationScalar8Bit {
			if shard.Quantized == nil {
				shard.Quantized = make(map[string]QuantizedVector)
			}
			shard.Quantized[rec.ID] = quantizeVector(rec.Vector)
		}
	}

	shard.Mutex.Unlock()

	if indexOptions.UseHNSW && shard.HNSWGraph != nil {
		db.updateHNSWGraphBatch(shard, records)
	}

	return newCount
}

func (db *AntarysDB) BatchInsertInCollection(collection string, records []struct {
	ID       string
	Vector   []float32
	Metadata any
}) error {
	if len(records) == 0 {
		return nil
	}

	startTime := time.Now()

	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collection)
	}

	numShards := len(coll.Shards)
	shardRecords := make([][]shardRecord, numShards)

	shardCounts := make([]int, numShards)
	for _, record := range records {
		shardIdx := getShardIndex(record.ID, uint32(numShards))
		shardCounts[shardIdx]++
	}

	for i := 0; i < numShards; i++ {
		if shardCounts[i] > 0 {
			shardRecords[i] = make([]shardRecord, 0, shardCounts[i])
		}
	}

	now := time.Now()
	for _, record := range records {
		shardIdx := getShardIndex(record.ID, uint32(numShards))

		vectorCopy := make([]float32, len(record.Vector))
		copy(vectorCopy, record.Vector)

		shardRecords[shardIdx] = append(shardRecords[shardIdx], shardRecord{
			ID:             record.ID,
			Vector:         vectorCopy,
			Metadata:       record.Metadata,
			NormalisedTime: now,
		})
	}

	var wg sync.WaitGroup
	var totalNewVectors int64

	for shardIdx, recs := range shardRecords {
		if len(recs) == 0 {
			continue
		}

		wg.Add(1)
		insertWorkerPool.Submit(func(idx int, records []shardRecord) func() {
			return func() {
				defer wg.Done()
				newCount := db.insertShardRecords(coll.Shards[idx], records, coll.IndexOptions)
				atomic.AddInt64(&totalNewVectors, newCount)
				atomic.AddInt64(&coll.VectorCount, newCount)
			}
		}(shardIdx, recs))
	}

	wg.Wait()

	db.dirty.Store(true)
	atomic.AddInt64(&db.stats.InsertedVectors, totalNewVectors)
	atomic.AddInt64(&db.stats.TotalInsertTime, time.Since(startTime).Nanoseconds())

	return nil
}

func (db *AntarysDB) InsertInCollection(collection string, id string, vector []float32, metadata any) error {
	startTime := time.Now()

	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collection)
	}

	if len(vector) != coll.Dimensions {
		return fmt.Errorf("vector dimension mismatch: got %d, expected %d",
			len(vector), coll.Dimensions)
	}

	shardIdx := getShardIndexFast(id, uint32(len(coll.Shards)))
	shard := coll.Shards[shardIdx]

	if shard.pendingInserts == nil {
		shard.initAsyncWorkers()
	}

	storedVector := make([]float32, len(vector))
	copy(storedVector, vector)

	done := make(chan error, 1)
	insert := &pendingInsert{
		id:       id,
		vector:   storedVector,
		metadata: metadata,
		created:  time.Now(),
		updated:  time.Now(),
		done:     done,
	}

	select {
	case shard.pendingInserts <- insert:

		atomic.AddInt64(&coll.VectorCount, 1)
		db.dirty.Store(true)
		atomic.AddInt64(&db.stats.InsertedVectors, 1)
		atomic.AddInt64(&db.stats.TotalInsertTime, time.Since(startTime).Nanoseconds())
		return nil

	default:

		return db.insertSynchronous(shard, id, vector, metadata, coll)
	}
}

func (db *AntarysDB) BatchInsert(collection string, batch []*BatchInsert) error {
	if len(batch) == 0 {
		return nil
	}

	startTime := time.Now()

	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collection)
	}

	numShards := len(coll.Shards)
	shardBatches := make(map[int][]*BatchInsert, numShards)

	for i := 0; i < numShards; i++ {
		shardBatches[i] = make([]*BatchInsert, 0, len(batch)/numShards+1)
	}

	for _, insert := range batch {
		shardIdx := getShardIndexFast(insert.Id, uint32(numShards))

		vectorCopy := make([]float32, len(insert.Vector))
		copy(vectorCopy, insert.Vector)

		insertCopy := &BatchInsert{
			Id:       insert.Id,
			Vector:   vectorCopy,
			Metadata: insert.Metadata,
			Created:  insert.Created,
			Updated:  insert.Updated,
		}

		shardBatches[shardIdx] = append(shardBatches[shardIdx], insertCopy)
	}

	var wg sync.WaitGroup
	var totalNewVectors int64

	for shardIdx, shardBatch := range shardBatches {
		if len(shardBatch) == 0 {
			continue
		}

		wg.Add(1)
		go func(idx int, batch []*BatchInsert) {
			defer wg.Done()

			shard := coll.Shards[idx]

			if shard.batchInsertChan == nil {
				shard.initAsyncWorkers()
			}

			select {
			case shard.batchInsertChan <- batch:
				atomic.AddInt64(&totalNewVectors, int64(len(batch)))
			default:

				shard.processBatchInsertLockless(batch)
				atomic.AddInt64(&totalNewVectors, int64(len(batch)))
			}
		}(shardIdx, shardBatch)
	}

	wg.Wait()

	atomic.AddInt64(&coll.VectorCount, totalNewVectors)
	db.dirty.Store(true)
	atomic.AddInt64(&db.stats.InsertedVectors, totalNewVectors)
	atomic.AddInt64(&db.stats.TotalInsertTime, time.Since(startTime).Nanoseconds())

	return nil
}

func getShardIndexFast(id string, numShards uint32) int {
	var hash uint32 = 2166136261
	for i := 0; i < len(id); i++ {
		hash ^= uint32(id[i])
		hash *= 16777619
	}
	return int(hash % numShards)
}

func (db *AntarysDB) insertSynchronous(shard *ShardedCollectionData, id string, vector []float32, metadata any, coll *Collection) error {
	storedVector := make([]float32, len(vector))
	copy(storedVector, vector)

	shard.Mutex.Lock()
	now := time.Now()

	isNew := true
	if _, exists := shard.Vectors[id]; exists {
		isNew = false
	}

	shard.Vectors[id] = storedVector
	shard.Metadata[id] = metadata
	shard.Updated[id] = now
	if isNew {
		shard.Created[id] = now
	}

	shard.Mutex.Unlock()

	if shard.hnswUpdateChan != nil {
		select {
		case shard.hnswUpdateChan <- &hnswUpdate{
			id:     id,
			vector: storedVector,
		}:
		default:

		}
	}

	if isNew {
		atomic.AddInt64(&coll.VectorCount, 1)
	}

	return nil
}

func (db *AntarysDB) updateHNSWGraphBatch(shard *ShardedCollectionData, records []shardRecord) {
	const batchSize = 50

	for i := 0; i < len(records); i += batchSize {
		end := i + batchSize
		if end > len(records) {
			end = len(records)
		}

		batch := records[i:end]

		shard.Mutex.Lock()
		for _, rec := range batch {
			db.updateHNSWGraph(shard, rec.ID, rec.Vector)
		}
		shard.Mutex.Unlock()
	}
}

func (db *AntarysDB) updateHNSWGraph(shard *ShardedCollectionData, id string, vector []float32) {
	if shard.HNSWGraph == nil {
		shard.HNSWGraph = &hnswGraph{
			Nodes:      make(map[uint32]*hnswNode),
			IDToNodeID: make(map[string]uint32),
			M:          16,
			MaxLevel:   0,
			NextNodeID: 1,
		}
	}

	graph := shard.HNSWGraph

	if nodeID, exists := graph.IDToNodeID[id]; exists {

		if node := graph.Nodes[nodeID]; node != nil {

			return
		}
	}

	nodeID := graph.NextNodeID
	graph.NextNodeID++

	level := randomLevel(graph.M)

	node := &hnswNode{
		ID:          fmt.Sprintf("%d", nodeID),
		VectorID:    id,
		Connections: make(map[int][]uint32),
	}

	for l := 0; l <= level; l++ {
		node.Connections[l] = make([]uint32, 0)
	}

	graph.Nodes[nodeID] = node
	graph.IDToNodeID[id] = nodeID

	if level > graph.MaxLevel || len(graph.Nodes) == 1 {
		graph.EntryPoint = nodeID
		graph.MaxLevel = level
		return
	}

	db.insertNode(shard, graph, nodeID, vector, level)
}

func (db *AntarysDB) insertNode(shard *ShardedCollectionData, graph *hnswGraph, nodeID uint32, vector []float32, targetLevel int) {
	ep := graph.EntryPoint

	for l := graph.MaxLevel; l > targetLevel; l-- {
		ep = db.searchLayerFast(graph, shard.Vectors, vector, ep, 1, l, make(map[uint32]bool))
	}

	for l := min(targetLevel, graph.MaxLevel); l >= 0; l-- {
		neighbors := db.findNeighbors(shard, graph, vector, ep, l, graph.M*2)
		selected := db.selectNeighbors(neighbors, graph.M)

		graph.Nodes[nodeID].Connections[l] = selected

		for _, nid := range selected {
			if neighbor := graph.Nodes[nid]; neighbor != nil {
				if neighbor.Connections[l] == nil {
					neighbor.Connections[l] = make([]uint32, 0)
				}

				found := false
				for _, existing := range neighbor.Connections[l] {
					if existing == nodeID {
						found = true
						break
					}
				}
				if !found {
					neighbor.Connections[l] = append(neighbor.Connections[l], nodeID)
				}
			}
		}

		ep = nodeID
	}
}

func (db *AntarysDB) findNeighbors(shard *ShardedCollectionData, graph *hnswGraph, vector []float32, ep uint32, level int, ef int) []queueItem {
	visited := make(map[uint32]bool, ef)
	candidates := make([]queueItem, 0, ef)

	if epNode := graph.Nodes[ep]; epNode != nil {
		epSim := cosineSimilarity(vector, shard.Vectors[epNode.VectorID])
		candidates = append(candidates, queueItem{nodeID: ep, similarity: epSim})
		visited[ep] = true
	}

	for i := 0; i < len(candidates) && len(candidates) < ef; i++ {
		current := candidates[i]

		if connections := graph.Nodes[current.nodeID].Connections[level]; connections != nil {
			for _, nid := range connections {
				if visited[nid] || len(candidates) >= ef {
					continue
				}

				visited[nid] = true
				if neighbor := graph.Nodes[nid]; neighbor != nil {
					sim := cosineSimilarity(vector, shard.Vectors[neighbor.VectorID])
					candidates = append(candidates, queueItem{nodeID: nid, similarity: sim})
				}
			}
		}
	}

	sortCandidates(candidates)
	return candidates
}

func (db *AntarysDB) selectNeighbors(candidates []queueItem, m int) []uint32 {
	if len(candidates) <= m {
		result := make([]uint32, len(candidates))
		for i, c := range candidates {
			result[i] = c.nodeID
		}
		return result
	}

	result := make([]uint32, m)
	for i := 0; i < m; i++ {
		result[i] = candidates[i].nodeID
	}
	return result
}

func (db *AntarysDB) Delete(id string) error {
	return db.DeleteFromCollection("default", id)
}

func (db *AntarysDB) DeleteFromCollection(collection string, id string) error {
	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collection)
	}

	shardIdx := getShardIndex(id, uint32(len(coll.Shards)))
	shard := coll.Shards[shardIdx]

	shard.Mutex.Lock()
	defer shard.Mutex.Unlock()

	if _, exists := shard.Vectors[id]; !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	delete(shard.Vectors, id)
	delete(shard.Metadata, id)
	delete(shard.Created, id)
	delete(shard.Updated, id)

	if shard.Quantized != nil {
		delete(shard.Quantized, id)
	}

	if shard.HNSWGraph != nil {
		if nodeID, exists := shard.HNSWGraph.IDToNodeID[id]; exists {

			for _, node := range shard.HNSWGraph.Nodes {
				for level, connections := range node.Connections {
					newConns := make([]uint32, 0, len(connections))
					for _, conn := range connections {
						if conn != nodeID {
							newConns = append(newConns, conn)
						}
					}
					node.Connections[level] = newConns
				}
			}

			delete(shard.HNSWGraph.Nodes, nodeID)
			delete(shard.HNSWGraph.IDToNodeID, id)

			if shard.HNSWGraph.EntryPoint == nodeID {
				if len(shard.HNSWGraph.Nodes) > 0 {
					for nid := range shard.HNSWGraph.Nodes {
						shard.HNSWGraph.EntryPoint = nid
						break
					}
				} else {
					shard.HNSWGraph.EntryPoint = 0
				}
			}
		}
	}

	atomic.AddInt64(&coll.VectorCount, -1)

	db.dirty.Store(true)

	return nil
}

func (db *AntarysDB) Get(id string) (VectorRecord, error) {
	return db.GetFromCollection("default", id)
}

func (db *AntarysDB) GetFromCollection(collection, id string) (VectorRecord, error) {
	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return VectorRecord{}, fmt.Errorf("collection %s does not exist", collection)
	}

	shardIdx := getShardIndex(id, uint32(len(coll.Shards)))
	shard := coll.Shards[shardIdx]

	shard.Mutex.RLock()
	defer shard.Mutex.RUnlock()

	vector, exists := shard.Vectors[id]
	if !exists {
		return VectorRecord{}, fmt.Errorf("vector with ID %s not found", id)
	}

	return VectorRecord{
		ID:         id,
		Vector:     vector,
		Metadata:   shard.Metadata[id],
		Created:    shard.Created[id],
		Updated:    shard.Updated[id],
		Collection: collection,
	}, nil
}

func (db *AntarysDB) Count() int64 {
	return db.CountInCollection("default")
}

func (db *AntarysDB) CountInCollection(collection string) int64 {
	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return 0
	}

	return atomic.LoadInt64(&coll.VectorCount)
}

func (db *AntarysDB) ListCollections() []string {
	db.Mutex.RLock()
	defer db.Mutex.RUnlock()

	collections := make([]string, 0, len(db.Collections))
	for name := range db.Collections {
		collections = append(collections, name)
	}

	return collections
}

func (db *AntarysDB) CreateCollection(name string, dimensions int) error {
	return db.CreateCollectionWithOptions(name, dimensions, db.config.IndexingOptions)
}

func (db *AntarysDB) CreateCollectionWithOptions(name string, dimensions int, options IndexingOptions) error {
	if dimensions <= 0 {
		return fmt.Errorf("dimensions must be positive, got %d", dimensions)
	}

	db.Mutex.Lock()
	defer db.Mutex.Unlock()

	if _, exists := db.Collections[name]; exists {
		return fmt.Errorf("collection %s already exists", name)
	}

	numShards := options.NumShards
	if numShards <= 0 {
		numShards = DefaultShards
	}

	shards := make([]*ShardedCollectionData, numShards)
	for i := 0; i < numShards; i++ {
		shards[i] = &ShardedCollectionData{
			Vectors:  make(map[string][]float32),
			Metadata: make(map[string]any),
			Created:  make(map[string]time.Time),
			Updated:  make(map[string]time.Time),
		}

		if options.UseHNSW {
			shards[i].HNSWGraph = &hnswGraph{
				Nodes:      make(map[uint32]*hnswNode),
				IDToNodeID: make(map[string]uint32),
				M:          options.M,
				MaxLevel:   0,
				NextNodeID: 1,
			}
			fmt.Printf("Initialized HNSW graph for shard %d with M=%d\n", i, options.M)
		}
	}

	db.Collections[name] = &Collection{
		Name:         name,
		Dimensions:   dimensions,
		VectorCount:  0,
		Shards:       shards,
		ShardMask:    uint32(numShards - 1),
		Initialized:  true,
		IndexOptions: options,
	}

	db.dirty.Store(true)

	return nil
}

func (db *AntarysDB) DeleteCollection(name string) error {
	db.Mutex.Lock()
	defer db.Mutex.Unlock()

	if name == "default" {
		return fmt.Errorf("cannot delete default collection")
	}

	if _, exists := db.Collections[name]; !exists {
		return fmt.Errorf("collection %s does not exist", name)
	}

	delete(db.Collections, name)

	db.dirty.Store(true)

	return nil
}

func (db *AntarysDB) SearchByMetadata(filter func(any) bool, opts *SearchOptions) ([]VectorRecord, error) {
	return db.SearchByMetadataInCollection("default", filter, opts)
}

func (db *AntarysDB) SearchByMetadataInCollection(collection string, filter func(any) bool, opts *SearchOptions) ([]VectorRecord, error) {
	if opts == nil {
		opts = &SearchOptions{
			MaxResults:          DefaultMaxResults,
			SimilarityThreshold: 0.0,
		}
	}

	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("collection %s does not exist", collection)
	}

	var results []VectorRecord

	var wg sync.WaitGroup
	resultsCh := make(chan []VectorRecord, len(coll.Shards))

	for _, shard := range coll.Shards {
		wg.Add(1)
		go func(s *ShardedCollectionData) {
			defer wg.Done()

			var shardResults []VectorRecord

			s.Mutex.RLock()
			for id, metadata := range s.Metadata {
				if filter(metadata) {
					shardResults = append(shardResults, VectorRecord{
						ID:         id,
						Vector:     s.Vectors[id],
						Metadata:   metadata,
						Created:    s.Created[id],
						Updated:    s.Updated[id],
						Collection: collection,
					})
				}
			}
			s.Mutex.RUnlock()

			resultsCh <- shardResults
		}(shard)
	}

	wg.Wait()
	close(resultsCh)

	for shardResults := range resultsCh {
		results = append(results, shardResults...)
	}

	if opts != nil {
		if opts.SortByCreated {
			sort.Slice(results, func(i, j int) bool {
				return results[i].Created.After(results[j].Created)
			})
		} else if opts.SortByUpdated {
			sort.Slice(results, func(i, j int) bool {
				return results[i].Updated.After(results[j].Updated)
			})
		}

		if opts.MaxResults > 0 && len(results) > opts.MaxResults {
			results = results[:opts.MaxResults]
		}
	}

	return results, nil
}

func (db *AntarysDB) Commit() error {
	if !db.dirty.Load() {
		return nil
	}

	startTime := time.Now()
	log.Printf("Starting commit to disk...")

	if err := os.MkdirAll(db.config.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %w", err)
	}

	type metadata struct {
		FormatVersion   int             `json:"format_version"`
		Collections     []string        `json:"collections"`
		Created         time.Time       `json:"created"`
		LastCommit      time.Time       `json:"last_commit"`
		IndexingOptions IndexingOptions `json:"indexing_options"`
		Compression     bool            `json:"compression"`
	}

	db.Mutex.RLock()
	collections := make([]string, 0, len(db.Collections))
	for name := range db.Collections {
		collections = append(collections, name)
	}

	for name, coll := range db.Collections {
		if err := db.saveCollectionData(name, coll); err != nil {
			db.Mutex.RUnlock()
			return fmt.Errorf("failed to save collection %s: %w", name, err)
		}
	}
	db.Mutex.RUnlock()

	meta := metadata{
		FormatVersion:   FormatVersion,
		Collections:     collections,
		Created:         time.Now(),
		LastCommit:      time.Now(),
		IndexingOptions: db.config.IndexingOptions,
		Compression:     db.config.Compression,
	}

	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	metaPath := filepath.Join(db.config.DataDir, "metadata.json")
	if err := os.WriteFile(metaPath, metaBytes, 0644); err != nil {
		return fmt.Errorf("failed to write metadata: %w", err)
	}

	db.dirty.Store(false)
	log.Printf("Commit completed in %v - saved %d collections", time.Since(startTime), len(collections))
	return nil
}

func (db *AntarysDB) saveCollectionData(name string, coll *Collection) error {
	collDir := filepath.Join(db.config.DataDir, name)
	if err := os.MkdirAll(collDir, 0755); err != nil {
		return fmt.Errorf("failed to create collection directory: %w", err)
	}

	type collectionMeta struct {
		Name         string          `json:"name"`
		Dimensions   int             `json:"dimensions"`
		VectorCount  int64           `json:"vector_count"`
		ShardCount   int             `json:"shard_count"`
		Created      time.Time       `json:"created"`
		LastModified time.Time       `json:"last_modified"`
		IndexOptions IndexingOptions `json:"index_options"`
	}

	meta := collectionMeta{
		Name:         name,
		Dimensions:   coll.Dimensions,
		VectorCount:  atomic.LoadInt64(&coll.VectorCount),
		ShardCount:   len(coll.Shards),
		Created:      time.Now(),
		LastModified: time.Now(),
		IndexOptions: coll.IndexOptions,
	}

	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal collection metadata: %w", err)
	}

	metaPath := filepath.Join(collDir, "collection.json")
	if err := os.WriteFile(metaPath, metaBytes, 0644); err != nil {
		return fmt.Errorf("failed to write collection metadata: %w", err)
	}

	for i, shard := range coll.Shards {
		if err := db.saveShardData(collDir, i, shard); err != nil {
			return fmt.Errorf("failed to save shard %d: %w", i, err)
		}
	}

	return nil
}

func (db *AntarysDB) saveShardData(collDir string, shardIdx int, shard *ShardedCollectionData) error {
	shardDir := filepath.Join(collDir, fmt.Sprintf("shard_%d", shardIdx))
	if err := os.MkdirAll(shardDir, 0755); err != nil {
		return fmt.Errorf("failed to create shard directory: %w", err)
	}

	shard.Mutex.RLock()
	defer shard.Mutex.RUnlock()

	type vectorData struct {
		ID       string    `json:"id"`
		Vector   []float32 `json:"vector"`
		Metadata any       `json:"metadata,omitempty"`
		Created  time.Time `json:"created"`
		Updated  time.Time `json:"updated"`
	}

	vectors := make([]vectorData, 0, len(shard.Vectors))
	for id, vector := range shard.Vectors {
		vectors = append(vectors, vectorData{
			ID:       id,
			Vector:   vector,
			Metadata: shard.Metadata[id],
			Created:  shard.Created[id],
			Updated:  shard.Updated[id],
		})
	}

	vectorsBytes, err := json.MarshalIndent(vectors, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal vectors: %w", err)
	}

	vectorsPath := filepath.Join(shardDir, "vectors.json")
	if err := os.WriteFile(vectorsPath, vectorsBytes, 0644); err != nil {
		return fmt.Errorf("failed to write vectors: %w", err)
	}

	log.Printf("Saved shard %d with %d vectors", shardIdx, len(vectors))
	return nil
}

func (db *AntarysDB) decrypt(data []byte) ([]byte, error) {
	if db.config.EncryptionKey == nil {
		return data, nil
	}

	block, err := aes.NewCipher(db.config.EncryptionKey)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	nonceSize := gcm.NonceSize()
	if len(data) < nonceSize {
		return nil, errors.New("ciphertext too short")
	}

	nonce, ciphertext := data[:nonceSize], data[nonceSize:]

	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

func (db *AntarysDB) Load() error {
	db.Mutex.Lock()
	defer db.Mutex.Unlock()

	metaPath := filepath.Join(db.config.DataDir, "metadata.json")
	if _, err := os.Stat(metaPath); os.IsNotExist(err) {
		log.Printf("No existing database found, creating new\n")
		return nil
	}

	metaBytes, err := os.ReadFile(metaPath)
	if err != nil {
		return fmt.Errorf("failed to read metadata: %w", err)
	}

	type metadata struct {
		FormatVersion   int             `json:"format_version"`
		Collections     []string        `json:"collections"`
		Created         time.Time       `json:"created"`
		LastCommit      time.Time       `json:"last_commit"`
		IndexingOptions IndexingOptions `json:"indexing_options"`
		Compression     bool            `json:"compression"`
	}

	var meta metadata
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return fmt.Errorf("failed to decode metadata: %w", err)
	}

	if meta.FormatVersion != FormatVersion {
		return fmt.Errorf("incompatible format version: got %d, expected %d",
			meta.FormatVersion, FormatVersion)
	}

	for _, name := range meta.Collections {
		coll, err := db.loadCollection(name)
		if err != nil {
			log.Printf("Warning: failed to load collection %s: %v\n", name, err)
			continue
		}
		db.Collections[name] = coll
		log.Printf("Loaded collection %s with %d vectors\n", name, atomic.LoadInt64(&coll.VectorCount))
	}

	log.Printf("Successfully loaded database with %d collections\n", len(db.Collections))
	return nil
}

func (db *AntarysDB) loadCollection(name string) (*Collection, error) {
	collDir := filepath.Join(db.config.DataDir, name)
	if _, err := os.Stat(collDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("collection directory doesn't exist")
	}

	metaPath := filepath.Join(collDir, "collection.json")
	metaBytes, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read collection metadata: %w", err)
	}

	if db.config.EncryptionKey != nil {
		metaBytes, err = db.decrypt(metaBytes)
		if err != nil {
			return nil, fmt.Errorf("failed to decrypt collection metadata: %w", err)
		}
	}

	type collectionMeta struct {
		Name         string          `json:"name"`
		Dimensions   int             `json:"dimensions"`
		VectorCount  int64           `json:"vector_count"`
		ShardCount   int             `json:"shard_count"`
		Created      time.Time       `json:"created"`
		LastModified time.Time       `json:"last_modified"`
		IndexOptions IndexingOptions `json:"index_options"`
	}

	var meta collectionMeta
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return nil, fmt.Errorf("failed to unmarshal collection metadata: %w", err)
	}

	if meta.Dimensions <= 0 {
		return nil, fmt.Errorf("invalid dimensions %d for collection %s", meta.Dimensions, name)
	}

	coll := &Collection{
		Name:         name,
		Dimensions:   meta.Dimensions,
		VectorCount:  meta.VectorCount,
		Shards:       make([]*ShardedCollectionData, meta.ShardCount),
		ShardMask:    uint32(meta.ShardCount - 1),
		Initialized:  true,
		IndexOptions: meta.IndexOptions,
	}

	for i := 0; i < meta.ShardCount; i++ {
		shard, err := db.loadShard(collDir, i, meta.Dimensions)
		if err != nil {
			return nil, fmt.Errorf("failed to load shard %d: %w", i, err)
		}

		coll.Shards[i] = shard
	}

	return coll, nil
}

func (db *AntarysDB) loadShard(collDir string, shardIdx int, dimensions int) (*ShardedCollectionData, error) {
	shardDir := filepath.Join(collDir, fmt.Sprintf("shard_%d", shardIdx))

	shard := &ShardedCollectionData{
		Vectors:  make(map[string][]float32),
		Metadata: make(map[string]any),
		Created:  make(map[string]time.Time),
		Updated:  make(map[string]time.Time),
	}

	vectorsPath := filepath.Join(shardDir, "vectors.json")
	if _, err := os.Stat(vectorsPath); err != nil {
		return shard, nil
	}

	vectorsBytes, err := os.ReadFile(vectorsPath)
	if err != nil {
		log.Printf("Warning: failed to read shard %d vectors: %v", shardIdx, err)
		return shard, nil
	}

	type vectorData struct {
		ID       string    `json:"id"`
		Vector   []float32 `json:"vector"`
		Metadata any       `json:"metadata,omitempty"`
		Created  time.Time `json:"created"`
		Updated  time.Time `json:"updated"`
	}

	var vectors []vectorData
	if err := json.Unmarshal(vectorsBytes, &vectors); err != nil {
		log.Printf("Warning: failed to unmarshal shard %d vectors: %v", shardIdx, err)
		return shard, nil
	}

	for _, vec := range vectors {
		shard.Vectors[vec.ID] = vec.Vector
		shard.Metadata[vec.ID] = vec.Metadata
		shard.Created[vec.ID] = vec.Created
		shard.Updated[vec.ID] = vec.Updated
	}

	log.Printf("Loaded shard %d with %d vectors", shardIdx, len(vectors))
	return shard, nil
}

func (db *AntarysDB) UpdateMetadata(collection, id string, metadata any) error {
	db.Mutex.RLock()
	coll, exists := db.Collections[collection]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collection)
	}

	shardIdx := getShardIndex(id, uint32(len(coll.Shards)))
	shard := coll.Shards[shardIdx]

	shard.Mutex.Lock()
	defer shard.Mutex.Unlock()

	if _, exists := shard.Vectors[id]; !exists {
		return fmt.Errorf("vector with ID %s not found", id)
	}

	shard.Metadata[id] = metadata
	shard.Updated[id] = time.Now()

	db.dirty.Store(true)

	return nil
}

func (db *AntarysDB) RebuildIndex(collectionName string, options IndexingOptions) error {
	db.Mutex.Lock()
	defer db.Mutex.Unlock()

	coll, exists := db.Collections[collectionName]
	if !exists {
		return fmt.Errorf("collection %s does not exist", collectionName)
	}

	coll.IndexOptions = options

	var wg sync.WaitGroup
	errChan := make(chan error, len(coll.Shards))

	for i, shard := range coll.Shards {
		wg.Add(1)
		go func(shardIdx int, s *ShardedCollectionData) {
			defer wg.Done()

			s.Mutex.Lock()
			defer s.Mutex.Unlock()

			if options.UseHNSW {
				s.HNSWGraph = &hnswGraph{
					Nodes:      make(map[uint32]*hnswNode),
					IDToNodeID: make(map[string]uint32),
					M:          options.M,
					MaxLevel:   0,
					NextNodeID: 1,
				}

				for id, vector := range s.Vectors {
					db.updateHNSWGraph(s, id, vector)
				}
			} else {
				s.HNSWGraph = nil
			}

			if options.Quantization == QuantizationScalar8Bit {
				if s.Quantized == nil {
					s.Quantized = make(map[string]QuantizedVector)
				}

				for id, vector := range s.Vectors {
					s.Quantized[id] = quantizeVector(vector)
				}
			} else {
				s.Quantized = nil
			}
		}(i, shard)
	}

	wg.Wait()

	select {
	case err := <-errChan:
		return err
	default:

	}

	db.dirty.Store(true)

	return nil
}

func (db *AntarysDB) GetStats() map[string]interface{} {
	db.stats.mutex.RLock()
	defer db.stats.mutex.RUnlock()

	return map[string]interface{}{
		"queries":              atomic.LoadInt64(&db.stats.Queries),
		"cache_hits":           atomic.LoadInt64(&db.stats.CacheHits),
		"inserted_vectors":     atomic.LoadInt64(&db.stats.InsertedVectors),
		"avg_search_time_ms":   float64(atomic.LoadInt64(&db.stats.TotalSearchTime)) / float64(max(1, atomic.LoadInt64(&db.stats.Queries))) / 1e6,
		"avg_insert_time_ms":   float64(atomic.LoadInt64(&db.stats.TotalInsertTime)) / float64(max(1, atomic.LoadInt64(&db.stats.InsertedVectors))) / 1e6,
		"last_commit_duration": db.stats.LastCommitDuration.String(),
		"collection_counts":    db.getCollectionCounts(),
	}
}

func (db *AntarysDB) getCollectionCounts() map[string]int64 {
	db.Mutex.RLock()
	defer db.Mutex.RUnlock()

	counts := make(map[string]int64)
	for name, coll := range db.Collections {
		counts[name] = atomic.LoadInt64(&coll.VectorCount)
	}

	return counts
}

func (db *AntarysDB) ListIDs(collectionName string) ([]string, error) {
	db.Mutex.RLock()
	coll, exists := db.Collections[collectionName]
	db.Mutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("collection %s does not exist", collectionName)
	}

	var ids []string

	for _, shard := range coll.Shards {
		shard.Mutex.RLock()
		for id := range shard.Vectors {
			ids = append(ids, id)
		}
		shard.Mutex.RUnlock()
	}

	return ids, nil
}

func (db *AntarysDB) PurgeCache() {
	db.cacheMutex.Lock()
	defer db.cacheMutex.Unlock()

	db.resultCache = make(map[string][]SearchResult)
	db.cacheKeys = make([]string, 0, db.config.CacheSize)
}

func (db *AntarysDB) Dump(outputDir string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	if err := db.Commit(); err != nil {
		return fmt.Errorf("failed to commit changes before dump: %w", err)
	}

	return filepath.Walk(db.config.DataDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if strings.Contains(path, "backups") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, err := filepath.Rel(db.config.DataDir, path)
		if err != nil {
			return err
		}

		destPath := filepath.Join(outputDir, relPath)

		if info.IsDir() {

			return os.MkdirAll(destPath, info.Mode())
		} else {

			return copyFile(path, destPath)
		}
	})
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}

	return out.Sync()
}

func (db *AntarysDB) Restore(inputDir string) error {
	if err := db.Close(); err != nil {
		return fmt.Errorf("failed to close database: %w", err)
	}

	if err := os.RemoveAll(db.config.DataDir); err != nil {
		return fmt.Errorf("failed to clear data directory: %w", err)
	}

	if err := os.MkdirAll(db.config.DataDir, 0755); err != nil {
		return fmt.Errorf("failed to create data directory: %w", err)
	}

	err := filepath.Walk(inputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(inputDir, path)
		if err != nil {
			return err
		}

		destPath := filepath.Join(db.config.DataDir, relPath)

		if info.IsDir() {

			return os.MkdirAll(destPath, info.Mode())
		} else {

			return copyFile(path, destPath)
		}
	})

	if err != nil {
		return fmt.Errorf("failed to restore files: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	db.Collections = make(map[string]*Collection)
	db.workerContext = ctx
	db.workerCancel = cancel
	db.queryQueue = make(chan QueryRequest, 1000)
	db.resultCache = make(map[string][]SearchResult)
	db.cacheKeys = make([]string, 0, db.config.CacheSize)

	db.workerPool = &errgroup.Group{}
	for i := 0; i < db.config.QueryConcurrency; i++ {
		db.workerPool.Go(func() error {
			return db.queryWorker(ctx)
		})
	}

	if err := db.Load(); err != nil {
		return fmt.Errorf("failed to load restored data: %w", err)
	}

	db.StartAutoCommit(db.config.CommitInterval)

	return nil
}

func NewAntarysDBWithConfig(config Config) (*AntarysDB, error) {
	if config.DataDir == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get home directory: %w", err)
		}
		config.DataDir = filepath.Join(homeDir, DefaultDataDirName)
	}

	if err := os.MkdirAll(config.DataDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	var encoder *zstd.Encoder
	var decoder *zstd.Decoder
	var encoderErr error
	var decoderErr error

	if config.Compression {
		encoder, encoderErr = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBestCompression))
		if encoderErr != nil {
			return nil, fmt.Errorf("failed to initialize compression encoder: %w", encoderErr)
		}

		decoder, decoderErr = zstd.NewReader(nil)
		if decoderErr != nil {
			encoder.Close()
			return nil, fmt.Errorf("failed to initialize compression decoder: %w", decoderErr)
		}
	}

	if config.QueryConcurrency <= 0 {
		config.QueryConcurrency = DefaultQueryConcurrency
	}

	ctx, cancel := context.WithCancel(context.Background())

	db := &AntarysDB{
		config:         config,
		Collections:    make(map[string]*Collection),
		stopCommit:     make(chan struct{}),
		queryQueue:     make(chan QueryRequest, 1000),
		workerContext:  ctx,
		workerCancel:   cancel,
		resultCache:    make(map[string][]SearchResult),
		cacheKeys:      make([]string, 0, config.CacheSize),
		querySemaphore: semaphore.NewWeighted(int64(config.QueryConcurrency)),
		encoder:        encoder,
		decoder:        decoder,
		stats:          &DBStats{},
	}

	db.workerPool = &errgroup.Group{}
	for i := 0; i < config.QueryConcurrency; i++ {
		db.workerPool.Go(func() error {
			return db.queryWorker(ctx)
		})
	}

	if err := db.Load(); err != nil {

		fmt.Printf("Warning: couldn't load existing database: %v (creating new)\n", err)

		if err := db.CreateCollection("default", DefaultDimension); err != nil {
			return nil, fmt.Errorf("failed to create default collection: %w", err)
		}
	}

	db.StartAutoCommit(config.CommitInterval)

	return db, nil
}

func (db *AntarysDB) LoadFromJSON(collectionName string, inputFile string) error {

	data, err := os.ReadFile(inputFile)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	var records []struct {
		ID       string                 `json:"id"`
		Vector   []float32              `json:"vector"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := json.Unmarshal(data, &records); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}

	db.Mutex.RLock()
	coll, exists := db.Collections[collectionName]
	db.Mutex.RUnlock()

	if !exists {

		if err := db.CreateCollection(collectionName, coll.Dimensions); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
	}

	batch := make([]struct {
		ID       string
		Vector   []float32
		Metadata any
	}, len(records))

	for i, record := range records {
		batch[i].ID = record.ID
		batch[i].Vector = record.Vector
		batch[i].Metadata = record.Metadata
	}

	return db.BatchInsertInCollection(collectionName, batch)
}

func (db *AntarysDB) ExportToJSON(collectionName string, outputFile string) error {

	ids, err := db.ListIDs(collectionName)
	if err != nil {
		return fmt.Errorf("failed to list IDs: %w", err)
	}

	type outputRecord struct {
		ID       string                 `json:"id"`
		Vector   []float32              `json:"vector"`
		Metadata map[string]interface{} `json:"metadata"`
		Created  time.Time              `json:"created"`
		Updated  time.Time              `json:"updated"`
	}

	records := make([]outputRecord, 0, len(ids))

	for _, id := range ids {
		vector, err := db.GetFromCollection(collectionName, id)
		if err != nil {
			continue
		}

		metadataMap := make(map[string]interface{})
		if metadata, ok := vector.Metadata.(map[string]interface{}); ok {
			metadataMap = metadata
		} else {

			metadataBytes, err := json.Marshal(vector.Metadata)
			if err == nil {
				json.Unmarshal(metadataBytes, &metadataMap)
			}
		}

		records = append(records, outputRecord{
			ID:       vector.ID,
			Vector:   vector.Vector,
			Metadata: metadataMap,
			Created:  vector.Created,
			Updated:  vector.Updated,
		})
	}

	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to encode JSON: %w", err)
	}

	if err := os.WriteFile(outputFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

func sortResults(results []SearchResult, opts *SearchOptions) {
	if opts.SortByCreated {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Record.Created.After(results[j].Record.Created)
		})
	} else if opts.SortByUpdated {
		sort.Slice(results, func(i, j int) bool {
			return results[i].Record.Updated.After(results[j].Record.Updated)
		})
	} else {

		sort.Slice(results, func(i, j int) bool {
			return results[i].Score > results[j].Score
		})
	}
}

func GetSIMDWidth() int {
	cpuInit.Do(detectCPUFeatures)
	return simdWidth
}

func quantizeVector(vector []float32) QuantizedVector {
	if len(vector) == 0 {
		return QuantizedVector{}
	}

	min, max := vector[0], vector[0]

	i := 1
	for i+7 < len(vector) {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]

		if v0 < min {
			min = v0
		}
		if v1 < min {
			min = v1
		}
		if v2 < min {
			min = v2
		}
		if v3 < min {
			min = v3
		}
		if v4 < min {
			min = v4
		}
		if v5 < min {
			min = v5
		}
		if v6 < min {
			min = v6
		}
		if v7 < min {
			min = v7
		}

		if v0 > max {
			max = v0
		}
		if v1 > max {
			max = v1
		}
		if v2 > max {
			max = v2
		}
		if v3 > max {
			max = v3
		}
		if v4 > max {
			max = v4
		}
		if v5 > max {
			max = v5
		}
		if v6 > max {
			max = v6
		}
		if v7 > max {
			max = v7
		}

		i += 8
	}

	for i < len(vector) {
		if vector[i] < min {
			min = vector[i]
		}
		if vector[i] > max {
			max = vector[i]
		}
		i++
	}

	var scale, offset float32
	if max > min {
		scale = (max - min) / 255.0
		offset = min
	} else {
		scale = 1.0
		offset = min
	}

	quantized := make([]byte, len(vector))
	invScale := 1.0 / scale

	for i, v := range vector {
		q := uint8((v - offset) * invScale)
		quantized[i] = q
	}

	return QuantizedVector{
		Data:       quantized,
		Dimensions: len(vector),
		Scale:      scale,
		Offset:     offset,
	}
}

func getShardIndex(id string, numShards uint32) int {
	h := fnv32(id)

	return int(h % numShards)
}

func fnv32(s string) uint32 {
	var h uint32 = 2166136261
	for i := 0; i < len(s); i++ {
		h = (h * 16777619) ^ uint32(s[i])
	}
	return h
}

func generateCacheKey(query []float32, opts *SearchOptions) string {
	var buf bytes.Buffer

	for _, v := range query {
		binary.Write(&buf, binary.LittleEndian, v)
	}

	if opts != nil {
		binary.Write(&buf, binary.LittleEndian, int32(opts.MaxResults))
		binary.Write(&buf, binary.LittleEndian, opts.SimilarityThreshold)
		binary.Write(&buf, binary.LittleEndian, opts.SortByCreated)
		binary.Write(&buf, binary.LittleEndian, opts.SortByUpdated)
		binary.Write(&buf, binary.LittleEndian, int32(opts.Quantization))
		binary.Write(&buf, binary.LittleEndian, opts.UseANN)
		binary.Write(&buf, binary.LittleEndian, int32(opts.EfSearch))
		binary.Write(&buf, binary.LittleEndian, opts.BFSMode)
		binary.Write(&buf, binary.LittleEndian, opts.GreedyMode)
	}

	hash := sha256.Sum256(buf.Bytes())
	return fmt.Sprintf("%x", hash)
}

func (db *AntarysDB) BuildIndex(collectionName string, options IndexingOptions) error {
	log.Printf("Building index for collection %s with options: %+v", collectionName, options)
	db.Mutex.RLock()
	coll, exists := db.Collections[collectionName]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collectionName)
	}

	log.Printf("Building HNSW index for collection %s with dimensions %d", collectionName, coll.Dimensions)
	return db.BuildHNSWIndex(collectionName, options)
}

func (db *AntarysDB) BuildHNSWIndex(collectionName string, options IndexingOptions) error {
	log.Printf("Building HNSW index for collection %s...", collectionName)
	startTime := time.Now()

	db.Mutex.RLock()
	coll, exists := db.Collections[collectionName]
	db.Mutex.RUnlock()

	if !exists {
		return fmt.Errorf("collection %s does not exist", collectionName)
	}

	if options.M <= 0 {
		options.M = DefaultHNSWM
	}

	if options.EfConstruction <= 0 {
		options.EfConstruction = DefaultHNSWEfConstruction
	}

	var wg sync.WaitGroup
	errChan := make(chan error, len(coll.Shards))

	for i, shard := range coll.Shards {
		wg.Add(1)
		go func(shardIdx int, s *ShardedCollectionData) {
			defer wg.Done()

			s.Mutex.Lock()
			s.HNSWGraph = &hnswGraph{
				Nodes:      make(map[uint32]*hnswNode),
				IDToNodeID: make(map[string]uint32),
				M:          options.M,
				MaxLevel:   0,
				NextNodeID: 1,
			}

			vectorIDs := make([]string, 0, len(s.Vectors))
			for id := range s.Vectors {
				vectorIDs = append(vectorIDs, id)
			}
			s.Mutex.Unlock()

			const batchSize = 1000
			for batchStart := 0; batchStart < len(vectorIDs); batchStart += batchSize {
				batchEnd := batchStart + batchSize
				if batchEnd > len(vectorIDs) {
					batchEnd = len(vectorIDs)
				}

				batch := vectorIDs[batchStart:batchEnd]

				for _, id := range batch {
					s.Mutex.Lock()
					vector, exists := s.Vectors[id]
					if exists {

						db.updateHNSWGraph(s, id, vector)
					}
					s.Mutex.Unlock()
				}
			}

			log.Printf("Shard %d: HNSW index construction complete with %d vectors",
				shardIdx, len(s.HNSWGraph.Nodes))

		}(i, shard)
	}

	wg.Wait()
	close(errChan)

	select {
	case err := <-errChan:
		return err
	default:

	}

	log.Printf("HNSW index construction complete for collection %s in %v",
		collectionName, time.Since(startTime))
	return nil
}

func (db *AntarysDB) GetCollectionDimensions(collectionName string) (int, error) {
	db.Mutex.RLock()
	defer db.Mutex.RUnlock()

	coll, exists := db.Collections[collectionName]
	if !exists {
		return 0, fmt.Errorf("collection %s does not exist", collectionName)
	}

	return coll.Dimensions, nil
}
