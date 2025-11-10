package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/antarys-ai/antarys"
	"github.com/dgraph-io/badger/v3"
	"github.com/gorilla/mux"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
)

const (
	DefaultPort                = 8080
	DefaultDimensions          = 1536
	DefaultShards              = 16
	DefaultQueryThreads        = 8
	DefaultMaxResults          = 100
	DefaultSimilarityThreshold = 0.7
	DefaultBatchSize           = 1000
	ShutdownTimeout            = 5 * time.Second
	BadgerDBPath               = "./metadata"
	DefaultEmbeddingModel      = 1
)

const (
	OptimalChunkSize      = 2000
	OptimalDBBatchSize    = 5000
	PreprocessScaleFactor = 1.5
	DBScaleFactor         = 2.5
	MetadataAsyncWrites   = true
	ForceGCInterval       = 10
	ReuseVectorBuffers    = true
	PreallocateVectors    = true
	MaxRequestBodySize    = 100 * 1024 * 1024
	MaxMetadataSize       = 1 * 1024 * 1024
)

const (
	MaxConcurrentStreams = 500
	ReadHeaderTimeout    = 5
	ReadTimeout          = 60
	WriteTimeout         = 300
	IdleTimeout          = 120
	MaxConnsPerHost      = 0

	GCPercentage      = 100
	PreemptiveGCBytes = 4 << 30

	TCPKeepAliveSeconds = 30
)

var (
	isReady      atomic.Bool
	startTime    time.Time
	requestCount atomic.Int64
)

type Config struct {
	Port                int
	DataDir             string
	EnableGPU           bool
	Shards              int
	QueryThreads        int
	CommitInterval      time.Duration
	CacheSize           int
	OptimizationLvl     int
	EnablePQ            bool
	EnableHNSW          bool
	MaxResults          int
	BatchSize           int
	SimilarityThreshold float32
	BadgerDir           string
	EnableEmbedding     bool
	EmbeddingModelID    int
	CacheDir            string
}

type EmbeddingModelInfo struct {
	ID          int    `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Dimensions  int    `json:"dimensions"`
	Downloaded  bool   `json:"downloaded"`
}

type EmbeddingRequest struct {
	Input     []string `json:"input"`
	BatchSize int      `json:"batchSize"`
}

type EmbeddingResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Model      string      `json:"model"`
	Dimensions int         `json:"dimensions"`
	Count      int         `json:"count"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

func getAvailableModels(cacheDir string) []EmbeddingModelInfo {
	supportedModels := antarys.ListSupportedModels()
	models := make([]EmbeddingModelInfo, len(supportedModels))

	for i, model := range supportedModels {
		modelID := i + 1
		models[i] = EmbeddingModelInfo{
			ID:          modelID,
			Name:        string(model.Model),
			Description: model.Description,
			Dimensions:  model.Dim,
			Downloaded:  isModelDownloaded(cacheDir, model.Model),
		}
	}

	return models
}

func isModelDownloaded(cacheDir string, model antarys.EmbeddingModel) bool {
	return antarys.HasModel(model, cacheDir)
}

func getModelByID(modelID int) (antarys.EmbeddingModel, error) {
	supportedModels := antarys.ListSupportedModels()
	if modelID < 1 || modelID > len(supportedModels) {
		return "", fmt.Errorf("invalid model ID %d, must be between 1 and %d", modelID, len(supportedModels))
	}
	return supportedModels[modelID-1].Model, nil
}

type APIService struct {
	config         Config
	router         *mux.Router
	server         *http.Server
	shutdownSignal chan struct{}
	db             *antarys.AntarysDB
	metaStore      *badger.DB

	embeddingModel   *antarys.FlagEmbedding
	embeddingEnabled bool
}

func NewAPIService(config Config) (*APIService, error) {
	service := &APIService{
		config:         config,
		router:         mux.NewRouter(),
		shutdownSignal: make(chan struct{}),
	}

	dbConfig := antarys.Config{
		DataDir:          config.DataDir,
		CommitInterval:   config.CommitInterval,
		QueryConcurrency: config.QueryThreads,
		CacheSize:        config.CacheSize,
		IndexingOptions: antarys.IndexingOptions{
			UseHNSW:              config.EnableHNSW,
			M:                    48,
			EfConstruction:       360,
			NumShards:            config.Shards,
			ParallelConstruction: true,
			Quantization:         antarys.QuantizationScalar8Bit,
		},
		Compression:     true,
		EnableProfiling: false,
	}

	var err error
	service.db, err = antarys.NewAntarysDBWithConfig(dbConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create antarys database instance: %w", err)
	}

	opts := badger.DefaultOptions(config.BadgerDir)
	opts.Logger = nil
	opts.ValueThreshold = 1024

	service.metaStore, err = badger.Open(opts)
	if err != nil {
		service.db.Close()
		return nil, fmt.Errorf("failed to open metadata database: %w", err)
	}

	service.setupRoutes()

	if config.EnableEmbedding {
		model, err := getModelByID(config.EmbeddingModelID)
		if err != nil {
			return nil, fmt.Errorf("failed to get embedding model: %w", err)
		}

		options := &antarys.InitOptions{
			Model:                model,
			CacheDir:             config.CacheDir,
			MaxLength:            512,
			ShowDownloadProgress: &[]bool{false}[0],
		}

		service.embeddingModel, err = antarys.NewFlagEmbedding(options)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize embedding model: %w", err)
		}

		service.embeddingEnabled = true
		fmt.Println("Embedding APIs available at /embedding/*")

		service.setupEmbeddingRoutes()
	}

	service.server = &http.Server{
		Addr: fmt.Sprintf(":%d", config.Port),
		Handler: h2c.NewHandler(service.router, &http2.Server{
			MaxConcurrentStreams: 100,
			MaxReadFrameSize:     1 << 20,
			IdleTimeout:          60 * time.Second,
		}),
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       180 * time.Second,
		ReadHeaderTimeout: 15 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	return service, nil
}

func (service *APIService) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (service *APIService) setupRoutes() {
	router := service.router
	router.Use(service.corsMiddleware)

	router.HandleFunc("/health", service.handleHealth).Methods("GET", "OPTIONS")
	router.HandleFunc("/info", service.handleInfo).Methods("GET", "OPTIONS")

	router.HandleFunc("/vectors/upsert", service.handleUpsert).Methods("POST", "OPTIONS")
	router.HandleFunc("/vectors/query", service.handleQuery).Methods("POST", "OPTIONS")
	router.HandleFunc("/vectors/delete", service.handleDelete).Methods("POST", "OPTIONS")
	router.HandleFunc("/vectors/batch_query", service.handleBatchQuery).Methods("POST", "OPTIONS")
	router.HandleFunc("/vectors/{id}", service.handleGetVector).Methods("GET", "OPTIONS")

	router.HandleFunc("/collections", service.handleCreateCollection).Methods("POST", "OPTIONS")
	router.HandleFunc("/collections", service.handleListCollections).Methods("GET", "OPTIONS")
	router.HandleFunc("/collections/{name}", service.handleDeleteCollection).Methods("DELETE", "OPTIONS")

	router.HandleFunc("/admin/commit", service.handleCommit).Methods("POST", "OPTIONS")
}

func (service *APIService) setupEmbeddingRoutes() {
	router := service.router
	router.Use(service.corsMiddleware)

	router.HandleFunc("/embedding/generate", service.handleEmbeddingGenerate).Methods("POST", "OPTIONS")
}

func (service *APIService) handleEmbeddingGenerate(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if !service.embeddingEnabled || service.embeddingModel == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error: "Embedding generation is not enabled",
		})
		return
	}

	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error: fmt.Sprintf("Invalid request body: %v", err),
		})
		return
	}

	if len(req.Input) == 0 {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error: "Input array cannot be empty",
		})
		return
	}

	if req.BatchSize <= 0 {
		req.BatchSize = 256
	}

	embeddings, err := service.embeddingModel.Embed(req.Input, req.BatchSize)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error: fmt.Sprintf("Failed to generate embeddings: %v", err),
		})
		return
	}

	modelInfo := getModelInfoByID(service.config.EmbeddingModelID)
	if modelInfo == nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error: "Model information not found",
		})
		return
	}

	response := EmbeddingResponse{
		Embeddings: embeddings,
		Model:      string(modelInfo.Model),
		Dimensions: modelInfo.Dimensions,
		Count:      len(embeddings),
	}

	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Failed to encode response: %v", err)
	}
}

func getModelInfoByID(modelID int) *struct {
	Model      antarys.EmbeddingModel
	Dimensions int
} {
	modelMap := map[int]struct {
		Model      antarys.EmbeddingModel
		Dimensions int
	}{
		1: {Model: antarys.BGEBaseEN, Dimensions: 768},
		2: {Model: antarys.BGEBaseENV15, Dimensions: 768},
		3: {Model: antarys.BGESmallEN, Dimensions: 384},
		4: {Model: antarys.BGESmallENV15, Dimensions: 384},
		5: {Model: antarys.BGESmallZH, Dimensions: 512},
	}

	if info, exists := modelMap[modelID]; exists {
		return &info
	}
	return nil
}

func (service *APIService) Start() error {

	startTime = time.Now()

	isReady.Store(true)

	log.Printf("Starting Antarys API server on port %d", service.config.Port)
	log.Printf("Shards: %d, GPU: %v, HNSW: %v, PQ: %v",
		service.config.Shards,
		service.config.EnableGPU, service.config.EnableHNSW, service.config.EnablePQ)

	go func() {
		if err := service.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	return nil
}

func (service *APIService) Shutdown() error {
	log.Println("shutting down...")
	isReady.Store(false)

	shutdownStart := time.Now()

	httpCtx, httpCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer httpCancel()

	if err := service.server.Shutdown(httpCtx); err != nil {
		log.Printf("HTTP server shutdown warning: %v", err)

		service.server.Close()
	}

	log.Printf("HTTP server stopped in %v", time.Since(shutdownStart))

	stopStart := time.Now()

	service.db.Mutex.RLock()
	var workerWg sync.WaitGroup

	for _, coll := range service.db.Collections {
		for _, shard := range coll.Shards {
			if shard.AsyncWorkerStop != nil {
				workerWg.Add(1)
				go func(s *antarys.ShardedCollectionData) {
					defer workerWg.Done()
					s.StopAsyncWorkers()
				}(shard)
			}
		}
	}

	service.db.Mutex.RUnlock()

	workerDone := make(chan struct{})
	go func() {
		workerWg.Wait()
		close(workerDone)
	}()

	select {
	case <-workerDone:
		log.Printf("Async workers stopped in %v", time.Since(stopStart))
	case <-time.After(1 * time.Second):
		log.Printf("Async workers stop timed out after 1s")
	}

	commitStart := time.Now()
	commitDone := make(chan error, 1)

	go func() {
		commitDone <- service.db.Commit()
	}()

	select {
	case err := <-commitDone:
		if err != nil {
			log.Printf("commit error: %v", err)
		} else {
			log.Printf("commit completed in %v", time.Since(commitStart))
		}
	case <-time.After(2 * time.Second):
		log.Printf("commit timed out after 2s")
	}

	if err := service.db.Close(); err != nil {
		log.Printf("Database close error: %v", err)
	}

	if err := service.metaStore.Close(); err != nil {
		log.Printf("Metadata store close error: %v", err)
	}

	if service.config.EnableEmbedding {
		if service.embeddingModel != nil {
			embeddingShutdownStart := time.Now()
			if err := service.embeddingModel.Destroy(); err != nil {
				log.Printf("Embedding model cleanup error: %v", err)
			} else {
				log.Printf("Embedding model destroyed in %v", time.Since(embeddingShutdownStart))
			}
		}
	}

	totalShutdownTime := time.Since(shutdownStart)
	log.Printf("shutdown completed in %v", totalShutdownTime)

	if totalShutdownTime > 5*time.Second {
		log.Printf("Warning: Shutdown took longer than 5 seconds")
	}

	return nil
}

func (service *APIService) handleHealth(w http.ResponseWriter, r *http.Request) {
	if !isReady.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	stats := service.db.GetStats()
	gpuInfo := map[string]interface{}{}

	info := map[string]interface{}{

		"hnsw_enabled": service.config.EnableHNSW,
		"pq_enabled":   service.config.EnablePQ,
		"num_shards":   service.config.Shards,
		"optimization": service.config.OptimizationLvl,
		"stats":        stats,
		"gpu_info":     gpuInfo,
		"version":      antarys.VERSION,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(info)
}

func (service *APIService) handleInfo(w http.ResponseWriter, r *http.Request) {
	uptime := time.Since(startTime).String()
	requestsServed := requestCount.Load()

	info := map[string]interface{}{
		"uptime":          uptime,
		"requests_served": requestsServed,
		"cpu_count":       runtime.NumCPU(),
		"goroutines":      runtime.NumGoroutine(),
		"version":         antarys.VERSION,
	}

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	info["memory_allocated_mb"] = float64(memStats.Alloc) / 1024 / 1024

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

func (service *APIService) handleUpsert(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)
	startTime := time.Now()

	body, err := io.ReadAll(io.LimitReader(r.Body, 50*1024*1024))
	if err != nil {
		http.Error(w, "Error reading request: "+err.Error(), http.StatusBadRequest)
		return
	}

	var request struct {
		ID         string        `json:"id"`
		Vector     []interface{} `json:"vector,omitempty"`
		Values     []interface{} `json:"values,omitempty"`
		Metadata   interface{}   `json:"metadata,omitempty"`
		Collection string        `json:"collection,omitempty"`
		Vectors    []interface{} `json:"vectors,omitempty"`
	}

	if err := json.Unmarshal(body, &request); err != nil {
		http.Error(w, "Invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	collection := "default"
	if request.Collection != "" {
		collection = request.Collection
	}

	if len(request.Vectors) > 0 {
		service.handleBatchUpsert(w, request.Vectors, collection)
		return
	}

	if request.ID == "" {
		http.Error(w, "Missing 'id' field", http.StatusBadRequest)
		return
	}

	var vectorValues []float32

	if len(request.Vector) > 0 {
		vectorValues = service.extractVectorFast(request.Vector)
	} else if len(request.Values) > 0 {
		vectorValues = service.extractVectorFast(request.Values)
	}

	antarys.NormaliseVectorLockless(vectorValues)

	if len(vectorValues) == 0 {
		http.Error(w, "Missing or invalid vector data", http.StatusBadRequest)
		return
	}

	collDim, _ := service.db.GetCollectionDimensions(collection)
	if collDim > 0 && len(vectorValues) != collDim {
		http.Error(w, fmt.Sprintf("Vector dimension mismatch: got %d, expected %d",
			len(vectorValues), collDim), http.StatusBadRequest)
		return
	}

	err = service.db.InsertInCollection(collection, request.ID, vectorValues, request.Metadata)
	if err != nil {
		http.Error(w, "Insert failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"id":      request.ID,
		"time_ms": time.Since(startTime).Milliseconds(),
	})
}

func (service *APIService) extractVectorFast(values []interface{}) []float32 {
	if len(values) == 0 {
		return nil
	}

	vector := antarys.GetOptimalVector(len(values))
	if len(vector) != len(values) {
		vector = make([]float32, len(values))
	}

	for i, val := range values {
		switch v := val.(type) {
		case float64:
			vector[i] = float32(v)
		case float32:
			vector[i] = v
		case int:
			vector[i] = float32(v)
		case int64:
			vector[i] = float32(v)
		default:
			if f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 32); err == nil {
				vector[i] = float32(f)
			}
		}
	}

	return vector
}

func (service *APIService) handleBatchUpsert(w http.ResponseWriter, vectors []interface{}, collection string) {
	startTime := time.Now()

	batch := make([]*antarys.BatchInsert, 0, len(vectors))
	now := time.Now()

	numWorkers := runtime.NumCPU()
	batchSize := (len(vectors) + numWorkers - 1) / numWorkers

	resultChans := make([]chan []*antarys.BatchInsert, numWorkers)

	for i := 0; i < numWorkers; i++ {
		start := i * batchSize
		end := start + batchSize
		if end > len(vectors) {
			end = len(vectors)
		}
		if start >= len(vectors) {
			break
		}

		resultChans[i] = make(chan []*antarys.BatchInsert, 1)

		go func(workerVectors []interface{}, resultChan chan []*antarys.BatchInsert) {
			workerBatch := make([]*antarys.BatchInsert, 0, len(workerVectors))

			for _, v := range workerVectors {
				vecMap, ok := v.(map[string]interface{})
				if !ok {
					continue
				}

				idVal, hasID := vecMap["id"]
				if !hasID {
					continue
				}

				id, ok := idVal.(string)
				if !ok {
					continue
				}

				var vectorValues []float32
				if vectorArray, ok := vecMap["vector"].([]interface{}); ok {
					vectorValues = service.extractVectorFast(vectorArray)
				} else if valuesArray, ok := vecMap["values"].([]interface{}); ok {
					vectorValues = service.extractVectorFast(valuesArray)
				}

				if len(vectorValues) == 0 {
					continue
				}

				antarys.NormaliseVectorLockless(vectorValues)

				workerBatch = append(workerBatch, &antarys.BatchInsert{
					Id:       id,
					Vector:   vectorValues,
					Metadata: vecMap["metadata"],
					Created:  now,
					Updated:  now,
				})
			}

			resultChan <- workerBatch
		}(vectors[start:end], resultChans[i])
	}

	for _, resultChan := range resultChans {
		if resultChan != nil {
			workerBatch := <-resultChan
			batch = append(batch, workerBatch...)
		}
	}

	if len(batch) == 0 {
		http.Error(w, "No valid vectors to insert", http.StatusBadRequest)
		return
	}

	err := service.db.BatchInsert(collection, batch)
	if err != nil {
		http.Error(w, "Batch insert failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	totalTime := time.Since(startTime)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":         true,
		"count":           len(batch),
		"time_ms":         totalTime.Milliseconds(),
		"vectors_per_sec": float64(len(batch)) / totalTime.Seconds(),
	})
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func (service *APIService) handleQuery(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusBadRequest)
		return
	}

	var request map[string]interface{}
	err = json.Unmarshal(body, &request)
	if err != nil {
		http.Error(w, "Invalid JSON format: "+err.Error(), http.StatusBadRequest)
		return
	}

	collection := "default"
	if collName, ok := request["collection"].(string); ok {
		collection = collName
	}

	var vectorValues []float32
	var hasVector bool

	if vectorArray, ok := request["vector"].([]interface{}); ok {
		hasVector = true
		vectorValues = make([]float32, len(vectorArray))
		for i, val := range vectorArray {
			switch v := val.(type) {
			case float64:
				vectorValues[i] = float32(v)
			case float32:
				vectorValues[i] = v
			case int:
				vectorValues[i] = float32(v)
			case int64:
				vectorValues[i] = float32(v)
			case string:
				f, err := strconv.ParseFloat(v, 32)
				if err == nil {
					vectorValues[i] = float32(f)
				}
			default:
				f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 32)
				if err == nil {
					vectorValues[i] = float32(f)
				}
			}
		}
	} else if valuesArray, ok := request["values"].([]interface{}); ok {

		hasVector = true
		vectorValues = make([]float32, len(valuesArray))
		for i, val := range valuesArray {
			switch v := val.(type) {
			case float64:
				vectorValues[i] = float32(v)
			case float32:
				vectorValues[i] = v
			case int:
				vectorValues[i] = float32(v)
			case int64:
				vectorValues[i] = float32(v)
			case string:
				f, err := strconv.ParseFloat(v, 32)
				if err == nil {
					vectorValues[i] = float32(f)
				}
			default:
				f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 32)
				if err == nil {
					vectorValues[i] = float32(f)
				}
			}
		}
	}

	if !hasVector {
		http.Error(w, "Missing or invalid 'vector' or 'values' field", http.StatusBadRequest)
		return
	}

	if len(vectorValues) == 0 {
		http.Error(w, "Query vector cannot be empty", http.StatusBadRequest)
		return
	}

	topK := service.config.MaxResults
	if topKVal, ok := request["top_k"].(float64); ok {
		topK = int(topKVal)
	}

	includeMetadata := true
	if includeMeta, ok := request["include_metadata"].(bool); ok {
		includeMetadata = includeMeta
	}

	includeVectors := false
	if includeVec, ok := request["include_vectors"].(bool); ok {
		includeVectors = includeVec
	}

	threshold := float32(0.0)
	if thresholdVal, ok := request["threshold"].(float64); ok {
		threshold = float32(thresholdVal)
	}

	useANN := true
	if useANNVal, ok := request["use_ann"].(bool); ok {
		useANN = useANNVal
	}

	efSearch := 400
	if efSearchVal, ok := request["ef_search"].(float64); ok {
		efSearch = int(efSearchVal)
	}

	minEf := max(topK*4, 300)
	maxEf := max(topK*10, 2000)
	efSearch = max(minEf, min(efSearch, maxEf))

	searchOpts := &antarys.SearchOptions{
		MaxResults:          topK,
		SimilarityThreshold: threshold,
		UseANN:              useANN,
		EfSearch:            efSearch,
		Quantization:        antarys.QuantizationScalar8Bit,
	}

	if filterVal, ok := request["filter"]; ok {
		if filterMap, ok := filterVal.(map[string]interface{}); ok {

			searchOpts.MetadataFilter = func(metadata any) bool {
				if metadata == nil {
					return false
				}

				mdMap, ok := metadata.(map[string]interface{})
				if !ok {
					return false
				}

				for key, value := range filterMap {
					metaVal, exists := mdMap[key]
					if !exists {
						return false
					}

					switch v := value.(type) {
					case string:
						if metaVal != v {
							return false
						}
					case float64:
						if metaFloat, ok := metaVal.(float64); !ok || metaFloat != v {
							return false
						}
					case bool:
						if metaBool, ok := metaVal.(bool); !ok || metaBool != v {
							return false
						}
					case []interface{}:
						metaList, ok := metaVal.([]interface{})
						if !ok {
							return false
						}
						found := false
						for _, item := range metaList {
							itemStr := fmt.Sprintf("%v", item)
							valueStr := fmt.Sprintf("%v", v)
							if itemStr == valueStr {
								found = true
								break
							}
						}
						if !found {
							return false
						}
					default:
						metaValStr := fmt.Sprintf("%v", metaVal)
						valueStr := fmt.Sprintf("%v", v)
						if metaValStr != valueStr {
							return false
						}
					}
				}

				return true
			}
		}
	}

	var results []antarys.SearchResult
	var searchErr error

	results, searchErr = service.db.SearchInCollection(collection, vectorValues, searchOpts)

	if searchErr != nil {
		http.Error(w, "Search error: "+searchErr.Error(), http.StatusInternalServerError)
		return
	}

	type responseResult struct {
		ID       string                 `json:"id"`
		Score    float32                `json:"score"`
		Vector   []float32              `json:"vector,omitempty"`
		Metadata map[string]interface{} `json:"metadata,omitempty"`
	}

	response := struct {
		Results []responseResult `json:"results"`
		Count   int              `json:"count"`
	}{
		Results: make([]responseResult, 0, len(results)),
		Count:   len(results),
	}

	for _, result := range results {
		res := responseResult{
			ID:    result.Record.ID,
			Score: result.Score,
		}

		if includeVectors {
			res.Vector = result.Record.Vector
		}

		if includeMetadata {
			switch v := result.Record.Metadata.(type) {
			case map[string]interface{}:
				res.Metadata = v
			default:

				mdBytes, err := json.Marshal(result.Record.Metadata)
				if err == nil {
					var mdMap map[string]interface{}
					if json.Unmarshal(mdBytes, &mdMap) == nil {
						res.Metadata = mdMap
					}
				}
			}

			if res.Metadata == nil {
				res.Metadata = make(map[string]interface{})
			}

			metaKey := fmt.Sprintf("%s:%s:meta", collection, result.Record.ID)
			err := service.metaStore.View(func(txn *badger.Txn) error {
				item, err := txn.Get([]byte(metaKey))
				if err != nil {
					return err
				}

				return item.Value(func(val []byte) error {
					var extraMeta map[string]interface{}
					if err := json.Unmarshal(val, &extraMeta); err != nil {
						return err
					}

					for k, v := range extraMeta {
						res.Metadata[k] = v
					}
					return nil
				})
			})

			if err != nil && err != badger.ErrKeyNotFound {
				log.Printf("Warning: Error retrieving extra metadata for %s: %v", result.Record.ID, err)
			}
		}

		response.Results = append(response.Results, res)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

func (service *APIService) handleBatchQuery(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusBadRequest)
		return
	}

	var request map[string]interface{}
	err = json.Unmarshal(body, &request)
	if err != nil {
		http.Error(w, "Invalid JSON format: "+err.Error(), http.StatusBadRequest)
		return
	}

	collection := "default"
	if collName, ok := request["collection"].(string); ok {
		collection = collName
	}

	queriesVal, ok := request["queries"]
	if !ok {
		http.Error(w, "Missing 'queries' field", http.StatusBadRequest)
		return
	}

	queries, ok := queriesVal.([]interface{})
	if !ok {
		http.Error(w, "Invalid 'queries' format, must be an array", http.StatusBadRequest)
		return
	}

	includeMetadata := true
	if includeMeta, ok := request["include_metadata"].(bool); ok {
		includeMetadata = includeMeta
	}

	includeVectors := false
	if includeVec, ok := request["include_vectors"].(bool); ok {
		includeVectors = includeVec
	}

	threshold := service.config.SimilarityThreshold
	if thresholdVal, ok := request["threshold"].(float64); ok {
		threshold = float32(thresholdVal)
	}

	useANN := service.config.EnableHNSW
	if useANNVal, ok := request["use_ann"].(bool); ok {
		useANN = useANNVal
	}

	type queryInfo struct {
		Vector    []float32
		TopK      int
		QueryID   string
		Filter    map[string]interface{}
		ResultIdx int
	}

	queryInfos := make([]queryInfo, 0, len(queries))

	for i, q := range queries {
		queryMap, ok := q.(map[string]interface{})
		if !ok {
			http.Error(w, fmt.Sprintf("Query %d is not an object", i), http.StatusBadRequest)
			return
		}

		var vectorValues []float32
		var hasVector bool

		if vectorArray, ok := queryMap["vector"].([]interface{}); ok {
			hasVector = true
			vectorValues = make([]float32, len(vectorArray))
			for j, val := range vectorArray {
				switch v := val.(type) {
				case float64:
					vectorValues[j] = float32(v)
				case float32:
					vectorValues[j] = v
				case int:
					vectorValues[j] = float32(v)
				case int64:
					vectorValues[j] = float32(v)
				case string:
					f, err := strconv.ParseFloat(v, 32)
					if err == nil {
						vectorValues[j] = float32(f)
					}
				default:

					f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 32)
					if err == nil {
						vectorValues[j] = float32(f)
					}
				}
			}
		} else if valuesArray, ok := queryMap["values"].([]interface{}); ok {

			hasVector = true
			vectorValues = make([]float32, len(valuesArray))
			for j, val := range valuesArray {
				switch v := val.(type) {
				case float64:
					vectorValues[j] = float32(v)
				case float32:
					vectorValues[j] = v
				case int:
					vectorValues[j] = float32(v)
				case int64:
					vectorValues[j] = float32(v)
				case string:
					f, err := strconv.ParseFloat(v, 32)
					if err == nil {
						vectorValues[j] = float32(f)
					}
				default:

					f, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 32)
					if err == nil {
						vectorValues[j] = float32(f)
					}
				}
			}
		}

		if !hasVector {
			http.Error(w, fmt.Sprintf("Query %d is missing 'vector' or 'values' field", i), http.StatusBadRequest)
			return
		}

		if len(vectorValues) == 0 {
			http.Error(w, fmt.Sprintf("Query %d has empty vector", i), http.StatusBadRequest)
			return
		}

		topK := service.config.MaxResults
		if topKVal, ok := queryMap["top_k"].(float64); ok {
			topK = int(topKVal)
		}

		queryID := fmt.Sprintf("query_%d", i)
		if idVal, ok := queryMap["query_id"].(string); ok {
			queryID = idVal
		}

		var filterMap map[string]interface{}
		if filterVal, ok := queryMap["filter"]; ok {
			if fMap, ok := filterVal.(map[string]interface{}); ok {
				filterMap = fMap
			}
		}

		queryInfos = append(queryInfos, queryInfo{
			Vector:    vectorValues,
			TopK:      topK,
			QueryID:   queryID,
			Filter:    filterMap,
			ResultIdx: i,
		})
	}

	type responseResult struct {
		ID       string                 `json:"id"`
		Score    float32                `json:"score"`
		Vector   []float32              `json:"vector,omitempty"`
		Metadata map[string]interface{} `json:"metadata,omitempty"`
	}

	type queryResult struct {
		QueryID string           `json:"query_id"`
		Results []responseResult `json:"results"`
		Count   int              `json:"count"`
	}

	response := struct {
		Results []queryResult `json:"results"`
	}{
		Results: make([]queryResult, len(queryInfos)),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	errChan := make(chan error, len(queryInfos))

	for _, qi := range queryInfos {
		wg.Add(1)
		go func(info queryInfo) {
			defer wg.Done()

			searchOpts := &antarys.SearchOptions{
				MaxResults:          info.TopK,
				SimilarityThreshold: threshold,
				UseANN:              useANN,
				EfSearch:            100,
			}

			if info.Filter != nil {

				searchOpts.MetadataFilter = func(metadata any) bool {
					if metadata == nil {
						return false
					}

					mdMap, ok := metadata.(map[string]interface{})
					if !ok {
						return false
					}

					for key, value := range info.Filter {
						metaVal, exists := mdMap[key]
						if !exists {
							return false
						}

						switch v := value.(type) {
						case string:

							if metaVal != v {
								return false
							}
						case float64:

							if metaFloat, ok := metaVal.(float64); !ok || metaFloat != v {
								return false
							}
						case bool:

							if metaBool, ok := metaVal.(bool); !ok || metaBool != v {
								return false
							}
						case []interface{}:

							metaList, ok := metaVal.([]interface{})
							if !ok {
								return false
							}
							found := false
							for _, item := range metaList {

								itemStr := fmt.Sprintf("%v", item)
								valueStr := fmt.Sprintf("%v", v)
								if itemStr == valueStr {
									found = true
									break
								}
							}
							if !found {
								return false
							}
						default:

							metaValStr := fmt.Sprintf("%v", metaVal)
							valueStr := fmt.Sprintf("%v", value)
							if metaValStr != valueStr {
								return false
							}
						}
					}

					return true
				}
			}

			var results []antarys.SearchResult
			var searchErr error

			results, searchErr = service.db.SearchInCollection(collection, info.Vector, searchOpts)

			if searchErr != nil {
				errChan <- fmt.Errorf("search error for query %s: %w", info.QueryID, searchErr)
				return
			}

			queryRes := queryResult{
				QueryID: info.QueryID,
				Results: make([]responseResult, 0, len(results)),
				Count:   len(results),
			}

			for _, result := range results {
				res := responseResult{
					ID:    result.Record.ID,
					Score: result.Score,
				}

				if includeVectors {
					res.Vector = result.Record.Vector
				}

				if includeMetadata {
					switch v := result.Record.Metadata.(type) {
					case map[string]interface{}:
						res.Metadata = v
					default:

						mdBytes, err := json.Marshal(result.Record.Metadata)
						if err == nil {
							var mdMap map[string]interface{}
							if json.Unmarshal(mdBytes, &mdMap) == nil {
								res.Metadata = mdMap
							}
						}
					}

					if res.Metadata == nil {
						res.Metadata = make(map[string]interface{})
					}

					metaKey := fmt.Sprintf("%s:%s:meta", collection, result.Record.ID)
					err := service.metaStore.View(func(txn *badger.Txn) error {
						item, err := txn.Get([]byte(metaKey))
						if err != nil {
							return err
						}

						return item.Value(func(val []byte) error {
							var extraMeta map[string]interface{}
							if err := json.Unmarshal(val, &extraMeta); err != nil {
								return err
							}

							for k, v := range extraMeta {
								res.Metadata[k] = v
							}
							return nil
						})
					})

					if err != nil && err != badger.ErrKeyNotFound {
						log.Printf("Warning: Error retrieving extra metadata for %s: %v", result.Record.ID, err)
					}
				}

				queryRes.Results = append(queryRes.Results, res)
			}

			mu.Lock()
			response.Results[info.ResultIdx] = queryRes
			mu.Unlock()
		}(qi)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

func (service *APIService) handleDelete(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	var request struct {
		IDs        []string `json:"ids"`
		Collection string   `json:"collection,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request format: "+err.Error(), http.StatusBadRequest)
		return
	}

	if len(request.IDs) == 0 {
		http.Error(w, "No IDs provided for deletion", http.StatusBadRequest)
		return
	}

	collection := "default"
	if request.Collection != "" {
		collection = request.Collection
	}

	failed := make([]string, 0)
	deleted := make([]string, 0)

	for _, id := range request.IDs {
		err := service.db.DeleteFromCollection(collection, id)
		if err != nil {
			failed = append(failed, id)
			log.Printf("Failed to delete vector %s: %v", id, err)
		} else {
			deleted = append(deleted, id)

			metaKey := fmt.Sprintf("%s:%s:meta", collection, id)
			if err := service.metaStore.Update(func(txn *badger.Txn) error {
				return txn.Delete([]byte(metaKey))
			}); err != nil {
				log.Printf("Warning: Failed to delete metadata for %s: %v", id, err)
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"deleted": deleted,
		"failed":  failed,
	})
}

func (service *APIService) handleGetVector(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	vars := mux.Vars(r)
	id := vars["id"]

	if id == "" {
		http.Error(w, "Vector ID not provided", http.StatusBadRequest)
		return
	}

	collection := r.URL.Query().Get("collection")
	if collection == "" {
		collection = "default"
	}

	record, err := service.db.GetFromCollection(collection, id)
	if err != nil {
		http.Error(w, "Vector not found: "+err.Error(), http.StatusNotFound)
		return
	}

	var extraMetadata map[string]interface{}
	metaKey := fmt.Sprintf("%s:%s:meta", collection, id)

	service.metaStore.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(metaKey))
		if err != nil {
			return err
		}

		return item.Value(func(val []byte) error {
			return json.Unmarshal(val, &extraMetadata)
		})
	})

	response := map[string]interface{}{
		"id":         record.ID,
		"vector":     record.Vector,
		"created_at": record.Created,
		"updated_at": record.Updated,
		"collection": record.Collection,
	}

	if record.Metadata != nil {
		switch v := record.Metadata.(type) {
		case map[string]interface{}:
			response["metadata"] = v
		default:

			mdBytes, err := json.Marshal(record.Metadata)
			if err == nil {
				var mdMap map[string]interface{}
				if json.Unmarshal(mdBytes, &mdMap) == nil {
					response["metadata"] = mdMap
				} else {
					response["metadata"] = record.Metadata
				}
			} else {
				response["metadata"] = record.Metadata
			}
		}
	} else {
		response["metadata"] = map[string]interface{}{}
	}

	if extraMetadata != nil {
		metadata, ok := response["metadata"].(map[string]interface{})
		if ok {
			for k, v := range extraMetadata {
				metadata[k] = v
			}
			response["metadata"] = metadata
		}
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

func (service *APIService) handleCreateCollection(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	var request struct {
		Name           string `json:"name"`
		Dimensions     int    `json:"dimensions,omitempty"`
		Shards         int    `json:"shards,omitempty"`
		EnableHNSW     bool   `json:"enable_hnsw,omitempty"`
		EnablePQ       bool   `json:"enable_pq,omitempty"`
		EfConstruction int    `json:"ef_construction,omitempty"`
		M              int    `json:"m,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request format: "+err.Error(), http.StatusBadRequest)
		return
	}

	if request.Name == "" {
		http.Error(w, "Collection name is required", http.StatusBadRequest)
		return
	}

	if request.Dimensions <= 0 {
		request.Dimensions = DefaultDimensions
	}

	if request.Shards <= 0 {
		request.Shards = service.config.Shards
	}

	if !request.EnableHNSW && !request.EnablePQ {
		request.EnableHNSW = service.config.EnableHNSW
		request.EnablePQ = service.config.EnablePQ
	}

	if request.EfConstruction <= 0 {
		request.EfConstruction = 200
	}

	if request.M <= 0 {
		request.M = 16
	}

	options := antarys.IndexingOptions{
		UseHNSW:              request.EnableHNSW,
		M:                    request.M,
		EfConstruction:       request.EfConstruction,
		NumShards:            request.Shards,
		ParallelConstruction: true,
	}

	if request.EnablePQ {
		options.Quantization = antarys.QuantizationScalar8Bit
	}

	err := service.db.CreateCollectionWithOptions(request.Name, request.Dimensions, options)
	if err != nil {
		http.Error(w, "Failed to create collection: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success":    true,
		"name":       request.Name,
		"dimensions": request.Dimensions,
	})
}

func (service *APIService) handleListCollections(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	collections := service.db.ListCollections()

	collectionInfo := make([]map[string]interface{}, 0, len(collections))

	for _, name := range collections {
		count := service.db.CountInCollection(name)
		info := map[string]interface{}{
			"name":  name,
			"count": count,
		}
		collectionInfo = append(collectionInfo, info)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"collections": collectionInfo,
		"count":       len(collections),
	})
}

func (service *APIService) handleDeleteCollection(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	vars := mux.Vars(r)
	name := vars["name"]

	if name == "" {
		http.Error(w, "Collection name not provided", http.StatusBadRequest)
		return
	}

	err := service.db.DeleteCollection(name)
	if err != nil {
		http.Error(w, "Failed to delete collection: "+err.Error(), http.StatusInternalServerError)
		return
	}

	prefix := []byte(name + ":")
	err = service.metaStore.DropPrefix(prefix)
	if err != nil {
		log.Printf("Warning: Failed to delete metadata for collection %s: %v", name, err)
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"name":    name,
	})
}

func (service *APIService) handleCommit(w http.ResponseWriter, r *http.Request) {
	requestCount.Add(1)

	err := service.db.Commit()
	if err != nil {
		http.Error(w, "Failed to commit: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
	})
}

func handleDownloadCommand(args []string, cacheDir string) {
	if len(args) < 2 {
		fmt.Println("Usage: antarys download <command>")
		fmt.Println("Commands:")
		fmt.Println("  onnx                    - Download ONNX runtime")
		fmt.Println("  embedding <model_id>    - Download embedding model")
		fmt.Println("  list-embeddings         - List available embedding models")
		fmt.Println("  remove embedding <id>   - Remove cached embedding model")
		os.Exit(1)
	}

	command := args[1]
	switch command {
	case "onnx":
		fmt.Println("Downloading ONNX runtime...")
		runtime := antarys.ORT_MAC_UNIVERSAL
		if antarys.HasOnnxRuntime(runtime, cacheDir) {
			fmt.Println("ONNX runtime is already downloaded and available")
			return
		}

		path, err := antarys.RetrieveRuntime(runtime, cacheDir, true)

		fmt.Printf("%s\n", path)

		if err != nil {
			fmt.Printf("Error downloading ONNX runtime: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("ONNX runtime downloaded successfully")

	case "embedding":
		if len(args) < 3 {
			fmt.Println("Warning: No model ID provided, downloading default model (ID: 1)")
			args = append(args, "1")
		}

		modelID, err := strconv.Atoi(args[2])
		if err != nil {
			fmt.Printf("Error: Invalid model ID '%s'\n", args[2])
			os.Exit(1)
		}

		model, err := getModelByID(modelID)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}

		if isModelDownloaded(cacheDir, model) {
			fmt.Printf("\nEmbedding model %d (%s) is already downloaded and available\n", modelID, model)
			return
		}

		_, err = antarys.RetrieveModel(model, cacheDir, true)
		if err != nil {
			fmt.Printf("Error downloading embedding model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Embedding model %d downloaded successfully\n", modelID)

	case "list-embeddings":
		models := getAvailableModels(cacheDir)
		fmt.Println("Available embedding models:")
		fmt.Println("ID\tName\t\t\t\tDimensions\tDownloaded\tDescription")
		fmt.Println("--\t----\t\t\t\t----------\t----------\t-----------")
		for _, model := range models {
			downloaded := "No"
			if model.Downloaded {
				downloaded = "Yes"
			}
			fmt.Printf("%d\t%-30s\t%d\t\t%s\t\t%s\n",
				model.ID, model.Name, model.Dimensions, downloaded, model.Description)
		}

	case "remove":
		if len(args) < 4 || args[2] != "embedding" {
			fmt.Println("Usage: antarys download remove embedding <model_id>")
			os.Exit(1)
		}

		modelID, err := strconv.Atoi(args[3])
		if err != nil {
			fmt.Printf("Error: Invalid model ID '%s'\n", args[3])
			os.Exit(1)
		}

		model, err := getModelByID(modelID)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			os.Exit(1)
		}

		if !isModelDownloaded(cacheDir, model) {
			fmt.Printf("Embedding model %d (%s) is not downloaded\n", modelID, model)
			return
		}

		modelPath := filepath.Join(cacheDir, string(model))
		err = os.RemoveAll(modelPath)
		if err != nil {
			fmt.Printf("Error removing embedding model: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Embedding model %d (%s) removed successfully\n", modelID, model)

	default:
		fmt.Printf("Unknown download command: %s\n", command)
		os.Exit(1)
	}
}

func main() {
	startTime := time.Now()
	debug.SetGCPercent(GCPercentage)
	runtime.GOMAXPROCS(runtime.NumCPU())

	if len(os.Args) > 1 && os.Args[1] == "download" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			log.Fatalf("Failed to get home directory: %v", err)
		}
		cacheDir := filepath.Join(homeDir, ".antarys")

		antarys.AntarysTitle()
		handleDownloadCommand(os.Args[1:], cacheDir)
		return
	}

	if len(os.Args) > 1 && os.Args[1] == "version" {
		fmt.Printf("version %s\n", antarys.VERSION)
		return
	}

	antarys.AntarysTitle()

	port := flag.Int("port", 8080, "API server port")
	dataDir := flag.String("data-dir", "./data", "Path to data directory")
	enableGPU := flag.Bool("enable-gpu", false, "Enable GPU acceleration")
	shards := flag.Int("shards", 16, "Number of shards")
	queryThreads := flag.Int("query-threads", runtime.NumCPU()*2, "Number of query threads")
	commitIntervalSec := flag.Int("commit-interval", 60, "Auto-commit interval in seconds")
	cacheSize := flag.Int("cache-size", 10000, "Size of result cache")
	enableHNSW := flag.Bool("enable-hnsw", true, "Enable HNSW indexing")
	enablePQ := flag.Bool("enable-pq", false, "Enable Product Quantization")
	optimizationLvl := flag.Int("optimization", 3, "Optimization level (0-3) for GPU")
	maxResults := flag.Int("max-results", 100, "Default maximum results")
	batchSize := flag.Int("batch-size", 5000, "Default batch size")
	similarityThreshold := flag.Float64("similarity", 0.7, "Default similarity threshold")
	badgerDir := flag.String("meta-dir", "./metadata", "Path to metadata storage")

	enableEmbedding := flag.Bool("enable-embedding", false, "Enable embedding generation")
	embeddingModelID := flag.Int("embedding-model", DefaultEmbeddingModel, "Embedding model ID to use")

	flag.Parse()

	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("Failed to get home directory: %v", err)
	}
	cacheDir := filepath.Join(homeDir, ".antarys")

	config := Config{
		Port:                *port,
		DataDir:             *dataDir,
		EnableGPU:           *enableGPU,
		Shards:              *shards,
		QueryThreads:        *queryThreads,
		CommitInterval:      time.Duration(*commitIntervalSec) * time.Second,
		CacheSize:           *cacheSize,
		OptimizationLvl:     *optimizationLvl,
		EnableHNSW:          *enableHNSW,
		EnablePQ:            *enablePQ,
		MaxResults:          *maxResults,
		BatchSize:           *batchSize,
		SimilarityThreshold: float32(*similarityThreshold),
		BadgerDir:           *badgerDir,
		EnableEmbedding:     *enableEmbedding,
		EmbeddingModelID:    *embeddingModelID,
		CacheDir:            cacheDir,
	}

	fmt.Println("Antarys version", antarys.VERSION)
	fmt.Println("Antarys is still in preview, visit antarys.ai for more")
	fmt.Printf("Starting API server (%s/%s) on %d CPUs with %d MB max memory",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), PreemptiveGCBytes/(1<<20))

	fmt.Println("")

	if config.EnableEmbedding {
		modelID := config.EmbeddingModelID

		runtimeLib, err := antarys.LoadDynamicOnnx()
		if err != nil {
			fmt.Printf("Error: Unsupported platform - %v\n", err)
			os.Exit(1)
		}

		if !antarys.HasOnnxRuntime(antarys.OnnxRuntime(runtimeLib), cacheDir) {
			fmt.Printf("Error: ONNX runtime not found. Please run the following command to download it:\n")
			fmt.Printf("  %s download onnx\n", os.Args[0])
			os.Exit(1)
		}

		if !antarys.IsOnnxValid(cacheDir) {
			fmt.Printf("Error: Corrupted ONNX runtime detected. Removing corrupt file...\n")
			runtimePath := filepath.Join(cacheDir, runtimeLib)
			if err := os.Remove(runtimePath); err != nil {
				fmt.Printf("Warning: Failed to remove corrupt ONNX runtime: %v\n", err)
			}
			fmt.Printf("Please run the following command to re-download ONNX runtime:\n")
			fmt.Printf("  %s download onnx\n", os.Args[0])
			os.Exit(1)
		}

		availableModels := getAvailableModels(cacheDir)
		var selectedModel *EmbeddingModelInfo
		for _, model := range availableModels {
			if model.ID == modelID {
				selectedModel = &model
				break
			}
		}

		if selectedModel == nil {
			fmt.Printf("Error: Invalid embedding model ID %d. Please run the following command to see available models:\n", modelID)
			fmt.Printf("  %s download list-embeddings\n", os.Args[0])
			os.Exit(1)
		}

		if !selectedModel.Downloaded {
			fmt.Printf("Error: Embedding model %d (%s) is not downloaded. Please run the following command to download it:\n",
				modelID, selectedModel.Name)
			fmt.Printf("  %s download embedding %d\n", os.Args[0], modelID)
			os.Exit(1)
		}

		if !antarys.IsModelValid(strconv.Itoa(modelID), cacheDir) {
			fmt.Printf("Error: Corrupted embedding model detected. Removing corrupt model...\n")

			if modelDir, exists := antarys.ModelsByIndex[modelID]; exists {
				modelPath := filepath.Join(cacheDir, modelDir)
				if err := os.RemoveAll(modelPath); err != nil {
					fmt.Printf("Warning: Failed to remove corrupt model: %v\n", err)
				}
			}

			fmt.Printf("Please run the following command to re-download the embedding model:\n")
			fmt.Printf("  %s download embedding %d\n", os.Args[0], modelID)
			os.Exit(1)
		}

		fmt.Printf("Embedding model %d (%s) available\n", modelID, selectedModel.Name)
	}

	service, err := NewAPIService(config)
	if err != nil {
		log.Fatalf("Failed to create API service: %v", err)
	}

	h2Server := &http2.Server{
		MaxConcurrentStreams: uint32(MaxConcurrentStreams),
		MaxReadFrameSize:     (1 << 20),
		IdleTimeout:          30 * time.Second,
	}

	service.server = &http.Server{
		Addr:              fmt.Sprintf(":%d", config.Port),
		Handler:           h2c.NewHandler(service.router, h2Server),
		ReadTimeout:       time.Duration(ReadTimeout) * time.Second,
		ReadHeaderTimeout: time.Duration(ReadHeaderTimeout) * time.Second,
		WriteTimeout:      time.Duration(WriteTimeout) * time.Second,
		IdleTimeout:       time.Duration(IdleTimeout) * time.Second,
		MaxHeaderBytes:    (1 << 20),
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	if err := service.Start(); err != nil {
		log.Fatalf("Failed to start API service: %v", err)
	}

	sig := <-sigChan
	log.Printf("kill signal %v, shutting down!", sig)

	_, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	forcedExitTimer := time.AfterFunc(40*time.Second, func() {
		log.Println("Forcing exit after 40s shutdown grace period")
		os.Exit(1)
	})
	defer forcedExitTimer.Stop()

	service.Shutdown()

	log.Printf("Server shutdown complete. Total uptime: %v", time.Since(startTime))
}
