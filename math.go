package antarys

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/gonum/floats"
)

var (
	simdWidth int = 8

	invSqrtLUT [1024]float32
	cpuInit    sync.Once

	vectorPool = sync.Pool{
		New: func() interface{} {
			return make([]float32, 0, 4096)
		},
	}

	resultPool = sync.Pool{
		New: func() interface{} {
			return make([]SearchResult, 0, 1000)
		},
	}

	blasImpl blas.Float32

	vectorPool1536 = sync.Pool{
		New: func() interface{} {
			return make([]float32, 1536)
		},
	}
	vectorPool768 = sync.Pool{
		New: func() interface{} {
			return make([]float32, 768)
		},
	}
	vectorPool512 = sync.Pool{
		New: func() interface{} {
			return make([]float32, 512)
		},
	}
	vectorPool256 = sync.Pool{
		New: func() interface{} {
			return make([]float32, 256)
		},
	}
	vectorPool128 = sync.Pool{
		New: func() interface{} {
			return make([]float32, 128)
		},
	}
	vectorPoolGeneric = sync.Pool{
		New: func() interface{} {
			return make([]float32, 0, 2048)
		},
	}

	mathStats struct {
		totalOperations    int64
		fastPathOps        int64
		blasOperations     int64
		precisionFallbacks int64
		vectorOperations   int64
	}

	mr = struct {
		src *rand.Source
		mu  sync.Mutex
		rng *rand.Rand
	}{
		src: func() *rand.Source {
			s := rand.NewSource(time.Now().UnixNano())
			return &s
		}(),
	}
)

func init() {
	mr.rng = rand.New(*mr.src)
	cpuInit.Do(detectCPUFeatures)

	blasImpl = blas32.Implementation()
}

func detectCPUFeatures() {

	numCPU := runtime.NumCPU()
	if numCPU >= 16 {
		simdWidth = 32
	} else if numCPU >= 8 {
		simdWidth = 16
	} else if numCPU >= 4 {
		simdWidth = 8
	} else {
		simdWidth = 4
	}

	initLookupTable()
}

func initLookupTable() {

	for i := 0; i < len(invSqrtLUT); i++ {
		x := float32(i) / float32(len(invSqrtLUT)-1) * 16.0
		if x > 0 {
			invSqrtLUT[i] = 1.0 / float32(math.Sqrt(float64(x)))
		} else {
			invSqrtLUT[i] = 0
		}
	}
}

func getVector(size int) []float32 {

	switch size {
	case 1536:
		return vectorPool1536.Get().([]float32)
	case 768:
		return vectorPool768.Get().([]float32)
	case 512:
		return vectorPool512.Get().([]float32)
	case 256:
		return vectorPool256.Get().([]float32)
	case 128:
		return vectorPool128.Get().([]float32)
	default:
		vec := vectorPool.Get().([]float32)
		if cap(vec) < size {
			return make([]float32, size)
		}
		return vec[:size]
	}
}

func putVector(vec []float32) {
	size := len(vec)
	switch size {
	case 1536:
		vectorPool1536.Put(vec)
	case 768:
		vectorPool768.Put(vec)
	case 512:
		vectorPool512.Put(vec)
	case 256:
		vectorPool256.Put(vec)
	case 128:
		vectorPool128.Put(vec)
	default:
		if cap(vec) <= 4096 {
			vectorPool.Put(vec[:0])
		}
	}
}

func getResultSlice(size int) []SearchResult {
	results := resultPool.Get().([]SearchResult)
	if cap(results) < size {
		return make([]SearchResult, 0, size)
	}
	return results[:0]
}

func putResultSlice(results []SearchResult) {
	if cap(results) <= 1000 {
		resultPool.Put(results[:0])
	}
}

func normaliseVector(vector []float32) {
	n := len(vector)
	if n == 0 {
		return
	}

	atomic.AddInt64(&mathStats.totalOperations, 1)
	atomic.AddInt64(&mathStats.vectorOperations, 1)

	if n <= 2048 {

		atomic.AddInt64(&mathStats.fastPathOps, 1)
		normalizeVectorFast(vector)
		return
	}

	atomic.AddInt64(&mathStats.blasOperations, 1)
	normalizeVectorHighPrecision(vector)
}

func normalizeVectorFast(vector []float32) {
	n := len(vector)
	var sum float32

	switch n {
	case 1536:
		sum = normSquared1536(vector)
	case 768:
		sum = normSquared768(vector)
	case 512:
		sum = normSquared512(vector)
	case 256:
		sum = normSquared256(vector)
	case 128:
		sum = normSquared128(vector)
	default:
		sum = normSquaredGeneric(vector)
	}

	if sum <= 1e-10 {
		return
	}

	invNorm := fastInvSqrt(sum)

	switch n {
	case 1536:
		applyNorm1536(vector, invNorm)
	case 768:
		applyNorm768(vector, invNorm)
	case 512:
		applyNorm512(vector, invNorm)
	case 256:
		applyNorm256(vector, invNorm)
	case 128:
		applyNorm128(vector, invNorm)
	default:
		applyNormGeneric(vector, invNorm)
	}
}

func normSquared1536(vector []float32) float32 {
	var sum float32

	for i := 0; i < 1536; i += 8 {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}
	return sum
}

func normSquared768(vector []float32) float32 {
	var sum float32
	for i := 0; i < 768; i += 8 {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}
	return sum
}

func normSquared512(vector []float32) float32 {
	var sum float32
	for i := 0; i < 512; i += 8 {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}
	return sum
}

func normSquared256(vector []float32) float32 {
	var sum float32
	for i := 0; i < 256; i += 8 {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}
	return sum
}

func normSquared128(vector []float32) float32 {
	var sum float32
	for i := 0; i < 128; i += 8 {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
	}
	return sum
}

func normSquaredGeneric(vector []float32) float32 {
	n := len(vector)
	var sum float32
	i := 0
	for i+7 < n {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
		i += 8
	}
	for i < n {
		sum += vector[i] * vector[i]
		i++
	}
	return sum
}

func applyNorm1536(vector []float32, invNorm float32) {
	for i := 0; i < 1536; i += 8 {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
	}
}

func applyNorm768(vector []float32, invNorm float32) {
	for i := 0; i < 768; i += 8 {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
	}
}

func applyNorm512(vector []float32, invNorm float32) {
	for i := 0; i < 512; i += 8 {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
	}
}

func applyNorm256(vector []float32, invNorm float32) {
	for i := 0; i < 256; i += 8 {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
	}
}

func applyNorm128(vector []float32, invNorm float32) {
	for i := 0; i < 128; i += 8 {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
	}
}

func applyNormGeneric(vector []float32, invNorm float32) {
	n := len(vector)
	i := 0
	for i+7 < n {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
		i += 8
	}
	for i < n {
		vector[i] *= invNorm
		i++
	}
}

func normalizeVectorHighPrecision(vector []float32) {
	n := len(vector)

	vec64 := make([]float64, n)
	for i, v := range vector {
		vec64[i] = float64(v)
	}

	norm := floats.Norm(vec64, 2)

	if norm <= 1e-14 {
		return
	}

	invNorm := float32(1.0 / norm)

	blasImpl.Sscal(n, invNorm, vector, 1)
}

func NormaliseVector(vector []float32) {
	normaliseVector(vector)
}

func fastInvSqrt(x float32) float32 {
	if x <= 0 {
		return 0
	}

	if x < 1e-6 {
		atomic.AddInt64(&mathStats.precisionFallbacks, 1)
		return 1.0 / float32(math.Sqrt(float64(x)))
	}

	if x > 16.0 {
		atomic.AddInt64(&mathStats.precisionFallbacks, 1)
		return 1.0 / float32(math.Sqrt(float64(x)))
	}

	idx := int(x * float32(len(invSqrtLUT)-1) / 16.0)
	if idx >= len(invSqrtLUT) {
		idx = len(invSqrtLUT) - 1
	}

	return invSqrtLUT[idx]
}

func cosineSimilarity(a, b []float32) float32 {
	n := len(a)
	if n != len(b) || n == 0 {
		return 0
	}

	atomic.AddInt64(&mathStats.totalOperations, 1)

	if n <= 2048 {
		atomic.AddInt64(&mathStats.fastPathOps, 1)
		return cosineSimilarityFast(a, b)
	}

	atomic.AddInt64(&mathStats.blasOperations, 1)
	return cosineSimilarityBLAS(a, b)
}

func cosineSimilarityFast(a, b []float32) float32 {
	n := len(a)

	switch n {
	case 1536:
		return dotProduct1536(a, b)
	case 768:
		return dotProduct768(a, b)
	case 512:
		return dotProduct512(a, b)
	case 256:
		return dotProduct256(a, b)
	case 128:
		return dotProduct128(a, b)
	default:
		return dotProductGeneric(a, b)
	}
}

func dotProduct1536(a, b []float32) float32 {
	var sum float32

	for i := 0; i < 1536; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	return precisionClampFast(sum)
}

func dotProduct768(a, b []float32) float32 {
	var sum float32
	for i := 0; i < 768; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	return precisionClampFast(sum)
}

func dotProduct512(a, b []float32) float32 {
	var sum float32
	for i := 0; i < 512; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	return precisionClampFast(sum)
}

func dotProduct256(a, b []float32) float32 {
	var sum float32
	for i := 0; i < 256; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	return precisionClampFast(sum)
}

func dotProduct128(a, b []float32) float32 {
	var sum float32
	for i := 0; i < 128; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	return precisionClampFast(sum)
}

func dotProductGeneric(a, b []float32) float32 {
	n := len(a)
	var sum float32
	i := 0
	for i+7 < n {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
		i += 8
	}
	for i < n {
		sum += a[i] * b[i]
		i++
	}
	return precisionClampFast(sum)
}

func cosineSimilarityBLAS(a, b []float32) float32 {
	n := len(a)

	a64 := make([]float64, n)
	b64 := make([]float64, n)

	for i := 0; i < n; i++ {
		a64[i] = float64(a[i])
		b64[i] = float64(b[i])
	}

	dotProduct := floats.Dot(a64, b64)

	return precisionClampFast(float32(dotProduct))
}

func precisionClampFast(similarity float32) float32 {

	if math.IsNaN(float64(similarity)) || math.IsInf(float64(similarity), 0) {
		atomic.AddInt64(&mathStats.precisionFallbacks, 1)
		return 0.0
	}

	if similarity > 1.0 {
		return 1.0
	}
	if similarity < -1.0 {
		return -1.0
	}

	return similarity
}

func randomLevel(m int) int {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	level := 0
	for mr.rng.Float32() < 0.5 && level < 16 {
		level++
	}
	return level
}

func randomLevelFast(m int) int {
	mr.mu.Lock()
	defer mr.mu.Unlock()
	level := 0

	r := mr.rng.Uint32()
	for (r&1) == 1 && level < 16 {
		level++
		r >>= 1
	}
	return level
}

func NormaliseVectorLockless(vector []float32) {
	if len(vector) == 0 {
		return
	}

	n := len(vector)

	var sum float32

	i := 0
	for i+7 < n {
		v0, v1, v2, v3 := vector[i], vector[i+1], vector[i+2], vector[i+3]
		v4, v5, v6, v7 := vector[i+4], vector[i+5], vector[i+6], vector[i+7]
		sum += v0*v0 + v1*v1 + v2*v2 + v3*v3 + v4*v4 + v5*v5 + v6*v6 + v7*v7
		i += 8
	}

	for i < n {
		sum += vector[i] * vector[i]
		i++
	}

	if sum <= 1e-10 {
		return
	}

	invNorm := fastInvSqrt(sum)

	i = 0
	for i+7 < n {
		vector[i] *= invNorm
		vector[i+1] *= invNorm
		vector[i+2] *= invNorm
		vector[i+3] *= invNorm
		vector[i+4] *= invNorm
		vector[i+5] *= invNorm
		vector[i+6] *= invNorm
		vector[i+7] *= invNorm
		i += 8
	}

	for i < n {
		vector[i] *= invNorm
		i++
	}
}

func NormaliseVectors(vectors [][]float32) {

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	for _, vector := range vectors {
		wg.Add(1)
		go func(v []float32) {
			defer wg.Done()
			sem <- struct{}{}
			normaliseVector(v)
			<-sem
		}(vector)
	}

	wg.Wait()
}

type MathStats struct {
	TotalOperations    int64
	FastPathOps        int64
	BlasOperations     int64
	PrecisionFallbacks int64
	VectorOperations   int64
}

func GetMathStats() MathStats {
	return MathStats{
		TotalOperations:    atomic.LoadInt64(&mathStats.totalOperations),
		FastPathOps:        atomic.LoadInt64(&mathStats.fastPathOps),
		BlasOperations:     atomic.LoadInt64(&mathStats.blasOperations),
		PrecisionFallbacks: atomic.LoadInt64(&mathStats.precisionFallbacks),
		VectorOperations:   atomic.LoadInt64(&mathStats.vectorOperations),
	}
}

func ResetMathStats() {
	atomic.StoreInt64(&mathStats.totalOperations, 0)
	atomic.StoreInt64(&mathStats.fastPathOps, 0)
	atomic.StoreInt64(&mathStats.blasOperations, 0)
	atomic.StoreInt64(&mathStats.precisionFallbacks, 0)
	atomic.StoreInt64(&mathStats.vectorOperations, 0)
}

func TestPrecisionAccuracy() map[string]float64 {
	results := make(map[string]float64)

	testValues := []float32{0.001, 0.01, 0.1, 1.0, 4.0, 16.0}
	var totalError float64

	for _, val := range testValues {
		expected := 1.0 / float32(math.Sqrt(float64(val)))
		actual := fastInvSqrt(val)
		error := math.Abs(float64(expected - actual))
		totalError += error
	}

	results["avg_invsqrt_error"] = totalError / float64(len(testValues))

	a := make([]float32, 1536)
	b := make([]float32, 1536)

	for i := 0; i < 1536; i++ {
		a[i] = float32(i) / 1535.0
		b[i] = float32(1535-i) / 1535.0
	}

	normaliseVector(a)
	normaliseVector(b)

	similarityFast := cosineSimilarityFast(a, b)
	similarityBLAS := cosineSimilarityBLAS(a, b)

	results["test_similarity_fast"] = float64(similarityFast)
	results["test_similarity_blas"] = float64(similarityBLAS)
	results["fast_vs_blas_diff"] = math.Abs(float64(similarityFast - similarityBLAS))

	stats := GetMathStats()
	results["fast_path_operations"] = float64(stats.FastPathOps)
	results["blas_operations"] = float64(stats.BlasOperations)
	results["total_operations"] = float64(stats.TotalOperations)
	if stats.TotalOperations > 0 {
		results["fast_path_ratio"] = float64(stats.FastPathOps) / float64(stats.TotalOperations)
		results["blas_usage_ratio"] = float64(stats.BlasOperations) / float64(stats.TotalOperations)
	}

	return results
}

func BatchNormalizeVectors(vectors [][]float32) {

	numWorkers := runtime.NumCPU()
	if numWorkers > len(vectors) {
		numWorkers = len(vectors)
	}

	jobs := make(chan []float32, len(vectors))
	var wg sync.WaitGroup

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for vector := range jobs {
				normaliseVector(vector)
			}
		}()
	}

	for _, vector := range vectors {
		jobs <- vector
	}
	close(jobs)

	wg.Wait()
}

func BatchCosineSimilarity(queries [][]float32, targets [][]float32) [][]float32 {
	if len(queries) == 0 || len(targets) == 0 {
		return nil
	}

	numQueries := len(queries)
	numTargets := len(targets)
	results := make([][]float32, numQueries)

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())

	for i := 0; i < numQueries; i++ {
		results[i] = make([]float32, numTargets)
		wg.Add(1)

		go func(queryIdx int) {
			defer wg.Done()
			sem <- struct{}{}

			query := queries[queryIdx]
			for j, target := range targets {
				results[queryIdx][j] = cosineSimilarity(query, target)
			}

			<-sem
		}(i)
	}

	wg.Wait()
	return results
}

func MatrixVectorMultiply(matrix [][]float32, vector []float32) []float32 {
	if len(matrix) == 0 || len(matrix[0]) != len(vector) {
		return nil
	}

	rows := len(matrix)
	cols := len(matrix[0])

	if rows*cols >= 100000 {

		matData := make([]float64, rows*cols)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				matData[i*cols+j] = float64(matrix[i][j])
			}
		}

		vecData := make([]float64, len(vector))
		for i, v := range vector {
			vecData[i] = float64(v)
		}

		result := make([]float64, rows)
		blasImpl.Sgemv(blas.NoTrans, rows, cols, 1.0,
			(*[1 << 30]float32)(unsafe.Pointer(&matData[0]))[:len(matData)], cols,
			vector, 1, 0.0,
			(*[1 << 30]float32)(unsafe.Pointer(&result[0]))[:len(result)], 1)

		resultSlice := make([]float32, rows)
		for i := 0; i < rows; i++ {
			resultSlice[i] = float32(result[i])
		}

		atomic.AddInt64(&mathStats.blasOperations, 1)
		return resultSlice
	}

	atomic.AddInt64(&mathStats.fastPathOps, 1)
	result := make([]float32, rows)

	for i := 0; i < rows; i++ {
		var sum float32
		row := matrix[i]
		j := 0

		for j+7 < cols {
			sum += row[j]*vector[j] + row[j+1]*vector[j+1] +
				row[j+2]*vector[j+2] + row[j+3]*vector[j+3] +
				row[j+4]*vector[j+4] + row[j+5]*vector[j+5] +
				row[j+6]*vector[j+6] + row[j+7]*vector[j+7]
			j += 8
		}

		for j < cols {
			sum += row[j] * vector[j]
			j++
		}

		result[i] = sum
	}

	return result
}
