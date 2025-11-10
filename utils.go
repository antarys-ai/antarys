package antarys

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	onnxruntime "github.com/yalue/onnxruntime_go"
	"golang.org/x/term"
)

const VERSION string = "0.2.0"
const ANTARYS_TITLE = `
              __
.---.-.-----.|  |_.---.-.----.--.--.-----.
|  _  |     ||   _|  _  |   _|  |  |__ --|
|___._|__|__||____|___._|__| |___  |_____|
                             |_____|
`

func AntarysTitle() {
	width, _, err := term.GetSize(int(os.Stdout.Fd()))
	if err != nil {
		width = 80
	}

	lines := strings.Split(ANTARYS_TITLE, "\n")
	for _, line := range lines {
		if len(line) > width {
			line = line[:width]
		}
		fmt.Printf("\033[32m%s\033[0m\n", line)
	}
}

func euclideanDistanceSquared(a, b []float32) float32 {
	if len(a) != len(b) {
		return math.MaxFloat32
	}

	var sumSq float64
	length := len(a)
	i := 0
	for ; i < length-7; i += 8 {
		d0 := float64(a[i]) - float64(b[i])
		d1 := float64(a[i+1]) - float64(b[i+1])
		d2 := float64(a[i+2]) - float64(b[i+2])
		d3 := float64(a[i+3]) - float64(b[i+3])
		d4 := float64(a[i+4]) - float64(b[i+4])
		d5 := float64(a[i+5]) - float64(b[i+5])
		d6 := float64(a[i+6]) - float64(b[i+6])
		d7 := float64(a[i+7]) - float64(b[i+7])
		sumSq += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 + d7*d7
	}

	for ; i < length; i++ {
		diff := float64(a[i]) - float64(b[i])
		sumSq += diff * diff
	}

	return float32(sumSq)
}

func kMeansPlusPlusInit(vectors [][]float32, k int) [][]float32 {
	n := len(vectors)
	dims := len(vectors[0])

	centroids := make([][]float32, k)
	for i := range centroids {
		centroids[i] = make([]float32, dims)
	}

	firstIdx := rand.Intn(n)
	copy(centroids[0], vectors[firstIdx])

	for c := 1; c < k; c++ {

		distances := make([]float32, n)
		sumDistances := float32(0)

		for i, vec := range vectors {
			minDist := float32(math.MaxFloat32)

			for j := 0; j < c; j++ {
				dist := euclideanDistanceSquared(vec, centroids[j])
				if dist < minDist {
					minDist = dist
				}
			}

			distances[i] = minDist
			sumDistances += minDist
		}

		target := rand.Float32() * sumDistances
		cumSum := float32(0)

		for i, dist := range distances {
			cumSum += dist
			if cumSum >= target {
				copy(centroids[c], vectors[i])
				break
			}
		}
	}

	return centroids
}

func flattenCentroids(centroids [][]float32) []float32 {
	k := len(centroids)
	if k == 0 {
		return []float32{}
	}

	dims := len(centroids[0])
	flat := make([]float32, k*dims)

	for i, centroid := range centroids {
		copy(flat[i*dims:], centroid)
	}

	return flat
}

func getCandidateSlice(capacity int) []queueItem {
	if capacity <= 512 {
		return candidatePool.Get().([]queueItem)
	}
	return make([]queueItem, 0, capacity)
}

func putCandidateSlice(s []queueItem) {
	if cap(s) <= 512 {
		candidatePool.Put(s[:0])
	}
}

func sortResultsFast(results []SearchResult) {
	if len(results) < 2 {
		return
	}

	if len(results) < 20 {
		for i := 1; i < len(results); i++ {
			for j := i; j > 0 && results[j].Score > results[j-1].Score; j-- {
				results[j], results[j-1] = results[j-1], results[j]
			}
		}
		return
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}

func sortCandidates(candidates []queueItem) {
	if len(candidates) < 2 {
		return
	}

	if len(candidates) < 20 {
		for i := 1; i < len(candidates); i++ {
			for j := i; j > 0 && candidates[j].similarity > candidates[j-1].similarity; j-- {
				candidates[j], candidates[j-1] = candidates[j-1], candidates[j]
			}
		}
		return
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].similarity > candidates[j].similarity
	})
}

var ValidChecksumFiles map[string]string = map[string]string{
	"libonnxruntime.1.22.0.dylib": "db045368293215c9d22aa7b8c983d688b3ae9ca1da3f64ffbe01ba7df31c3355",
	"libonnxruntime_arm64.so":     "0afd69a0ae38c5099fd0e8604dda398ac43dee67cd9c6394b5142b19e82528de",
	"libonnxruntime_x64.so":       "3da6146e14e7b8aaec625dde11d6114c7457c87a5f93d744897da8781e35c673",
}

var ModelsByIndex = map[int]string{
	1: "fast-bge-base-en",
	2: "fast-bge-base-en-v1.5",
	3: "fast-bge-small-en",
	4: "fast-bge-small-en-v1.5",
	5: "fast-bge-small-zh-v1.5",
}

func LoadDynamicOnnx() (string, error) {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.1.22.0.dylib", nil
	case "linux":
		switch runtime.GOARCH {
		case "arm64":
			return "libonnxruntime_arm64.so", nil
		case "amd64":
			return "libonnxruntime_x64.so", nil
		default:
			return "", fmt.Errorf("unsupported Linux architecture: %s. Embedding is only supported on ARM64 and x64", runtime.GOARCH)
		}
	case "windows":
		return "", fmt.Errorf("embedding is not supported on Windows")
	default:
		return "", fmt.Errorf("unsupported operating system: %s. Embedding is only supported on macOS and Linux", runtime.GOOS)
	}
}

func ValidateChecksum(path string, expected string) bool {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		panic(err)
	}

	actual := hex.EncodeToString(hash.Sum(nil))
	return actual == expected
}

func IsOnnxValid(cacheDir string) bool {

	for filename, expectedChecksum := range ValidChecksumFiles {

		if filepath.Ext(filename) == ".gz" {
			continue
		}

		filePath := filepath.Join(cacheDir, filename)

		if _, err := os.Stat(filePath); os.IsNotExist(err) {

			continue
		}

		if !ValidateChecksum(filePath, expectedChecksum) {
			fmt.Printf("Checksum validation failed for %s\n", filename)
			return false
		}
	}

	return true
}

func IsModelValid(id string, cacheDir string) bool {
	modelDirs := map[string]string{
		"1": "fast-bge-base-en",
		"2": "fast-bge-base-en-v1.5",
		"3": "fast-bge-small-en",
		"4": "fast-bge-small-en-v1.5",
		"5": "fast-bge-small-zh-v1.5",
	}

	modelDir, exists := modelDirs[id]
	if !exists {
		fmt.Printf("Unknown model ID: %s\n", id)
		return false
	}

	modelPath := filepath.Join(cacheDir, modelDir)

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("Model directory does not exist: %s\n", modelPath)
		return false
	}

	requiredFiles := []string{
		"config.json",
		"model_optimized.onnx",
		"ort_config.json",
		"special_tokens_map.json",
		"tokenizer_config.json",
		"tokenizer.json",
		"vocab.txt",
	}

	for _, filename := range requiredFiles {
		filePath := filepath.Join(modelPath, filename)
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			fmt.Printf("Required file missing: %s\n", filePath)
			return false
		}
	}

	onnxModelPath := filepath.Join(modelPath, "model_optimized.onnx")

	if !onnxruntime.IsInitialized() {

		runtimeLib := filepath.Join(cacheDir, "libonnxruntime.1.22.0.dylib")
		onnxruntime.SetSharedLibraryPath(runtimeLib)

		err := onnxruntime.InitializeEnvironment()

		if err != nil {
			fmt.Printf("Failed to initialize ONNX runtime: %v\n", err)
			return false
		}

		defer func() {
			if err := onnxruntime.DestroyEnvironment(); err != nil {
				fmt.Printf("Warning: Failed to cleanup ONNX environment: %v\n", err)
			}
		}()
	}

	_, _, err := onnxruntime.GetInputOutputInfo(onnxModelPath)
	if err != nil {
		fmt.Printf("Failed to load ONNX model %s: %v\n", onnxModelPath, err)
		return false
	}

	metadata, err := onnxruntime.GetModelMetadata(onnxModelPath)
	if err != nil {
		fmt.Printf("Failed to get model metadata for %s: %v\n", onnxModelPath, err)
		return false
	}
	defer metadata.Destroy()

	fmt.Printf("Model %s validation successful\n", id)
	return true
}

func ValidateOnnxAndModel(modelID string, cacheDir string) error {

	if !IsOnnxValid(cacheDir) {
		return fmt.Errorf("ONNX runtime validation failed")
	}

	if !IsModelValid(modelID, cacheDir) {
		return fmt.Errorf("model %s validation failed", modelID)
	}

	return nil
}
