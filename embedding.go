// taken from https://github.com/Anush008/fastembed-go
package antarys

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sync"

	"github.com/schollz/progressbar/v3"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

type EmbeddingModel string

const (
	BGEBaseEN     EmbeddingModel = "fast-bge-base-en"
	BGEBaseENV15  EmbeddingModel = "fast-bge-base-en-v1.5"
	BGESmallEN    EmbeddingModel = "fast-bge-small-en"
	BGESmallENV15 EmbeddingModel = "fast-bge-small-en-v1.5"
	BGESmallZH    EmbeddingModel = "fast-bge-small-zh-v1.5"
)

type FlagEmbedding struct {
	tokenizer *tokenizer.Tokenizer
	model     EmbeddingModel
	maxLength int
	modelPath string
}

type InitOptions struct {
	Model                EmbeddingModel
	ExecutionProviders   []string
	MaxLength            int
	CacheDir             string
	ShowDownloadProgress *bool
}

type ModelInfo struct {
	Model       EmbeddingModel
	Dim         int
	Description string
}

func NewFlagEmbedding(options *InitOptions) (*FlagEmbedding, error) {
	if options == nil {
		options = &InitOptions{}
	}

	if options.CacheDir == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			log.Print("could not create cache directory for Antarys")
			return nil, err
		}

		defaultCache := filepath.Join(homeDir, ".antarys")
		if err := os.MkdirAll(defaultCache, 0755); err != nil {
			log.Print("could not create cache directory for Antarys")
			return nil, err
		}

		options.CacheDir = defaultCache
	}

	if options.Model == "" {
		options.Model = BGESmallENV15
	}

	if options.MaxLength == 0 {
		options.MaxLength = 512
	}

	if options.ShowDownloadProgress == nil {
		showDownloadProgress := true
		options.ShowDownloadProgress = &showDownloadProgress
	}

	if onnxPath := os.Getenv("ONNX_PATH"); onnxPath != "" {
		ort.SetSharedLibraryPath(onnxPath)
	}

	if !ort.IsInitialized() {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, err
		}
	}

	modelPath, err := RetrieveModel(options.Model, options.CacheDir, *options.ShowDownloadProgress)
	if err != nil {
		return nil, err
	}

	tknzer, err := loadTokenizer(modelPath, options.MaxLength)
	if err != nil {
		return nil, err
	}
	return &FlagEmbedding{
		tokenizer: tknzer,
		model:     options.Model,
		maxLength: options.MaxLength,
		modelPath: modelPath,
	}, nil
}

func (f *FlagEmbedding) Destroy() error {
	return ort.DestroyEnvironment()
}

func (f *FlagEmbedding) onnxEmbed(input []string) ([]([]float32), error) {
	inputs := make([]tokenizer.EncodeInput, len(input))
	for index, v := range input {
		sequence := tokenizer.NewInputSequence(v)
		inputs[index] = tokenizer.NewSingleEncodeInput(sequence)
	}

	encodings, err := f.tokenizer.EncodeBatch(inputs, true)
	if err != nil {
		return nil, err
	}

	inputIdsFlat, inputMaskFlat, inputTypeIdsFlat := make([]int64, 0), make([]int64, 0), make([]int64, 0)
	for _, encoding := range encodings {
		inputIds, inputMask, inputTypeIds := encodingToInt32(encoding.GetIds(), encoding.GetAttentionMask(), encoding.GetTypeIds())
		inputIdsFlat = append(inputIdsFlat, inputIds...)
		inputMaskFlat = append(inputMaskFlat, inputMask...)
		inputTypeIdsFlat = append(inputTypeIdsFlat, inputTypeIds...)
	}

	inputShape := ort.NewShape(int64(len(inputs)), int64(encodings[0].Len()))

	inputTensorID, err := ort.NewTensor(inputShape, inputIdsFlat)
	if err != nil {
		return nil, err
	}
	defer inputTensorID.Destroy()

	inputTensorMask, err := ort.NewTensor(inputShape, inputMaskFlat)

	if err != nil {
		return nil, err
	}
	defer inputTensorMask.Destroy()

	inputTensorType, err := ort.NewTensor(inputShape, inputTypeIdsFlat)

	if err != nil {
		return nil, err
	}
	defer inputTensorType.Destroy()

	modelInfo, err := getModelInfo(f.model)
	if err != nil {
		return nil, err
	}

	outputShape := ort.NewShape(int64(len(inputs)), int64(int64(encodings[0].Len())), int64(modelInfo.Dim))
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(filepath.Join(f.modelPath, "model_optimized.onnx"), []string{
		"input_ids", "attention_mask", "token_type_ids",
	}, []string{
		"last_hidden_state",
	}, []ort.ArbitraryTensor{
		inputTensorID, inputTensorMask, inputTensorType,
	}, []ort.ArbitraryTensor{outputTensor},
		nil)

	if err != nil {
		return nil, err
	}

	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, err
	}

	return getEmbeddings(outputTensor.GetData(), outputTensor.GetShape()), nil
}

func (f *FlagEmbedding) Embed(input []string, batchSize int) ([]([]float32), error) {
	if batchSize <= 0 {
		batchSize = 256
	}
	embeddings := make([]([]float32), len(input))
	var wg sync.WaitGroup
	errorCh := make(chan error, len(input))

	for i := 0; i < len(input); i += batchSize {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			end := i + batchSize
			if end > len(input) {
				end = len(input)
			}
			batchOut, err := f.onnxEmbed(input[i:end])
			if err != nil {
				errorCh <- err
			}

			copy(embeddings[i:end], batchOut)
		}(i)
	}
	wg.Wait()
	close(errorCh)

	if len(errorCh) > 0 {
		return nil, <-errorCh
	}
	return embeddings, nil
}

func (f *FlagEmbedding) QueryEmbed(input string) ([]float32, error) {
	query := "query: " + input
	data, err := f.onnxEmbed([]string{query})
	if err != nil {
		return nil, err
	}
	return data[0], nil
}

func (f *FlagEmbedding) PassageEmbed(input []string, batchSize int) ([]([]float32), error) {
	processedInput := make([]string, len(input))
	for i, v := range input {
		processedInput[i] = "passage: " + v
	}
	return f.Embed(processedInput, batchSize)
}

func ListSupportedModels() []ModelInfo {
	return []ModelInfo{
		{
			Model:       BGEBaseEN,
			Dim:         768,
			Description: "Base English model",
		},
		{
			Model:       BGEBaseENV15,
			Dim:         768,
			Description: "v1.5 release of the base English model",
		},
		{
			Model:       BGESmallEN,
			Dim:         384,
			Description: "Fast English model",
		},
		{
			Model:       BGESmallENV15,
			Dim:         384,
			Description: "Fast, default English model",
		},
		{
			Model:       BGESmallZH,
			Dim:         512,
			Description: "Fast Chinese model",
		},
	}
}

func loadTokenizer(modelPath string, maxLength int) (*tokenizer.Tokenizer, error) {
	tknzer, err := pretrained.FromFile(filepath.Join(modelPath, "tokenizer.json"))

	if err != nil {
		return nil, err
	}

	configData, err := os.ReadFile(filepath.Join(modelPath, "config.json"))
	if err != nil {
		return nil, err
	}

	var config map[string]interface{}
	err = json.Unmarshal(configData, &config)

	if err != nil {
		return nil, err
	}

	tokenizerConfigData, err := os.ReadFile(filepath.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return nil, err
	}

	var tokenizerConfig map[string]interface{}
	err = json.Unmarshal(tokenizerConfigData, &tokenizerConfig)

	if err != nil {
		return nil, err
	}

	tokensMapData, err := os.ReadFile(filepath.Join(modelPath, "special_tokens_map.json"))
	if err != nil {
		return nil, err
	}

	var tokensMap map[string]interface{}
	err = json.Unmarshal(tokensMapData, &tokensMap)

	if err != nil {
		return nil, err
	}

	modelMaxLen := int(min(float64(math.MaxInt32), math.Abs(tokenizerConfig["model_max_length"].(float64))))
	maxLength = min(maxLength, modelMaxLen)

	tknzer.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxLength,
		Strategy:  tokenizer.LongestFirst,
		Stride:    0,
	})

	paddingParams := tokenizer.PaddingParams{

		Strategy:  *tokenizer.NewPaddingStrategy(),
		Direction: tokenizer.Right,
		PadId:     int(config["pad_token_id"].(float64)),
		PadToken:  tokenizerConfig["pad_token"].(string),
		PadTypeId: 0,
	}
	tknzer.WithPadding(&paddingParams)

	specialTokens := make([]tokenizer.AddedToken, 0)

	for _, v := range tokensMap {
		switch t := v.(type) {
		case map[string]interface{}:
			{
				specialToken := tokenizer.AddedToken{
					Content:    t["content"].(string),
					SingleWord: t["single_word"].(bool),
					LStrip:     t["lstrip"].(bool),
					RStrip:     t["rstrip"].(bool),
					Normalized: t["normalized"].(bool),
				}
				specialTokens = append(specialTokens, specialToken)
			}
		case string:
			specialToken := tokenizer.AddedToken{
				Content: t,
			}
			specialTokens = append(specialTokens, specialToken)
		default:
			panic(fmt.Sprintf("unknown type for special_tokens_map.json%T", t))
		}
	}
	tknzer.AddSpecialTokens(specialTokens)

	return tknzer, nil
}

func getModelInfo(model EmbeddingModel) (ModelInfo, error) {
	for _, m := range ListSupportedModels() {
		if m.Model == model {
			return m, nil
		}
	}
	return ModelInfo{}, fmt.Errorf("model %s not found", model)
}

func HasModel(model EmbeddingModel, cacheDir string) bool {
	return FileExists(filepath.Join(cacheDir, string(model)))
}

func RetrieveModel(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {
	if _, err := os.Stat(filepath.Join(cacheDir, string(model))); !errors.Is(err, fs.ErrNotExist) {
		return filepath.Join(cacheDir, string(model)), nil
	}
	return downloadModel(model, cacheDir, showDownloadProgress)
}

func downloadModel(model EmbeddingModel, cacheDir string, showDownloadProgress bool) (string, error) {

	downloadURL := fmt.Sprintf(
		"https://github.com/antarys-ai/antarys-releases/releases/download/embed/%s.tar.gz",
		model,
	)

	response, err := http.Get(downloadURL)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return "", fmt.Errorf("model download failed: %s", response.Status)
	}

	if showDownloadProgress {
		bar := progressbar.DefaultBytes(
			response.ContentLength,
			"Downloading "+string(model),
		)
		reader := progressbar.NewReader(response.Body, bar)
		err = untar(&reader, cacheDir)
	} else {
		fmt.Printf("Downloading %s...", model)
		err = untar(response.Body, cacheDir)
	}

	if err != nil {
		return "", err
	}

	return filepath.Join(cacheDir, string(model)), nil
}

func untar(tarball io.Reader, target string) error {
	archive, err := gzip.NewReader(tarball)
	if err != nil {
		return err
	}
	defer archive.Close()

	tarReader := tar.NewReader(archive)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		path := filepath.Join(target, header.Name)
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(path, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
				return err
			}

			file, err := os.Create(path)
			if err != nil {
				return err
			}
			defer file.Close()
			if _, err := io.Copy(file, tarReader); err != nil {
				return err
			}
		}
	}
	return nil
}

func normalize(v []float32) []float32 {
	norm := float32(0.0)
	for _, val := range v {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))
	epsilon := float32(1e-12)

	normalized := make([]float32, len(v))
	for i, val := range v {
		normalized[i] = (val / norm) + epsilon
	}

	return normalized
}

func getEmbeddings(data []float32, dimensions []int64) []([]float32) {
	x, y, z := dimensions[0], dimensions[1], dimensions[2]
	embeddings := make([][]float32, x)
	var i int64
	for i = 0; i < x; i++ {
		startIndex := i * y * z
		endIndex := startIndex + z
		embeddings[i] = normalize(data[startIndex:endIndex])
	}
	return embeddings
}

func encodingToInt32(inputA, inputB, inputC []int) ([]int64, []int64, []int64) {
	if len(inputA) != len(inputB) || len(inputB) != len(inputC) {
		panic("input lengths do not match")
	}
	outputA := make([]int64, len(inputA))
	outputB := make([]int64, len(inputB))
	outputC := make([]int64, len(inputC))
	for i := range inputA {
		outputA[i] = int64(inputA[i])
		outputB[i] = int64(inputB[i])
		outputC[i] = int64(inputC[i])
	}
	return outputA, outputB, outputC
}
