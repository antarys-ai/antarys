package antarys

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"

	"github.com/schollz/progressbar/v3"
)

type OnnxRuntime string

const (
	ORT_MAC_UNIVERSAL OnnxRuntime = "libonnxruntime.1.22.0.dylib"
)

func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil || !os.IsNotExist(err)
}

func HasOnnxRuntime(runtime OnnxRuntime, cacheDir string) bool {
	return FileExists(filepath.Join(cacheDir, string(runtime)))
}

func RetrieveRuntime(runtime OnnxRuntime, cacheDir string, showDownloadProgress bool) (string, error) {
	if _, err := os.Stat(filepath.Join(cacheDir, string(runtime))); !errors.Is(err, fs.ErrNotExist) {
		return filepath.Join(cacheDir, string(runtime)), nil
	}
	return downloadOnnx(runtime, cacheDir, showDownloadProgress)
}

func downloadOnnx(runtime OnnxRuntime, cacheDir string, showProgress bool) (string, error) {
	downloadUrl := fmt.Sprintf("https://github.com/antarys-ai/antarys-releases/releases/download/embed/%s", runtime)
	response, err := http.Get(downloadUrl)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode > 299 {
		return "", fmt.Errorf("runtime download failed: %s", response.Status)
	}

	localFilePath := filepath.Join(cacheDir, string(runtime))

	outputFile, err := os.Create(localFilePath)
	if err != nil {
		return "", err
	}
	defer outputFile.Close()

	var reader io.Reader = response.Body

	if showProgress {

		bar := progressbar.DefaultBytes(
			response.ContentLength,
			"Downloading "+string(runtime),
		)

		reader = io.TeeReader(response.Body, bar)
	}

	_, err = io.Copy(outputFile, reader)
	if err != nil {
		return "", err
	}

	fmt.Printf("\nDownload complete! File saved to: %s\n", localFilePath)

	return localFilePath, nil
}
