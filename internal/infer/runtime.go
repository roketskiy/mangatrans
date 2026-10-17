package infer

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Runtime owns the process-wide ONNX Runtime environment. Only one may exist.
type Runtime struct {
	mu     sync.Mutex
	inited bool
	ep     string
	err    error
}

var (
	global     *Runtime
	globalOnce sync.Once
)

// Env returns the shared runtime, initializing it on first use.
func Env(ortDir, cudaDir string) (*Runtime, error) {
	globalOnce.Do(func() {
		global = &Runtime{}
		global.err = global.init(ortDir, cudaDir)
	})
	return global, global.err
}

func (r *Runtime) init(ortDir, cudaDir string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.inited {
		return nil
	}

	// The ORT provider DLLs and the CUDA runtime must both be on the loader path.
	if cudaDir != "" {
		os.Setenv("PATH", ortDir+";"+cudaDir+";"+os.Getenv("PATH"))
	} else {
		os.Setenv("PATH", ortDir+";"+os.Getenv("PATH"))
	}

	dll := filepath.Join(ortDir, "onnxruntime.dll")
	if _, err := os.Stat(dll); err != nil {
		return fmt.Errorf("onnxruntime.dll not found at %s", dll)
	}
	ort.SetSharedLibraryPath(dll)
	ort.SetEnvironmentLogLevel(ort.LoggingLevelError)

	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize onnxruntime: %w", err)
	}
	r.inited = true
	return nil
}

// Destroy releases the environment. Call once at exit.
func (r *Runtime) Destroy() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.inited {
		return nil
	}
	r.inited = false
	return ort.DestroyEnvironment()
}

// SetProvider records which provider a session actually got.
func (r *Runtime) SetProvider(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ep = name
}

// Provider returns the last recorded provider name.
func (r *Runtime) Provider() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.ep
}

// CUDAOptions returns session options with the CUDA EP appended, or CPU-only
// options when cuda is false.
func CUDAOptions(cuda bool, vramLimit int64) (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		opts.Destroy()
		return nil, err
	}
	if !cuda {
		return opts, nil
	}

	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		opts.Destroy()
		return nil, err
	}
	defer cudaOpts.Destroy()

	if vramLimit <= 0 {
		vramLimit = 2 << 30
	}
	err = cudaOpts.Update(map[string]string{
		"device_id":                 "0",
		"arena_extend_strategy":     "kSameAsRequested",
		"gpu_mem_limit":             fmt.Sprintf("%d", vramLimit),
		"cudnn_conv_algo_search":    "HEURISTIC",
		"do_copy_in_default_stream": "1",
	})
	if err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("set cuda options: %w", err)
	}
	if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		opts.Destroy()
		return nil, fmt.Errorf("append cuda ep: %w", err)
	}
	return opts, nil
}
