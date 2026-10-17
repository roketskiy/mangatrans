package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	ortDir   = `E:\work\transl\runtime\ort-cuda13`
	cudaDir  = `E:\Python312\Lib\site-packages\torch\lib`
	modelCtd = `E:\work\transl\.cache\onnx-probe\comictextdetector.onnx`
)

func main() {
	os.Setenv("PATH", ortDir+";"+cudaDir+";"+os.Getenv("PATH"))
	ort.SetSharedLibraryPath(filepath.Join(ortDir, "onnxruntime.dll"))
	ort.SetEnvironmentLogLevel(ort.LoggingLevelInfo)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Println("ENV_FAIL", err)
		os.Exit(2)
	}
	defer ort.DestroyEnvironment()

	cudaMs := bench("CUDAExecutionProvider", true)
	cpuMs := bench("CPUExecutionProvider", false)
	fmt.Printf("RESULT cuda_median_ms=%.1f cpu_median_ms=%.1f speedup=%.1fx\n",
		cudaMs, cpuMs, cpuMs/cudaMs)
}

func bench(tag string, useCUDA bool) float64 {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		fmt.Println(tag, "OPTS_FAIL", err)
		os.Exit(3)
	}
	defer opts.Destroy()
	opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	if useCUDA {
		cuda, _ := ort.NewCUDAProviderOptions()
		defer cuda.Destroy()
		cuda.Update(map[string]string{
			"device_id":                 "0",
			"arena_extend_strategy":     "kSameAsRequested",
			"gpu_mem_limit":             "2147483648",
			"cudnn_conv_algo_search":    "HEURISTIC",
			"do_copy_in_default_stream": "1",
		})
		if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
			fmt.Println(tag, "CUDA_APPEND_FAIL", err)
			os.Exit(4)
		}
	}
	in, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, 1024, 1024))
	defer in.Destroy()
	blk, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 64512, 7))
	defer blk.Destroy()
	seg, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, 1024, 1024))
	defer seg.Destroy()
	det, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 2, 1024, 1024))
	defer det.Destroy()

	sess, err := ort.NewAdvancedSession(modelCtd,
		[]string{"images"}, []string{"blk", "seg", "det"},
		[]ort.Value{in}, []ort.Value{blk, seg, det}, opts)
	if err != nil {
		fmt.Println(tag, "SESSION_FAIL", err)
		os.Exit(5)
	}
	defer sess.Destroy()

	for i := 0; i < 2; i++ {
		sess.Run()
	}
	var best float64 = 1e9
	for i := 0; i < 5; i++ {
		t := time.Now()
		if err := sess.Run(); err != nil {
			fmt.Println(tag, "RUN_FAIL", err)
			os.Exit(6)
		}
		ms := float64(time.Since(t).Microseconds()) / 1000
		if ms < best {
			best = ms
		}
	}
	fmt.Printf("%-22s best_ms=%.1f\n", tag, best)
	return best
}
