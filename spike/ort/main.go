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
	modelAot = `E:\work\transl\.cache\onnx-probe\aot.onnx`
)

func main() {
	// Both the ORT provider DLLs and the CUDA runtime must be on the loader path.
	os.Setenv("PATH", ortDir+";"+cudaDir+";"+os.Getenv("PATH"))

	ort.SetSharedLibraryPath(filepath.Join(ortDir, "onnxruntime.dll"))
	ort.SetEnvironmentLogLevel(ort.LoggingLevelWarning)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Println("ENV_FAIL", err)
		os.Exit(2)
	}
	defer ort.DestroyEnvironment()

	if !runAot() {
		os.Exit(3)
	}
	if !runCtd() {
		os.Exit(4)
	}
	fmt.Println("PASS")
}

func newOptions() (*ort.SessionOptions, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll)

	cuda, err := ort.NewCUDAProviderOptions()
	if err != nil {
		return nil, err
	}
	defer cuda.Destroy()
	err = cuda.Update(map[string]string{
		"device_id":                "0",
		"arena_extend_strategy":    "kSameAsRequested",
		"gpu_mem_limit":            "2147483648",
		"cudnn_conv_algo_search":   "HEURISTIC",
		"do_copy_in_default_stream": "1",
	})
	if err != nil {
		return nil, err
	}
	if err := opts.AppendExecutionProviderCUDA(cuda); err != nil {
		return nil, err
	}
	return opts, nil
}

func runAot() bool {
	fmt.Println("--- aot.onnx (LaMa-like, 512x512)")
	opts, err := newOptions()
	if err != nil {
		fmt.Println("OPTS_FAIL", err)
		return false
	}
	defer opts.Destroy()

	in, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, 512, 512))
	defer in.Destroy()
	mask, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, 512, 512))
	defer mask.Destroy()
	out, _ := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, 512, 512))
	defer out.Destroy()

	sess, err := ort.NewAdvancedSession(modelAot,
		[]string{"image", "mask"}, []string{"inpainted"},
		[]ort.Value{in, mask}, []ort.Value{out}, opts)
	if err != nil {
		fmt.Println("SESSION_FAIL", err)
		return false
	}
	defer sess.Destroy()

	times := make([]float64, 0, 4)
	for i := 0; i < 4; i++ {
		t := time.Now()
		if err := sess.Run(); err != nil {
			fmt.Println("RUN_FAIL", err)
			return false
		}
		times = append(times, float64(time.Since(t).Microseconds())/1000)
	}
	fmt.Printf("runs_ms %.1f\n", times)
	return true
}

func runCtd() bool {
	fmt.Println("--- comictextdetector.onnx (1024x1024)")
	opts, err := newOptions()
	if err != nil {
		fmt.Println("OPTS_FAIL", err)
		return false
	}
	defer opts.Destroy()

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
		fmt.Println("SESSION_FAIL", err)
		return false
	}
	defer sess.Destroy()

	times := make([]float64, 0, 4)
	for i := 0; i < 4; i++ {
		t := time.Now()
		if err := sess.Run(); err != nil {
			fmt.Println("RUN_FAIL", err)
			return false
		}
		times = append(times, float64(time.Since(t).Microseconds())/1000)
	}
	fmt.Printf("runs_ms %.1f\n", times)
	return true
}
