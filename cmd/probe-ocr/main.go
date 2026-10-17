package main

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"mangatrans/internal/cbz"
	"mangatrans/internal/detect"
	"mangatrans/internal/infer"
	"mangatrans/internal/llm"
)

const (
	llamaExe   = `C:\Users\Public\Temp\opencode\llamacpp\llama\llama-server.exe`
	ocrModel   = `E:\work\transl\models\PaddleOCR-VL-manga-Q4_K_M.gguf`
	ocrMMProj  = `E:\work\transl\models\mmproj-Q8_0.gguf`
	detModel   = `E:\work\transl\.cache\onnx-probe\comictextdetector.onnx`
	ortDir     = `E:\work\transl\runtime\ort-cuda13`
	cudaDir    = `E:\Python312\Lib\site-packages\torch\lib`
	outDir     = `E:\work\transl\.cache\ocr-out`
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("usage: probe-ocr <input.cbz>")
		os.Exit(1)
	}
	os.MkdirAll(outDir, 0o755)

	pages, err := cbz.ReadCBZ(os.Args[1])
	if err != nil {
		panic(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	// Start the OCR server once for all pages.
	fmt.Println("starting llama-server (OCR)...")
	srv, err := llm.Start(ctx, llm.Config{
		Exe:     llamaExe,
		Model:   ocrModel,
		MMProj:  ocrMMProj,
		LogPath: filepath.Join(outDir, "llama-ocr.log"),
		Args: []string{
			"-ngl", "99",
			"-c", "4096",
			"--temp", "0",
			"--jinja",
		},
	})
	if err != nil {
		panic(err)
	}
	defer srv.Kill()
	fmt.Println("OCR server ready on port", srv.Port())

	rt, err := infer.Env(ortDir, cudaDir)
	if err != nil {
		panic(err)
	}
	defer rt.Destroy()

	det, err := detect.Load(detModel, true, 3<<30)
	if err != nil {
		panic(err)
	}
	defer det.Close()

	params := detect.DefaultParams()
	params.ConfThreshold = 0.35

	totalOCR := 0
	for i, p := range pages {
		if i >= 5 {
			break
		}
		img, _, err := image.Decode(strings.NewReader(string(p.Bytes)))
		if err != nil {
			fmt.Printf("page %d decode: %v\n", i, err)
			continue
		}

		res, err := det.Run(img, params)
		if err != nil {
			fmt.Printf("page %d detect: %v\n", i, err)
			continue
		}
		fmt.Printf("page %d: %d boxes\n", i, len(res.Boxes))

		for j, box := range res.Boxes {
			// Crop with padding so the OCR model sees the bubble border.
			pad := float32(6)
			x0 := box.X - pad
			y0 := box.Y - pad
			w := box.W + 2*pad
			h := box.H + 2*pad
			crop := cropImage(img, int(x0), int(y0), int(w), int(h))
			if crop == nil {
				continue
			}

			var buf bytes.Buffer
			if err := png.Encode(&buf, crop); err != nil {
				continue
			}

			msg := llm.MultimodalMessage{
				Role: "user",
				Content: []interface{}{
					llm.ImageContentPart{Type: "image_url", ImageURL: llm.ImageURLStruct(llm.DataURL("image/png", buf.Bytes()))},
					llm.TextContentPart{Type: "text", Text: "OCR:"},
				},
			}
			t0 := time.Now()
			text, err := srv.ChatMultimodal(ctx, []llm.MultimodalMessage{msg}, 0.0, 256)
			dt := time.Since(t0)
			if err != nil {
				fmt.Printf("  box %d OCR: %v\n", j, err)
				continue
			}
			text = strings.TrimSpace(text)
			fmt.Printf("  box %d (%.0fms): %q\n", j, float64(dt.Microseconds())/1000, text)
			totalOCR++

			// save crops for later review
			cf, _ := os.Create(filepath.Join(outDir, fmt.Sprintf("p%02d_b%02d.png", i, j)))
			cf.Write(buf.Bytes())
			cf.Close()
		}
	}
	fmt.Println("total boxes OCR'd:", totalOCR)
}

func cropImage(src image.Image, x, y, w, h int) image.Image {
	b := src.Bounds()
	// clamp
	if x < b.Min.X {
		x = b.Min.X
	}
	if y < b.Min.Y {
		y = b.Min.Y
	}
	if x+w > b.Max.X {
		w = b.Max.X - x
	}
	if y+h > b.Max.Y {
		h = b.Max.Y - y
	}
	if w <= 0 || h <= 0 {
		return nil
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	for dy := 0; dy < h; dy++ {
		for dx := 0; dx < w; dx++ {
			dst.Set(dx, dy, src.At(x+dx, y+dy))
		}
	}
	return dst
}
