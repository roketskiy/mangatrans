//go:build windows

package inpaint

import (
	"image"
	"image/color"
	"os"
	"runtime"
	"testing"

	"mangatrans/internal/infer"
)

func TestInpaintRemovesMarkedText(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("ONNX runtime test is Windows-only")
	}
	const modelPath = `E:\work\transl\.cache\onnx-probe\lama-manga.onnx`
	if _, err := os.Stat(modelPath); err != nil {
		t.Skip("LaMa model is not available")
	}
	// The shared ORT environment must be initialized before any session loads.
	if _, err := infer.Env(`E:\work\transl\runtime\ort-cuda13`, ``); err != nil {
		t.Skipf("onnxruntime env unavailable: %v", err)
	}
	m, err := Load(modelPath, false, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	img := image.NewRGBA(image.Rect(0, 0, 512, 512))
	mask := image.NewGray(img.Bounds())
	for y := 0; y < 512; y++ {
		for x := 0; x < 512; x++ {
			img.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	for _, r := range []image.Rectangle{image.Rect(180, 190, 200, 260), image.Rect(205, 230, 250, 245)} {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				img.SetRGBA(x, y, color.RGBA{A: 255})
				mask.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	out, err := m.Inpaint(img, mask)
	if err != nil {
		t.Fatal(err)
	}
	if out.Bounds().Dx() != 512 || out.Bounds().Dy() != 512 {
		t.Fatalf("unexpected output size: %v", out.Bounds())
	}
	// The synthetic page is white with black rectangles as fake text, and the
	// window around them is pure white, so the solid-background fast path
	// should fill the holes with white. Assert the fake text is gone: every
	// previously masked pixel must now be (near) white.
	worst := 0
	for y := 0; y < 512; y++ {
		for x := 0; x < 512; x++ {
			if mask.GrayAt(x, y).Y == 0 {
				continue
			}
			c := out.At(x, y)
			r, _, _, _ := c.RGBA()
			d := 255 - int(r>>8)
			if d > worst {
				worst = d
			}
		}
	}
	if worst > 16 {
		t.Fatalf("masked pixels were not restored toward white, worst residual=%d", worst)
	}
}
