package main

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"time"

	"mangatrans/internal/cbz"
	"mangatrans/internal/detect"
	"mangatrans/internal/infer"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("usage: probe-detect <input.cbz> [outdir]")
		os.Exit(1)
	}
	in := os.Args[1]
	outDir := `E:\work\transl\.cache\detect-out`
	if len(os.Args) > 2 {
		outDir = os.Args[2]
	}
	os.MkdirAll(outDir, 0o755)

	pages, err := cbz.ReadCBZ(in)
	if err != nil {
		panic(err)
	}
	fmt.Printf("pages %d\n", len(pages))

	rt, err := infer.Env(`E:\work\transl\runtime\ort-cuda13`, `E:\Python312\Lib\site-packages\torch\lib`)
	if err != nil {
		panic(err)
	}
	defer rt.Destroy()

	m, err := detect.Load(`E:\work\transl\.cache\onnx-probe\comictextdetector.onnx`, true, 3<<30)
	if err != nil {
		panic(err)
	}
	defer m.Close()

	params := detect.DefaultParams()
	for i, p := range pages {
		if i >= 6 {
			break
		}
		img, err := decode(p.Bytes)
		if err != nil {
			fmt.Printf("page %d decode: %v\n", i, err)
			continue
		}
		b := img.Bounds()
		t0 := time.Now()
		res, err := m.Run(img, params)
		if err != nil {
			fmt.Printf("page %d detect: %v\n", i, err)
			continue
		}
		dt := time.Since(t0)

		var masked int
		for _, v := range res.Mask.Pix {
			if v != 0 {
				masked++
			}
		}
		fmt.Printf("page %d %dx%d boxes=%d mask_px=%.1f%% %v\n",
			i, b.Dx(), b.Dy(), len(res.Boxes), 100*float64(masked)/float64(len(res.Mask.Pix)), dt.Round(time.Millisecond))

		vis := overlay(img, res.Boxes)
		out := filepath.Join(outDir, fmt.Sprintf("p%02d.png", i))
		savePNG(out, vis)
		_ = os.Stdout
	}
}

func decode(b []byte) (image.Image, error) {
	if img, err := png.Decode(bytes.NewReader(b)); err == nil {
		return img, nil
	}
	if img, err := jpeg.Decode(bytes.NewReader(b)); err == nil {
		return img, nil
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	return img, err
}

func overlay(src image.Image, boxes []infer.Rect) image.Image {
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)

	red := color.RGBA{255, 0, 0, 255}
	for _, r := range boxes {
		x0, y0 := int(r.X), int(r.Y)
		x1, y1 := int(r.X+r.W), int(r.Y+r.H)
		for t := 0; t < 2; t++ {
			drawHLine(dst, x0, x1, y0+t, red)
			drawHLine(dst, x0, x1, y1-t, red)
			drawVLine(dst, y0, y1, x0+t, red)
			drawVLine(dst, y0, y1, x1-t, red)
		}
	}
	return dst
}

func drawHLine(img *image.RGBA, x0, x1, y int, c color.RGBA) {
	if y < 0 || y >= img.Bounds().Dy() {
		return
	}
	for x := x0; x <= x1; x++ {
		if x >= 0 && x < img.Bounds().Dx() {
			img.SetRGBA(x, y, c)
		}
	}
}

func drawVLine(img *image.RGBA, y0, y1, x int, c color.RGBA) {
	if x < 0 || x >= img.Bounds().Dx() {
		return
	}
	for y := y0; y <= y1; y++ {
		if y >= 0 && y < img.Bounds().Dy() {
			img.SetRGBA(x, y, c)
		}
	}
}

func savePNG(path string, img image.Image) {
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	png.Encode(f, img)
}
