package main

import (
	"bytes"
	"encoding/json"
	"flag"
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
	sweep := flag.Bool("sweep", false, "run the focused postprocessing parameter sweep")
	flag.Parse()
	if flag.NArg() < 1 {
		fmt.Println("usage: probe-detect [-sweep] <input.cbz> [outdir]")
		os.Exit(1)
	}
	in := flag.Arg(0)
	outDir := `E:\work\transl\.cache\detect-out`
	if flag.NArg() > 1 {
		outDir = flag.Arg(1)
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

	variants := []struct {
		name   string
		params detect.Params
	}{
		{"conf-025", paramsWith(func(p *detect.Params) { p.ConfThreshold = 0.25 })},
		{"conf-035", paramsWith(func(p *detect.Params) { p.ConfThreshold = 0.35 })},
		{"conf-045", paramsWith(func(p *detect.Params) { p.ConfThreshold = 0.45 })},
		{"conf-055", paramsWith(func(p *detect.Params) { p.ConfThreshold = 0.55 })},
		{"nms-020", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.NMSThreshold = 0.35, 0.20 })},
		{"nms-050", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.NMSThreshold = 0.35, 0.50 })},
		{"nms-070", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.NMSThreshold = 0.35, 0.70 })},
		{"box-000", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.BoxDilation = 0.35, 0 })},
		{"box-003", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.BoxDilation = 0.35, 3 })},
		{"close-000", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.CloseRadius = 0.35, 0 })},
		{"close-005", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.CloseRadius = 0.35, 5 })},
		{"close-015", paramsWith(func(p *detect.Params) { p.ConfThreshold, p.CloseRadius = 0.35, 15 })},
	}
	if !*sweep {
		variants = []struct {
			name   string
			params detect.Params
		}{{"default", paramsWith(func(p *detect.Params) { p.ConfThreshold = 0.35 })}}
	}
	type record struct {
		Variant string        `json:"variant"`
		Page    string        `json:"page"`
		Params  detect.Params `json:"params"`
		Boxes   []infer.Rect  `json:"boxes"`
		MaskPct float64       `json:"mask_percent"`
	}
	var records []record
	for i, page := range pages {
		if i >= 6 {
			break
		}
		img, err := decode(page.Bytes)
		if err != nil {
			fmt.Printf("page %s decode: %v\n", page.Name, err)
			continue
		}
		for _, variant := range variants {
			t0 := time.Now()
			res, err := m.Run(img, variant.params)
			if err != nil {
				fmt.Printf("%s %s detect: %v\n", variant.name, page.Name, err)
				continue
			}
			dt := time.Since(t0)
			var masked int
			for _, v := range res.Mask.Pix {
				if v != 0 {
					masked++
				}
			}
			maskPct := 100 * float64(masked) / float64(len(res.Mask.Pix))
			fmt.Printf("%-9s page=%-12s boxes=%2d mask=%.1f%% time=%v\n",
				variant.name, page.Name, len(res.Boxes), maskPct, dt.Round(time.Millisecond))
			records = append(records, record{variant.name, page.Name, variant.params, res.Boxes, maskPct})
			if *sweep {
				dir := filepath.Join(outDir, variant.name)
				if err := os.MkdirAll(dir, 0o755); err != nil {
					panic(err)
				}
				savePNG(filepath.Join(dir, page.Name), overlay(img, res.Boxes))
			} else {
				savePNG(filepath.Join(outDir, page.Name), overlay(img, res.Boxes))
			}
		}
	}
	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(outDir, "boxes.json"), data, 0o644); err != nil {
		panic(err)
	}
}

func paramsWith(change func(*detect.Params)) detect.Params {
	p := detect.DefaultParams()
	p.ConfThreshold = 0.35
	change(&p)
	return p
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
