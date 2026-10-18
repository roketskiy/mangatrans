// probe-seg runs the bubble segmenter over a CBZ and writes one overlay per
// page: each instance mask tinted, with its box outlined.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"mangatrans/internal/cbz"
	"mangatrans/internal/infer"
	"mangatrans/internal/segment"
)

var tints = []color.RGBA{
	{R: 255, G: 80, B: 80, A: 255},
	{R: 80, G: 160, B: 255, A: 255},
	{R: 120, G: 220, B: 120, A: 255},
	{R: 255, G: 200, B: 60, A: 255},
	{R: 220, G: 120, B: 255, A: 255},
	{R: 90, G: 230, B: 220, A: 255},
	{R: 255, G: 130, B: 200, A: 255},
	{R: 180, G: 180, B: 90, A: 255},
}

func main() {
	in := flag.String("in", "", "input CBZ")
	model := flag.String("model", `E:\work\transl\models\mangalens-bubble-seg.onnx`, "segmenter ONNX")
	outDir := flag.String("out", ".cache/seg-probe", "output directory")
	pages := flag.String("pages", "", "1-based comma-separated pages (default all)")
	conf := flag.Float64("conf", 0.25, "instance confidence threshold")
	flag.Parse()
	if *in == "" || *model == "" {
		fmt.Println("usage: probe-seg -in book.cbz -model seg.onnx -out dir [-pages 1,2]")
		os.Exit(1)
	}
	if err := run(*in, *model, *outDir, *pages, float32(*conf)); err != nil {
		log.Printf("probe-seg: %v", err)
		os.Exit(1)
	}
}

func run(in, model, outDir, pageArg string, conf float32) error {
	wanted, err := parsePages(pageArg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	rt, err := infer.Env(`E:\work\transl\runtime\ort-cuda13`, `E:\Python312\Lib\site-packages\torch\lib`)
	if err != nil {
		return err
	}
	defer rt.Destroy()
	seg, err := segment.Load(model, true, 3<<30, 1600)
	if err != nil {
		return err
	}
	defer seg.Close()

	pgs, err := cbz.ReadCBZ(in)
	if err != nil {
		return err
	}
	if len(wanted) == 0 {
		for i := range pgs {
			wanted = append(wanted, i+1)
		}
	}
	for _, pageNo := range wanted {
		idx := pageNo - 1
		if idx < 0 || idx >= len(pgs) {
			return fmt.Errorf("page %d outside 1..%d", pageNo, len(pgs))
		}
		src, _, err := image.Decode(bytes.NewReader(pgs[idx].Bytes))
		if err != nil {
			return fmt.Errorf("decode page %d: %w", pageNo, err)
		}
		bubbles, err := seg.Run(src, segment.Params{ConfThreshold: conf, IOUThreshold: 0.7, MaskThreshold: 0.5, Side: 1600})
		if err != nil {
			return fmt.Errorf("page %d: %w", pageNo, err)
		}
		overlay := drawOverlay(src, bubbles)
		path := filepath.Join(outDir, fmt.Sprintf("seg-%02d.png", pageNo))
		if err := writePNG(path, overlay); err != nil {
			return err
		}
		log.Printf("page %d: %d bubbles -> %s", pageNo, len(bubbles), path)
	}
	return nil
}

func drawOverlay(src image.Image, bubbles []segment.Bubble) *image.RGBA {
	b := src.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, src, b.Min, draw.Src)
	for i, bub := range bubbles {
		tint := tints[i%len(tints)]
		tint.A = 70
		for y := bub.Box.Min.Y; y < bub.Box.Max.Y; y++ {
			for x := bub.Box.Min.X; x < bub.Box.Max.X; x++ {
				if bub.Mask.AlphaAt(x, y).A == 255 {
					out.SetRGBA(x, y, mix(out.RGBAAt(x, y), tint))
				}
			}
		}
		edge := color.RGBA{R: tint.R, G: tint.G, B: tint.B, A: 255}
		drawBoxOutline(out, bub.Box, edge)
	}
	return out
}

func mix(base color.RGBA, tint color.RGBA) color.RGBA {
	return color.RGBA{
		R: uint8((int(base.R)*3 + int(tint.R)) / 4),
		G: uint8((int(base.G)*3 + int(tint.G)) / 4),
		B: uint8((int(base.B)*3 + int(tint.B)) / 4),
		A: 255,
	}
}

func drawBoxOutline(out *image.RGBA, r image.Rectangle, c color.RGBA) {
	for x := r.Min.X; x < r.Max.X; x++ {
		out.SetRGBA(x, r.Min.Y, c)
		out.SetRGBA(x, r.Max.Y-1, c)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		out.SetRGBA(r.Min.X, y, c)
		out.SetRGBA(r.Max.X-1, y, c)
	}
}

func parsePages(arg string) ([]int, error) {
	if strings.TrimSpace(arg) == "" {
		return nil, nil
	}
	var out []int
	for _, part := range strings.Split(arg, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil {
			return nil, fmt.Errorf("bad page %q: %w", part, err)
		}
		out = append(out, n)
	}
	return out, nil
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
