// probe-layout compares the classic rectangular bubble anchor with shaped
// bubble areas while holding OCR, translations, and inpainting fixed.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"mangatrans/internal/bubble"
	"mangatrans/internal/cbz"
	"mangatrans/internal/infer"
	"mangatrans/internal/inpaint"
	"mangatrans/internal/render"
)

type block struct {
	Box      image.Rectangle `json:"box"`
	Vertical bool            `json:"vertical"`
	OCR      string          `json:"ocr"`
	ZH       string          `json:"zh"`
}
type pageReport struct {
	Page                 int                `json:"page"`
	Entry                string             `json:"entry"`
	SourceSHA256         string             `json:"source_sha256"`
	Texts                []string           `json:"texts"`
	Boxes                []image.Rectangle  `json:"boxes"`
	AreaBounds           []*image.Rectangle `json:"area_bounds"`
	FallbackCount        int                `json:"fallback_count"`
	ChangedTextPixels    int                `json:"changed_text_pixels"`
	OutsideAllowedPixels int                `json:"outside_allowed_pixels"`
	Outputs              map[string]string  `json:"outputs"`
}
type manifest struct {
	Direction   render.Direction `json:"text_direction"`
	Input       string           `json:"input"`
	InputSHA256 string           `json:"input_sha256"`
	Pages       []pageReport     `json:"pages"`
}

func main() {
	if err := run(); err != nil {
		log.Printf("probe-layout: %v", err)
		os.Exit(1)
	}
}

func run() error {
	in := flag.String("in", "", "input CBZ")
	blocksDir := flag.String("blocks", "", "directory containing blocksNN.json and maskNN.png")
	workDir := flag.String("work", "", "existing engine dump directory (defaults to -blocks)")
	outDir := flag.String("out", "", "output directory")
	pageArg := flag.String("pages", "3,7,8,10,11,12", "1-based comma-separated pages")
	cleanDir := flag.String("clean-dir", "", "reuse clean-NN.png pages and skip ONNX initialization")
	fontPath := flag.String("font", `C:\Windows\Fonts\msyhbd.ttc`, "rendering font")
	directionFlag := flag.String("text-direction", "vertical", "new layout: vertical, horizontal or auto; classic keeps cached direction")
	model := flag.String("model", `E:\work\transl\.cache\onnx-probe\lama-manga.onnx`, "LaMa inpainting model")
	ortDir := flag.String("ort-dir", "runtime/ort-cuda13", "ONNX Runtime directory")
	cudaDir := flag.String("cuda-dir", `E:\Python312\Lib\site-packages\torch\lib`, "CUDA DLL directory")
	flag.Parse()
	direction, err := render.ParseDirection(*directionFlag)
	if err != nil {
		return err
	}
	if *in == "" || *blocksDir == "" || *outDir == "" {
		return fmt.Errorf("usage: go run ./cmd/probe-layout -in book.cbz -blocks dump -out result [-pages 3,7] [-clean-dir clean-pages]")
	}
	if *workDir == "" {
		*workDir = *blocksDir
	}
	pagesWanted, err := parsePages(*pageArg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	archive, err := os.ReadFile(*in)
	if err != nil {
		return fmt.Errorf("read input: %w", err)
	}
	pages, err := cbz.ReadCBZ(*in)
	if err != nil {
		return err
	}
	var lap *inpaint.Model
	if *cleanDir == "" {
		runtime, err := infer.Env(*ortDir, *cudaDir)
		if err != nil {
			return err
		}
		defer runtime.Destroy()
		lap, err = inpaint.Load(*model, true, 3<<30)
		if err != nil {
			return err
		}
		defer lap.Close()
	}
	report := manifest{Input: *in, InputSHA256: hash(archive), Direction: direction}
	for _, pageNo := range pagesWanted {
		log.Printf("page %d: preparing fixed-text comparison", pageNo)
		idx := pageNo - 1
		if idx < 0 || idx >= len(pages) {
			return fmt.Errorf("page %d is outside CBZ page count %d", pageNo, len(pages))
		}
		maskPath := filepath.Join(*workDir, fmt.Sprintf("mask%02d.png", idx))
		blockPath := filepath.Join(*workDir, fmt.Sprintf("blocks%02d.json", idx))
		maskImg, err := readImage(maskPath)
		if err != nil {
			return fmt.Errorf("page %d mask: %w", pageNo, err)
		}
		mask, ok := maskImg.(*image.Gray)
		if !ok {
			return fmt.Errorf("page %d mask %s is %T, expected grayscale PNG", pageNo, maskPath, maskImg)
		}
		data, err := os.ReadFile(blockPath)
		if err != nil {
			return fmt.Errorf("page %d blocks: %w", pageNo, err)
		}
		var srcBlocks []block
		if err := json.Unmarshal(data, &srcBlocks); err != nil {
			return fmt.Errorf("page %d blocks JSON: %w", pageNo, err)
		}
		source, err := decode(pages[idx].Bytes)
		if err != nil {
			return fmt.Errorf("decode page %d: %w", pageNo, err)
		}
		if mask.Bounds().Dx() != source.Bounds().Dx() || mask.Bounds().Dy() != source.Bounds().Dy() {
			return fmt.Errorf("page %d mask size %v differs from source %v", pageNo, mask.Bounds(), source.Bounds())
		}
		cleanPath := filepath.Join(*cleanDir, fmt.Sprintf("clean-%02d.png", pageNo))
		var clean image.Image
		if *cleanDir != "" {
			clean, err = readImage(cleanPath)
		} else {
			clean, err = lap.Inpaint(context.Background(), source, mask)
		}
		if err != nil {
			return fmt.Errorf("page %d inpaint/read clean: %w", pageNo, err)
		}
		if clean == nil {
			return fmt.Errorf("page %d produced no clean image", pageNo)
		}
		if clean.Bounds().Dx() != source.Bounds().Dx() || clean.Bounds().Dy() != source.Bounds().Dy() {
			return fmt.Errorf("page %d clean image size %v differs from source %v", pageNo, clean.Bounds(), source.Bounds())
		}
		classic, shaped, areaOverlay, allowed, pr, err := comparePage(source, clean, srcBlocks, *fontPath, pageNo, direction)
		if err != nil {
			return err
		}
		paths := map[string]string{}
		outputs := map[string]image.Image{"clean": clean, "before": classic, "after": shaped, "areas": areaOverlay, "allowed": allowed, "compare": sideBySide(classic, shaped)}
		for name, img := range outputs {
			path := filepath.Join(*outDir, fmt.Sprintf("%s-%02d.png", name, pageNo))
			if err := writePNG(path, img); err != nil {
				return err
			}
			paths[name] = path
		}
		pr.Page, pr.Entry, pr.SourceSHA256, pr.Outputs = pageNo, pages[idx].Name, hash(pages[idx].Bytes), paths
		report.Pages = append(report.Pages, pr)
		log.Printf("page %d: %d blocks, %d box fallbacks, %d text pixels, %d outside allowed area", pageNo, len(pr.Texts), pr.FallbackCount, pr.ChangedTextPixels, pr.OutsideAllowedPixels)
	}
	b, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outDir, "manifest.json"), b, 0o644); err != nil {
		return err
	}
	return nil
}

func comparePage(source, clean image.Image, in []block, fontPath string, pageNo int, direction render.Direction) (image.Image, image.Image, image.Image, image.Image, pageReport, error) {
	active := make([]block, 0, len(in))
	for _, b := range in {
		if strings.TrimSpace(b.ZH) != "" {
			active = append(active, b)
		}
	}
	boxes := make([]image.Rectangle, len(active))
	for i, b := range active {
		boxes[i] = b.Box
	}
	finder := bubble.Find(source, 210)
	areas := bubble.Find(clean, 210).FitAreas(boxes)
	classicBlocks, shapedBlocks := make([]render.Block, 0, len(active)), make([]render.Block, 0, len(active))
	used := map[image.Rectangle]bool{}
	fallbacks := 0
	areaBounds := make([]*image.Rectangle, len(active))
	for i, b := range active {
		c := render.Block{Box: b.Box, Text: b.ZH, Vertical: b.Vertical, TextColor: textColor(source, b.Box)}
		if anchor, ok := finder.Fit(b.Box); ok && !used[anchor] {
			c.Anchor = anchor
			used[anchor] = true
			c.TextColor = textColor(source, anchor)
		}
		classicBlocks = append(classicBlocks, c)
		textSample := b.Box
		if areas[i] != nil {
			textSample = areas[i].Bounds()
		}
		n := render.Block{Box: b.Box, Text: b.ZH, Vertical: direction.IsVertical(b.Vertical), TextColor: textColor(source, textSample), Area: areas[i]}
		if areas[i] == nil {
			fallbacks++
		} else {
			r := areas[i].Bounds()
			areaBounds[i] = &r
		}
		shapedBlocks = append(shapedBlocks, n)
	}
	opt := render.Options{FontPath: fontPath, MinSize: 1.2, MaxSize: 16, Padding: 3, TextColor: color.RGBA{A: 255}}
	before, err := render.Render(clean, classicBlocks, opt)
	if err != nil {
		return nil, nil, nil, nil, pageReport{}, fmt.Errorf("page %d classic render: %w", pageNo, err)
	}
	after, err := render.Render(clean, shapedBlocks, opt)
	if err != nil {
		return nil, nil, nil, nil, pageReport{}, fmt.Errorf("page %d shaped render: %w", pageNo, err)
	}
	allowed := allowedPixels(source.Bounds(), boxes, areas)
	changed, outside := pixelChanges(clean, after, allowed)
	if len(active) > 0 && changed == 0 {
		return nil, nil, nil, nil, pageReport{}, fmt.Errorf("page %d shaped render changed no pixels for %d nonempty translations", pageNo, len(active))
	}
	if outside > 0 {
		return nil, nil, nil, nil, pageReport{}, fmt.Errorf("page %d shaped render changed %d pixels outside allowed areas", pageNo, outside)
	}
	texts := make([]string, len(active))
	for i, b := range active {
		texts[i] = b.ZH
	}
	return before, after, overlayAreas(source, boxes, areas), allowed, pageReport{Texts: texts, Boxes: boxes, AreaBounds: areaBounds, FallbackCount: fallbacks, ChangedTextPixels: changed, OutsideAllowedPixels: outside}, nil
}

func parsePages(s string) ([]int, error) {
	var out []int
	seen := map[int]bool{}
	for _, v := range strings.Split(s, ",") {
		n, e := strconv.Atoi(strings.TrimSpace(v))
		if e != nil || n < 1 {
			return nil, fmt.Errorf("invalid 1-based page %q", v)
		}
		if !seen[n] {
			out = append(out, n)
			seen[n] = true
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no pages requested")
	}
	return out, nil
}
func readImage(p string) (image.Image, error) {
	b, e := os.ReadFile(p)
	if e != nil {
		return nil, e
	}
	im, _, e := image.Decode(bytes.NewReader(b))
	return im, e
}
func decode(b []byte) (image.Image, error) {
	im, _, e := image.Decode(bytes.NewReader(b))
	return im, e
}
func writePNG(p string, im image.Image) error {
	f, e := os.Create(p)
	if e != nil {
		return e
	}
	if e = png.Encode(f, im); e != nil {
		f.Close()
		return e
	}
	return f.Close()
}
func hash(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func textColor(im image.Image, box image.Rectangle) color.RGBA {
	r := box.Intersect(im.Bounds())
	if r.Empty() {
		return color.RGBA{A: 255}
	}
	var hist [256]int
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			c := color.RGBAModel.Convert(im.At(x, y)).(color.RGBA)
			lum := (299*int(c.R) + 587*int(c.G) + 114*int(c.B)) / 1000
			hist[lum]++
		}
	}
	bg := 0
	for i := 1; i < 256; i++ {
		if hist[i] > hist[bg] {
			bg = i
		}
	}
	if bg > 127 {
		return color.RGBA{A: 255}
	}
	return color.RGBA{R: 255, G: 255, B: 255, A: 255}
}
func overlayAreas(src image.Image, boxes []image.Rectangle, areas []*image.Alpha) image.Image {
	b := src.Bounds()
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(out, out.Bounds(), src, b.Min, draw.Src)
	for _, a := range areas {
		if a == nil {
			continue
		}
		r := a.Bounds().Intersect(src.Bounds())
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if a.AlphaAt(x, y).A == 255 {
					p := image.Pt(x-b.Min.X, y-b.Min.Y)
					c := out.RGBAAt(p.X, p.Y)
					out.SetRGBA(p.X, p.Y, color.RGBA{R: uint8((int(c.R) + 40) / 2), G: uint8((int(c.G) + 80) / 2), B: uint8((int(c.B) + 255) / 2), A: 255})
				}
			}
		}
	}
	for _, r := range boxes {
		drawRect(out, r.Sub(b.Min), color.RGBA{R: 255, G: 70, A: 255})
	}
	return out
}

// allowedPixels combines every shaped safe area with the legacy OCR rectangles
// used by fallback blocks. Its coordinates remain in the page coordinate space.
func allowedPixels(bounds image.Rectangle, boxes []image.Rectangle, areas []*image.Alpha) *image.Gray {
	allowed := image.NewGray(bounds)
	for i, area := range areas {
		if area != nil {
			r := area.Bounds().Intersect(bounds)
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					if area.AlphaAt(x, y).A == 255 {
						allowed.SetGray(x, y, color.Gray{Y: 255})
					}
				}
			}
		} else if i < len(boxes) {
			r := boxes[i].Intersect(bounds)
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					allowed.SetGray(x, y, color.Gray{Y: 255})
				}
			}
		}
	}
	return allowed
}

func pixelChanges(clean, rendered image.Image, allowed *image.Gray) (changed, outside int) {
	b := clean.Bounds().Intersect(rendered.Bounds())
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if color.RGBAModel.Convert(clean.At(x, y)) == color.RGBAModel.Convert(rendered.At(x, y)) {
				continue
			}
			changed++
			if allowed.GrayAt(x, y).Y == 0 {
				outside++
			}
		}
	}
	return changed, outside
}
func sideBySide(a, b image.Image) image.Image {
	ab, bb := a.Bounds(), b.Bounds()
	w, h := ab.Dx()+bb.Dx(), max(ab.Dy(), bb.Dy())
	out := image.NewRGBA(image.Rect(0, 0, w, h+18))
	draw.Draw(out, out.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
	draw.Draw(out, image.Rect(0, 18, ab.Dx(), 18+ab.Dy()), a, ab.Min, draw.Src)
	draw.Draw(out, image.Rect(ab.Dx(), 18, w, 18+bb.Dy()), b, bb.Min, draw.Src)
	d := font.Drawer{Dst: out, Src: image.NewUniform(color.Black), Face: basicfont.Face7x13}
	d.Dot = fixed.P(3, 13)
	d.DrawString("BEFORE")
	d.Dot = fixed.P(ab.Dx()+3, 13)
	d.DrawString("AFTER")
	return out
}
func drawRect(dst *image.RGBA, r image.Rectangle, c color.RGBA) {
	r = r.Intersect(dst.Bounds())
	if r.Empty() {
		return
	}
	for x := r.Min.X; x < r.Max.X; x++ {
		dst.SetRGBA(x, r.Min.Y, c)
		dst.SetRGBA(x, r.Max.Y-1, c)
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		dst.SetRGBA(r.Min.X, y, c)
		dst.SetRGBA(r.Max.X-1, y, c)
	}
}
