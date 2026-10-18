// probe-regions visualizes OCR regions split from existing OCR boxes.
package main

import (
	"bytes"
	"encoding/json"
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

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"mangatrans/internal/bubble"
	"mangatrans/internal/cbz"
)

type sourceBlock struct {
	Box image.Rectangle `json:"box"`
	OCR string          `json:"ocr"`
	ZH  string          `json:"zh"`
}

type regionReport struct {
	Index          int              `json:"index"`
	Parent         int              `json:"parent"`
	ParentBox      image.Rectangle  `json:"parent_box"`
	ZH             string           `json:"cached_parent_zh"`
	OCR            string           `json:"cached_parent_ocr"`
	Box            image.Rectangle  `json:"box"`
	AreaBounds     *image.Rectangle `json:"area_bounds,omitempty"`
	CropMaskBounds *image.Rectangle `json:"crop_mask_bounds,omitempty"`
	Crop           string           `json:"crop"`
}

type pageReport struct {
	Page    int            `json:"page"`
	Entry   string         `json:"entry"`
	Overlay string         `json:"overlay"`
	Regions []regionReport `json:"regions"`
}

type report struct {
	Input string       `json:"input"`
	Pages []pageReport `json:"pages"`
}

func main() {
	if err := run(); err != nil {
		log.Printf("probe-regions: %v", err)
		os.Exit(1)
	}
}

func run() error {
	in := flag.String("in", "", "input CBZ")
	blocksDir := flag.String("blocks", "", "directory containing blocksNN.json and maskNN.png")
	outDir := flag.String("out", "", "output directory")
	pageArg := flag.String("pages", "3,7,8,10,11,12", "1-based comma-separated pages")
	flag.Parse()
	if *in == "" || *blocksDir == "" || *outDir == "" {
		return fmt.Errorf("usage: probe-regions -in book.cbz -blocks dumpdir -out regions [-pages 3,7,8]")
	}
	wanted, err := parsePages(*pageArg)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	pages, err := cbz.ReadCBZ(*in)
	if err != nil {
		return err
	}
	r := report{Input: *in}
	for _, pageNo := range wanted {
		idx := pageNo - 1
		if idx < 0 || idx >= len(pages) {
			return fmt.Errorf("page %d is outside CBZ page count %d", pageNo, len(pages))
		}
		blocksPath := filepath.Join(*blocksDir, fmt.Sprintf("blocks%02d.json", idx))
		maskPath := filepath.Join(*blocksDir, fmt.Sprintf("mask%02d.png", idx))
		blocksData, err := os.ReadFile(blocksPath)
		if err != nil {
			return fmt.Errorf("page %d blocks: %w", pageNo, err)
		}
		var blocks []sourceBlock
		if err := json.Unmarshal(blocksData, &blocks); err != nil {
			return fmt.Errorf("page %d blocks JSON: %w", pageNo, err)
		}
		source, _, err := image.Decode(bytes.NewReader(pages[idx].Bytes))
		if err != nil {
			return fmt.Errorf("decode page %d: %w", pageNo, err)
		}
		maskFile, err := os.Open(maskPath)
		if err != nil {
			return fmt.Errorf("page %d mask: %w", pageNo, err)
		}
		maskImg, _, decodeErr := image.Decode(maskFile)
		closeErr := maskFile.Close()
		if decodeErr != nil {
			return fmt.Errorf("page %d decode mask: %w", pageNo, decodeErr)
		}
		if closeErr != nil {
			return fmt.Errorf("page %d close mask: %w", pageNo, closeErr)
		}
		mask, ok := maskImg.(*image.Gray)
		if !ok {
			return fmt.Errorf("page %d mask is %T, expected grayscale PNG", pageNo, maskImg)
		}
		if source.Bounds().Dx() != mask.Bounds().Dx() || source.Bounds().Dy() != mask.Bounds().Dy() {
			return fmt.Errorf("page %d mask size %v differs from source %v", pageNo, mask.Bounds(), source.Bounds())
		}
		boxes := make([]image.Rectangle, len(blocks))
		for i := range blocks {
			boxes[i] = blocks[i].Box
		}
		regions := bubble.SplitTextRegions(source, mask, boxes)
		pageOut := pageReport{Page: pageNo, Entry: pages[idx].Name}
		overlay := makeOverlay(source, boxes, regions)
		pageOut.Overlay = filepath.Join(*outDir, fmt.Sprintf("overlay-%02d.png", pageNo))
		if err := writePNG(pageOut.Overlay, overlay); err != nil {
			return err
		}
		for i, region := range regions {
			if region.Parent < 0 || region.Parent >= len(blocks) {
				return fmt.Errorf("page %d region %d has invalid parent index %d", pageNo, i, region.Parent)
			}
			crop := region.Crop(source, 6)
			if crop == nil {
				return fmt.Errorf("page %d region %d crop is empty", pageNo, i)
			}
			cropPath := filepath.Join(*outDir, fmt.Sprintf("crop-%02d-%03d.png", pageNo, i+1))
			if err := writePNG(cropPath, crop); err != nil {
				return err
			}
			var areaBounds *image.Rectangle
			if region.Area != nil {
				b := region.Area.Bounds()
				areaBounds = &b
			}
			var cropMaskBounds *image.Rectangle
			if region.CropMask != nil {
				b := region.CropMask.Bounds()
				cropMaskBounds = &b
			}
			parent := blocks[region.Parent]
			pageOut.Regions = append(pageOut.Regions, regionReport{
				Index: i, Parent: region.Parent, ParentBox: parent.Box, ZH: parent.ZH,
				OCR: parent.OCR, Box: region.Box, AreaBounds: areaBounds,
				CropMaskBounds: cropMaskBounds, Crop: cropPath,
			})
		}
		log.Printf("page %d: %d parents, %d OCR regions", pageNo, len(blocks), len(regions))
		r.Pages = append(r.Pages, pageOut)
	}
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(*outDir, "regions.json"), data, 0o644)
}

func makeOverlay(source image.Image, parents []image.Rectangle, regions []bubble.TextRegion) *image.RGBA {
	b := source.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, source, b.Min, draw.Src)
	for _, r := range parents {
		drawRect(out, r, color.RGBA{G: 210, B: 255, A: 255})
	}
	for i, r := range regions {
		drawRect(out, r.Box, color.RGBA{R: 255, G: 48, A: 255})
		label(out, r.Box.Min, strconv.Itoa(i+1))
	}
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

func label(dst *image.RGBA, at image.Point, s string) {
	if at.X+len(s)*7+4 > dst.Bounds().Max.X {
		at.X = dst.Bounds().Max.X - len(s)*7 - 4
	}
	if at.Y-14 < dst.Bounds().Min.Y {
		at.Y += 14
	}
	bg := image.Rect(at.X, at.Y-13, at.X+len(s)*7+4, at.Y+2).Intersect(dst.Bounds())
	draw.Draw(dst, bg, image.NewUniform(color.RGBA{R: 255, G: 48, A: 255}), image.Point{}, draw.Src)
	d := font.Drawer{Dst: dst, Src: image.NewUniform(color.White), Face: basicfont.Face7x13, Dot: fixed.P(at.X+2, at.Y)}
	d.DrawString(s)
}

func writePNG(path string, im image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, im); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func parsePages(s string) ([]int, error) {
	var pages []int
	seen := map[int]bool{}
	for _, part := range strings.Split(s, ",") {
		page, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || page < 1 {
			return nil, fmt.Errorf("invalid 1-based page %q", part)
		}
		if !seen[page] {
			pages = append(pages, page)
			seen[page] = true
		}
	}
	if len(pages) == 0 {
		return nil, fmt.Errorf("no pages requested")
	}
	return pages, nil
}
