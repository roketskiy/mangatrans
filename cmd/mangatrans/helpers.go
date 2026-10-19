package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math"
	"os"
	"strings"

	"mangatrans/internal/llm"
	"mangatrans/internal/pageimage"
	"mangatrans/internal/render"
)

func decodeImage(b []byte) (image.Image, error) {
	return pageimage.Decode(b)
}

func encodeJPEG(img image.Image, quality int) ([]byte, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// cropRect returns a copy of the region, or nil when the region is empty.
func cropRect(src image.Image, x, y, w, h float32, pad int) image.Image {
	b := src.Bounds()
	x0 := int(x) - pad
	y0 := int(y) - pad
	x1 := int(x+w) + pad
	y1 := int(y+h) + pad
	if x0 < b.Min.X {
		x0 = b.Min.X
	}
	if y0 < b.Min.Y {
		y0 = b.Min.Y
	}
	if x1 > b.Max.X {
		x1 = b.Max.X
	}
	if y1 > b.Max.Y {
		y1 = b.Max.Y
	}
	if x1-x0 < 4 || y1-y0 < 4 {
		return nil
	}
	dst := image.NewRGBA(image.Rect(0, 0, x1-x0, y1-y0))
	for yy := 0; yy < y1-y0; yy++ {
		for xx := 0; xx < x1-x0; xx++ {
			dst.Set(xx, yy, src.At(x0+xx, y0+yy))
		}
	}
	return dst
}

// cleanOCR removes the wrappers a vision model occasionally adds. The model
// is asked for the text alone; a stray label or code fence must not become
// part of the translation or the mask.
func cleanOCR(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "`")
	for _, p := range []string{"OCR:", "OCR：", "文字：", "文本：", "识别结果："} {
		s = strings.TrimSpace(strings.TrimPrefix(s, p))
	}
	s = strings.ReplaceAll(s, "\r", "")
	return strings.TrimSpace(s)
}

// ocrBox asks the OCR model to read one cropped region.
func ocrBox(ctx context.Context, srv *llm.Server, crop image.Image) (string, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, crop); err != nil {
		return "", fmt.Errorf("encode crop: %w", err)
	}
	msg := llm.MultimodalMessage{
		Role: "user",
		Content: []interface{}{
			llm.ImageContentPart{
				Type:     "image_url",
				ImageURL: llm.ImageURLStruct("data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())),
			},
			llm.TextContentPart{Type: "text", Text: "OCR:"},
		},
	}
	return srv.ChatMultimodal(ctx, []llm.MultimodalMessage{msg}, 0.0, 256)
}

// estimateFontSize guesses the source type size from ink density: mask area
// over the recognised character count, assuming Japanese glyphs cover about
// a third of their em square. Returns 0 when either input is unusable.
func estimateFontSize(mask *image.Gray, box image.Rectangle, chars int) float64 {
	if mask == nil || chars <= 0 {
		return 0
	}
	r := box.Intersect(mask.Rect)
	area := 0
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if mask.GrayAt(x, y).Y > 0 {
				area++
			}
		}
	}
	if area == 0 {
		return 0
	}
	return math.Sqrt(float64(area) / float64(chars) / 0.33)
}

// restrictMask keeps text-mask pixels only inside boxes that will be redrawn.
// A box is expanded slightly because the detected rectangle can clip the
// outermost strokes of a glyph.
func restrictMask(full *image.Gray, blocks []render.Block) *image.Gray {
	if full == nil {
		return nil
	}
	b := full.Bounds()
	out := image.NewGray(b)
	for _, blk := range blocks {
		r := blk.Box.Inset(-8).Intersect(b)
		if r.Empty() {
			continue
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			if blk.SourceArea != nil {
				for x := r.Min.X; x < r.Max.X; x++ {
					if blk.SourceArea.AlphaAt(x, y).A == 255 {
						out.SetGray(x, y, full.GrayAt(x, y))
					}
				}
				continue
			}
			// PixOffset respects a non-zero image origin and stride. The old
			// y*Stride arithmetic silently assumed both were zero.
			src := full.PixOffset(r.Min.X, y)
			dst := out.PixOffset(r.Min.X, y)
			copy(out.Pix[dst:dst+r.Dx()], full.Pix[src:src+r.Dx()])
		}
	}
	return out
}

// textColor picks black or white from the median luminance of the box. The
// median approximates the background (bubble interior or panel fill), so the
// text takes the opposite colour. Manga occasionally puts white dialogue on a
// black panel; drawing that translation in black would make it invisible.
func textColor(img image.Image, box image.Rectangle) color.RGBA {
	if img == nil {
		return color.RGBA{A: 255}
	}
	r := box.Intersect(img.Bounds())
	if r.Empty() {
		return color.RGBA{A: 255}
	}
	var hist [256]int
	total := 0
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			c := color.RGBAModel.Convert(img.At(x, y)).(color.RGBA)
			lum := (299*uint32(c.R) + 587*uint32(c.G) + 114*uint32(c.B)) / 1000
			hist[lum]++
			total++
		}
	}
	if total == 0 {
		return color.RGBA{A: 255}
	}
	// The most frequent luminance is the background: strokes cover less area
	// than the bubble or panel behind them, even in a tight text box. A median
	// would flip to white on dense vertical text.
	background := 0
	for i := 1; i < 256; i++ {
		if hist[i] > hist[background] {
			background = i
		}
	}
	if background > 127 {
		return color.RGBA{A: 255}
	}
	return color.RGBA{R: 255, G: 255, B: 255, A: 255}
}

type blockDump struct {
	Box      image.Rectangle `json:"box"`
	Vertical bool            `json:"vertical"`
	OCR      string          `json:"ocr"`
	ZH       string          `json:"zh"`
}

func writeBlocks(path string, blocks []blockDump) {
	data, err := json.MarshalIndent(blocks, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(path, data, 0o644)
}

// readTail returns the last maxLines lines of a text file, reading at most the
// final 32 KiB. It is used only on the error path to show a server log before
// the run directory is removed.
func readTail(path string, maxLines int) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return ""
	}
	const maxBytes = 32 << 10
	if st.Size() > maxBytes {
		if _, err := f.Seek(-maxBytes, io.SeekEnd); err != nil {
			return ""
		}
	}
	data, err := io.ReadAll(io.LimitReader(f, maxBytes))
	if err != nil {
		return ""
	}
	text := strings.TrimRight(string(data), "\r\n")
	if text == "" {
		return ""
	}
	lines := strings.Split(text, "\n")
	if len(lines) > maxLines {
		lines = lines[len(lines)-maxLines:]
	}
	return strings.Join(lines, "\n")
}

func writePNG(path string, img image.Image) {
	if img == nil {
		return
	}
	f, err := os.Create(path)
	if err != nil {
		return
	}
	defer f.Close()
	png.Encode(f, img)
}
