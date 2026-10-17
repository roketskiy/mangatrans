package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	_ "image/png"
	"os"
	"path/filepath"

	"mangatrans/internal/bubble"
	"mangatrans/internal/cbz"
)

type block struct {
	Box      image.Rectangle `json:"box"`
	Vertical bool            `json:"vertical"`
	OCR      string          `json:"ocr"`
	ZH       string          `json:"zh"`
}

func main() {
	if len(os.Args) < 3 {
		fmt.Println("usage: probe-bubbles <book.cbz> <blocks-dir> [out-dir]")
		os.Exit(1)
	}
	outDir := ".cache/bubbles"
	if len(os.Args) > 3 {
		outDir = os.Args[3]
	}
	os.MkdirAll(outDir, 0o755)

	pages, err := cbz.ReadCBZ(os.Args[1])
	if err != nil {
		panic(err)
	}
	for i := 0; i < len(pages) && i < 8; i++ {
		data, err := os.ReadFile(filepath.Join(os.Args[2], fmt.Sprintf("blocks%02d.json", i)))
		if err != nil {
			continue
		}
		var blocks []block
		if err := json.Unmarshal(data, &blocks); err != nil {
			panic(err)
		}
		img, err := decode(pages[i].Bytes)
		if err != nil {
			panic(err)
		}
		f := bubble.Find(img, 210)
		out := image.NewRGBA(img.Bounds())
		draw.Draw(out, out.Bounds(), img, img.Bounds().Min, draw.Src)

		found := 0
		for _, b := range blocks {
			drawRect(out, b.Box, color.RGBA{G: 200, A: 255})
			if r, ok := f.Fit(b.Box); ok {
				found++
				drawRect(out, r, color.RGBA{R: 230, A: 255})
				fmt.Printf("p%02d %-24q text=%v bubble=%v\n", i, trunc(b.OCR), b.Box, r)
			} else {
				fmt.Printf("p%02d %-24q text=%v no bubble (%s)\n", i, trunc(b.OCR), b.Box, f.Reason)
			}
		}
		fmt.Printf("page %d: %d/%d fitted\n", i, found, len(blocks))
		name := filepath.Join(outDir, fmt.Sprintf("p%02d.png", i))
		fh, err := os.Create(name)
		if err != nil {
			panic(err)
		}
		png.Encode(fh, out)
		fh.Close()
	}
}

func drawRect(dst *image.RGBA, r image.Rectangle, c color.RGBA) {
	for x := r.Min.X; x < r.Max.X; x++ {
		for _, y := range []int{r.Min.Y, r.Max.Y - 1} {
			if image.Pt(x, y).In(dst.Bounds()) {
				dst.SetRGBA(x, y, c)
			}
		}
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for _, x := range []int{r.Min.X, r.Max.X - 1} {
			if image.Pt(x, y).In(dst.Bounds()) {
				dst.SetRGBA(x, y, c)
			}
		}
	}
}

func decode(b []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(b))
	return img, err
}

func trunc(s string) string {
	r := []rune(s)
	if len(r) > 12 {
		return string(r[:12])
	}
	return s
}
