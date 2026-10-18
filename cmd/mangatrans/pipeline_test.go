package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"sync/atomic"
	"testing"
	"time"

	"mangatrans/internal/bubble"
	"mangatrans/internal/render"
)

func TestOCRWorkersPreserveBoxOrderAndBoundConcurrency(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 100, 40))
	for y := 0; y < 40; y++ {
		for x := 0; x < 100; x++ {
			im.SetRGBA(x, y, color.RGBA{R: uint8(x), A: 255})
		}
	}
	boxes := []bubble.TextRegion{{Box: image.Rect(12, 12, 20, 20)}, {Box: image.Rect(42, 12, 50, 20)}, {Box: image.Rect(72, 12, 80, 20)}, {Box: image.Rect(200, 12, 208, 20)}}
	var active, peak atomic.Int32
	results := recognizeRegions(context.Background(), im, boxes, 2,
		func(ctx context.Context, crop image.Image) (string, error) {
			n := active.Add(1)
			for old := peak.Load(); n > old; old = peak.Load() {
				if peak.CompareAndSwap(old, n) {
					break
				}
			}
			defer active.Add(-1)
			r, _, _, _ := crop.At(0, 0).RGBA()
			if r>>8 == 6 {
				time.Sleep(30 * time.Millisecond)
			} else {
				time.Sleep(5 * time.Millisecond)
			}
			return fmt.Sprint(r >> 8), nil
		})
	for i, want := range []string{"6", "36", "66"} {
		if results[i].text != want || results[i].err != nil {
			t.Fatalf("box %d: %+v; want %s", i, results[i], want)
		}
	}
	if !results[3].cropSkipped {
		t.Fatal("out-of-bounds crop must be skipped")
	}
	if peak.Load() != 2 {
		t.Fatalf("expected two concurrent requests, peak=%d", peak.Load())
	}
}

func TestTranslationBatchesKeepIDsAndPageReferences(t *testing.T) {
	pages := []*pageResult{{index: 0, ocr: []string{"甲", "乙"}}, {index: 1, ocr: []string{"丙", "丁"}}, {index: 4, ocr: []string{"戊"}}}
	batches := planTranslations(pages, 3, 2)
	if len(batches) != 3 {
		t.Fatalf("got %d batches, want 3", len(batches))
	}
	count := 0
	for _, batch := range batches {
		seen := map[int]bool{}
		if len(batch.blocks) > 3 {
			t.Fatal("batch exceeds size")
		}
		for _, block := range batch.blocks {
			if seen[block.ID] {
				t.Fatalf("duplicate id %d", block.ID)
			}
			seen[block.ID] = true
			count++
			loc := batch.locations[block.ID]
			var original string
			for _, page := range pages {
				if page.index == loc.page {
					original = page.ocr[loc.block]
				}
			}
			if original != block.Text {
				t.Fatalf("id %d mapped to wrong source", block.ID)
			}
		}
	}
	if count != 5 || batches[2].blocks[0].Text != "戊" {
		t.Fatal("lost text or crossed the page-span boundary")
	}
	separate := planTranslations(pages, 12, 1)
	if len(separate) != 3 {
		t.Fatal("pageSpan=1 must preserve per-page batches")
	}
}

func TestSplitRegionOCRAndEraseCannotConsumeSibling(t *testing.T) {
	page := image.NewRGBA(image.Rect(0, 0, 60, 50))
	mask := image.NewGray(page.Bounds())
	left, right := image.NewAlpha(page.Bounds()), image.NewAlpha(page.Bounds())
	for y := 0; y < 50; y++ {
		for x := 0; x < 60; x++ {
			page.Set(x, y, color.White)
			if x < 30 {
				left.SetAlpha(x, y, color.Alpha{A: 255})
			} else {
				right.SetAlpha(x, y, color.Alpha{A: 255})
			}
			if y >= 12 && y < 38 && ((x >= 20 && x < 25) || (x >= 35 && x < 40)) {
				page.Set(x, y, color.Black)
				mask.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	// Bounding rectangles overlap; only source ownership can keep the second
	// dialogue out of the first OCR request and out of its erase operation.
	box := image.Rect(10, 10, 50, 40)
	regions := []bubble.TextRegion{{Box: box, CropMask: right}, {Box: box, CropMask: left}}
	requests := 0
	results := recognizeRegions(context.Background(), page, regions, 1, func(_ context.Context, crop image.Image) (string, error) {
		wantX := 35
		otherX := 20
		if requests == 1 {
			wantX, otherX = otherX, wantX
		}
		// The crop origin is (4,4), including its six-pixel margin.
		if color.GrayModel.Convert(crop.At(wantX-4, 20-4)).(color.Gray).Y != 0 || color.GrayModel.Convert(crop.At(otherX-4, 20-4)).(color.Gray).Y != 255 {
			t.Fatal("OCR input contains sibling text or lost its own text")
		}
		requests++
		if requests == 2 {
			return "", fmt.Errorf("failed sibling")
		}
		return "first dialogue", nil
	})
	if requests != 2 || results[0].text != "first dialogue" || results[1].err == nil {
		t.Fatal("dialogues were not recognized independently")
	}
	erase := restrictMask(mask, []render.Block{{Box: box, SourceArea: right}})
	if erase.GrayAt(35, 20).Y != 255 || erase.GrayAt(20, 20).Y != 0 {
		t.Fatal("successful dialogue erased a failed sibling's source")
	}
	pages := []*pageResult{{index: 0, ocr: []string{"upper original", "lower original"}}}
	batches := planTranslations(pages, 12, 2)
	if len(batches) != 1 || len(batches[0].blocks) != 2 || batches[0].blocks[0].Text != "upper original" || batches[0].blocks[1].Text != "lower original" {
		t.Fatal("independent OCR dialogues were merged for translation")
	}
}
