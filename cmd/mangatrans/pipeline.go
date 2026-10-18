package main

import (
	"context"
	"image"
	"sync"

	"mangatrans/internal/bubble"
	"mangatrans/internal/perf"
	"mangatrans/internal/render"
	"mangatrans/internal/translate"
)

type pageResult struct {
	index  int
	mask   *image.Gray
	blocks []render.Block
	ocr    []string
}

type ocrResult struct {
	text        string
	err         error
	cropSkipped bool
	durationNS  int64
}

// Each worker owns one result slot. Detection and inpainting remain serial
// because their ONNX sessions share mutable input/output tensors.
func recognizeRegions(ctx context.Context, img image.Image, regions []bubble.TextRegion, workers int,
	recognize func(context.Context, image.Image) (string, error)) []ocrResult {
	results := make([]ocrResult, len(regions))
	jobs := make(chan int, len(regions))
	for i := range regions {
		jobs <- i
	}
	close(jobs)
	var wg sync.WaitGroup
	for range min(workers, len(regions)) {
		wg.Go(func() {
			for i := range jobs {
				region := regions[i]
				boxCtx := perf.WithItem(ctx, i)
				end := perf.ContextSpan(boxCtx, "nested", "crop")
				crop := region.Crop(img, 6)
				end(nil)
				if crop == nil {
					results[i].cropSkipped = true
					continue
				}
				if err := ctx.Err(); err != nil {
					results[i].err = err
					continue
				}
				started := perf.Now()
				end = perf.ContextSpan(boxCtx, "nested", "ocr")
				results[i].text, results[i].err = recognize(boxCtx, crop)
				end(results[i].err)
				results[i].durationNS = perf.Now() - started
			}
		})
	}
	wg.Wait()
	return results
}

// IDs are local to one request. Its location map disambiguates identical
// page-local IDs without making the model generate ever-growing numbers.
type translationLocation struct{ page, block int }
type translationBatch struct {
	blocks    []translate.Block
	locations map[int]translationLocation
}

func planTranslations(pages []*pageResult, size, pageSpan int) []translationBatch {
	var batches []translationBatch
	batch := translationBatch{locations: make(map[int]translationLocation)}
	firstPage, nextID := -1, 1
	flush := func() {
		if len(batch.blocks) == 0 {
			return
		}
		batches = append(batches, batch)
		batch = translationBatch{locations: make(map[int]translationLocation)}
		firstPage = -1
		nextID = 1
	}
	for _, page := range pages {
		for k, text := range page.ocr {
			if len(batch.blocks) == size || (firstPage >= 0 && page.index-firstPage >= pageSpan) {
				flush()
			}
			if firstPage < 0 {
				firstPage = page.index
			}
			batch.blocks = append(batch.blocks, translate.Block{ID: nextID, Text: text})
			batch.locations[nextID] = translationLocation{page: page.index, block: k}
			nextID++
		}
	}
	flush()
	return batches
}
