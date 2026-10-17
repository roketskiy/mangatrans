package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"mangatrans/internal/bubble"
	"mangatrans/internal/cbz"
	"mangatrans/internal/detect"
	"mangatrans/internal/infer"
	"mangatrans/internal/inpaint"
	"mangatrans/internal/llm"
	"mangatrans/internal/render"
	"mangatrans/internal/translate"
)

type paths struct {
	llamaExe  string
	ocrModel  string
	ocrMMProj string
	mtModel   string
	detModel  string
	lapModel  string
	ortDir    string
	cudaDir   string
	fontPath  string
}

func main() {
	in := flag.String("in", "", "input .cbz/.zip")
	out := flag.String("out", "", "output .cbz")
	work := flag.String("work", "", "scratch dir for debug images")
	limit := flag.Int("limit", 0, "process only the first N pages (0 = all)")
	conf := flag.Float64("conf", 0.35, "detection confidence threshold")
	mtModel := flag.String("mt-model", `E:\work\transl\models\Hy-MT2-1.8B-UD-Q4_K_XL.gguf`, "translation gguf")
	fontPath := flag.String("font", `C:\Windows\Fonts\msyhbd.ttc`, "font file for translations")
	glossaryPath := flag.String("glossary", "", "term glossary: JSON map or source=target lines")
	benchPath := flag.String("bench-json", "", "write stage timings as JSON to this path")
	flag.Parse()
	if *in == "" || *out == "" {
		log.Fatal("usage: mangatrans -in book.cbz -out book.zh.cbz [-work dir] [-limit N]")
	}
	started := time.Now()
	b := &bench{}

	p := paths{
		llamaExe:  envOr("MANGATRANS_LLAMA", `C:\Users\Public\Temp\opencode\llamacpp\llama\llama-server.exe`),
		ocrModel:  envOr("MANGATRANS_OCR_MODEL", `E:\work\transl\models\PaddleOCR-VL-manga-Q4_K_M.gguf`),
		ocrMMProj: envOr("MANGATRANS_MMPROJ", `E:\work\transl\models\mmproj-Q8_0.gguf`),
		mtModel:   *mtModel,
		detModel:  envOr("MANGATRANS_DET_MODEL", `E:\work\transl\.cache\onnx-probe\comictextdetector.onnx`),
		lapModel:  envOr("MANGATRANS_LAMA_MODEL", `E:\work\transl\.cache\onnx-probe\lama-manga.onnx`),
		ortDir:    envOr("MANGATRANS_ORT_DIR", `E:\work\transl\runtime\ort-cuda13`),
		cudaDir:   envOr("MANGATRANS_CUDA_DIR", `E:\Python312\Lib\site-packages\torch\lib`),
		fontPath:  *fontPath,
	}

	if *work != "" {
		os.MkdirAll(*work, 0o755)
	}

	tRead := time.Now()
	pages, err := cbz.ReadCBZ(*in)
	if err != nil {
		log.Fatalf("read input: %v", err)
	}
	b.ReadMS = time.Since(tRead).Milliseconds()
	if *limit > 0 && len(pages) > *limit {
		pages = pages[:*limit]
	}
	log.Printf("pages: %d", len(pages))
	b.Pages = len(pages)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
	defer cancel()

	// Stage A: OCR. One llama-server instance, then released so the
	// translation model can take the same VRAM.
	log.Printf("starting OCR server")
	tLoad := time.Now()
	ocrSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.ocrModel, MMProj: p.ocrMMProj,
		LogPath: filepath.Join(os.TempDir(), "mangatrans-ocr.log"),
		Args:    []string{"-ngl", "99", "-c", "4096", "--temp", "0", "--jinja"},
	})
	if err != nil {
		log.Fatalf("start OCR server: %v", err)
	}
	b.LoadOCRMS = time.Since(tLoad).Milliseconds()

	rt, err := infer.Env(p.ortDir, p.cudaDir)
	if err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer rt.Destroy()

	tLoad = time.Now()
	det, err := detect.Load(p.detModel, true, 3<<30)
	if err != nil {
		log.Fatalf("load detector: %v", err)
	}
	b.LoadDetMS = time.Since(tLoad).Milliseconds()
	defer det.Close()

	type pageResult struct {
		index  int
		name   string
		img    image.Image
		mask   *image.Gray
		blocks []render.Block
		texts  []translate.Block
		ocr    []string
	}
	results := make([]*pageResult, 0, len(pages))

	detParams := detect.DefaultParams()
	detParams.ConfThreshold = float32(*conf)

	ocrStart := time.Now()
	for i, pg := range pages {
		tDecode := time.Now()
		img, err := decodeImage(pg.Bytes)
		if err != nil {
			log.Printf("page %d (%s): decode failed: %v", i, pg.Name, err)
			continue
		}
		b.DecodeMS += time.Since(tDecode).Milliseconds()
		tDetect := time.Now()
		res, err := det.Run(img, detParams)
		if err != nil {
			log.Printf("page %d: detect failed: %v", i, err)
			continue
		}
		b.DetectMS += time.Since(tDetect).Milliseconds()

		pr := &pageResult{index: i, name: pg.Name, img: img, mask: res.Mask}
		for j, box := range res.Boxes {
			crop := cropRect(img, box.X, box.Y, box.W, box.H, 6)
			if crop == nil {
				continue
			}
			tOCR := time.Now()
			text, err := ocrBox(ctx, ocrSrv, crop)
			b.OCRMS += time.Since(tOCR).Milliseconds()
			b.OCRBoxes++
			if err != nil {
				log.Printf("page %d box %d: ocr failed: %v", i, j, err)
				continue
			}
			text = strings.TrimSpace(text)
			if text == "" || isNoise(text) {
				continue
			}
			id := len(pr.blocks) + 1
			pr.blocks = append(pr.blocks, render.Block{
				Box:      image.Rect(int(box.X), int(box.Y), int(box.X+box.W), int(box.Y+box.H)),
				Vertical: box.H > box.W,
			})
			pr.texts = append(pr.texts, translate.Block{ID: id, Text: text})
			pr.ocr = append(pr.ocr, text)
		}
		log.Printf("page %d: %d boxes, %d with text", i, len(res.Boxes), len(pr.texts))
		results = append(results, pr)
	}
	ocrSrv.Kill()
	log.Printf("OCR done in %s", time.Since(ocrStart).Round(time.Second))

	// Stage B: translation, batched across pages so the model sees context.
	log.Printf("starting translation server")
	tLoad = time.Now()
	mtSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.mtModel,
		LogPath: filepath.Join(os.TempDir(), "mangatrans-mt.log"),
		Args:    []string{"-ngl", "99", "-c", "8192", "--jinja"},
	})
	if err != nil {
		log.Fatalf("start translation server: %v", err)
	}
	b.LoadMTMS = time.Since(tLoad).Milliseconds()
	defer mtSrv.Kill()

	var glossary map[string]string
	if *glossaryPath != "" {
		glossary, err = translate.LoadGlossary(*glossaryPath)
		if err != nil {
			log.Fatalf("load glossary: %v", err)
		}
		log.Printf("glossary: %d terms", len(glossary))
	}

	tr := translate.New(mtSrv, glossary)
	const perBatch = 12
	mtStart := time.Now()
	for _, pr := range results {
		translated := make(map[int]string, len(pr.texts))
		for start := 0; start < len(pr.texts); start += perBatch {
			end := start + perBatch
			if end > len(pr.texts) {
				end = len(pr.texts)
			}
			batch := pr.texts[start:end]
			tTr := time.Now()
			got, err := tr.TranslateBatch(ctx, batch)
			b.TranslateMS += time.Since(tTr).Milliseconds()
			b.TranslateBlocks += len(batch)
			if err != nil {
				log.Printf("page %d: translate batch failed: %v", pr.index, err)
				continue
			}
			for id, txt := range got {
				translated[id] = txt
			}
		}
		for k := range pr.blocks {
			id := k + 1
			if txt, ok := translated[id]; ok && txt != "" {
				pr.blocks[k].Text = txt
			}
		}
	}
	// Single-line retry for blocks the batch model left empty. Name lists and
	// lines that begin with ellipses often trigger an empty string, and
	// keeping the Japanese there is worse than one extra request.
	retries := 0
	for _, pr := range results {
		for k := range pr.blocks {
			if pr.blocks[k].Text != "" {
				continue
			}
			tRetry := time.Now()
			txt, err := tr.TranslateOne(ctx, pr.ocr[k])
			b.TranslateMS += time.Since(tRetry).Milliseconds()
			if err != nil {
				log.Printf("page %d block %d: single retry failed: %v", pr.index, k+1, err)
				continue
			}
			if txt != "" && txt != pr.ocr[k] {
				pr.blocks[k].Text = txt
				retries++
			}
		}
	}
	log.Printf("single-line retries: %d", retries)
	for _, pr := range results {
		for _, blk := range pr.blocks {
			if blk.Text != "" {
				b.Blocks++
				b.Chars += len([]rune(blk.Text))
			}
		}
	}

	log.Printf("translation done in %s", time.Since(mtStart).Round(time.Second))
	for _, pr := range results {
		n := 0
		for _, b := range pr.blocks {
			if b.Text != "" {
				n++
			}
		}
		log.Printf("page %d: translated %d/%d blocks", pr.index, n, len(pr.blocks))
	}

	// Stage C: erase original text, then draw translations.
	log.Printf("starting inpainting")
	tLoad = time.Now()
	lap, err := inpaint.Load(p.lapModel, true, 3<<30)
	if err != nil {
		log.Fatalf("load inpaint model: %v", err)
	}
	b.LoadLamaMS = time.Since(tLoad).Milliseconds()
	defer lap.Close()

	renderOpt := render.Options{
		FontPath:  p.fontPath,
		MinSize:   1.2,
		MaxSize:   16.0,
		Padding:   3,
		TextColor: color.RGBA{A: 255},
	}

	outPages := make([]cbz.Page, 0, len(pages))
	byName := make(map[string]*pageResult, len(results))
	for _, pr := range results {
		byName[pr.name] = pr
	}

	finishStart := time.Now()
	for i, pg := range pages {
		pr := byName[pg.Name]
		if pr == nil {
			outPages = append(outPages, pg)
			continue
		}
		// Only erase boxes that actually have a translation. Otherwise a
		// failed or skipped block would lose its original text and gain
		// nothing, which is strictly worse than leaving the page alone.
		active := make([]render.Block, 0, len(pr.blocks))
		for _, b := range pr.blocks {
			if b.Text != "" {
				active = append(active, b)
			}
		}
		if len(active) == 0 {
			outPages = append(outPages, pg)
			continue
		}
		// Build the inpaint mask from the original OCR boxes, before bubble
		// fitting changes anything. The mask must cover every original glyph,
		// including those outside the bubble's largest inner rectangle.
		mask := restrictMask(pr.mask, active)
		// Centre the translation in the speech bubble, but keep the original
		// text footprint for sizing. A longer translation then cannot grow
		// past the space the artist left for text.
		finder := bubble.Find(pr.img, 210)
		usedBubble := make(map[image.Rectangle]bool, len(active))
		for k := range active {
			if r, ok := finder.Fit(active[k].Box); ok && !usedBubble[r] {
				active[k].Anchor = r
				usedBubble[r] = true
			}
			sample := active[k].Box
			if !active[k].Anchor.Empty() {
				sample = active[k].Anchor
			}
			active[k].TextColor = textColor(pr.img, sample)
		}
		if *work != "" {
			writePNG(filepath.Join(*work, fmt.Sprintf("mask%02d.png", i)), mask)
			dumps := make([]blockDump, 0, len(pr.blocks))
			for k, b := range pr.blocks {
				dumps = append(dumps, blockDump{Box: b.Box, Vertical: b.Vertical, OCR: pr.ocr[k], ZH: b.Text})
			}
			writeBlocks(filepath.Join(*work, fmt.Sprintf("blocks%02d.json", i)), dumps)
		}
		tInpaint := time.Now()
		clean, err := lap.Inpaint(pr.img, mask)
		b.InpaintMS += time.Since(tInpaint).Milliseconds()
		if err != nil {
			log.Printf("page %d: inpaint failed: %v", i, err)
			clean = pr.img
		}
		tRender := time.Now()
		final, err := render.Render(clean, active, renderOpt)
		b.RenderMS += time.Since(tRender).Milliseconds()
		if err != nil {
			log.Printf("page %d: render failed: %v", i, err)
			final = clean
		}
		tEncode := time.Now()
		data, err := encodeJPEG(final, 92)
		b.EncodeMS += time.Since(tEncode).Milliseconds()
		if err != nil {
			log.Printf("page %d: encode failed: %v", i, err)
			outPages = append(outPages, pg)
			continue
		}
		outPages = append(outPages, cbz.Page{Name: pg.Name, Bytes: data})
		if *work != "" {
			os.WriteFile(filepath.Join(*work, fmt.Sprintf("p%02d.jpg", i)), data, 0o644)
		}
	}
	log.Printf("finish done in %s", time.Since(finishStart).Round(time.Second))

	tWrite := time.Now()
	if err := cbz.WriteCBZ(*out, outPages); err != nil {
		log.Fatalf("write output: %v", err)
	}
	b.WriteMS = time.Since(tWrite).Milliseconds()
	log.Printf("wrote %s (%d pages)", *out, len(outPages))

	// Read the output back and decode every page. This catches a truncated
	// zip or an encoder failure before the user opens the file.
	tValidate := time.Now()
	if check, err := cbz.ReadCBZ(*out); err != nil {
		log.Printf("validate output: %v", err)
	} else {
		bad := 0
		for _, p := range check {
			if _, err := decodeImage(p.Bytes); err != nil {
				bad++
			}
		}
		if len(check) != len(outPages) {
			log.Printf("validate output: %d pages, expected %d", len(check), len(outPages))
		} else if bad > 0 {
			log.Printf("validate output: %d of %d pages do not decode", bad, len(check))
		} else {
			log.Printf("validate output: %d pages, all decode", len(check))
		}
	}
	b.ValidateMS = time.Since(tValidate).Milliseconds()
	b.report(log.Printf, time.Since(started))
	if *benchPath != "" {
		if err := b.writeJSON(*benchPath); err != nil {
			log.Printf("write benchmark json: %v", err)
		} else {
			log.Printf("benchmark json: %s", *benchPath)
		}
	}
}

func isNoise(s string) bool {
	// Pure punctuation or repeated symbols carry no translatable content.
	hasLetter := false
	for _, r := range s {
		if (r >= 'ぁ' && r <= 'ん') || (r >= 'ァ' && r <= 'ン') ||
			(r >= '一' && r <= '鿿') || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') {
			hasLetter = true
			break
		}
	}
	return !hasLetter
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// bench accumulates per-stage timings for the end-of-run report. The counters
// make throughput comparable between books of different sizes.
type bench struct {
	Pages           int   `json:"pages"`
	Blocks          int   `json:"blocks"`
	Chars           int   `json:"chars"`
	ReadMS          int64 `json:"read_ms"`
	DecodeMS        int64 `json:"decode_ms"`
	DetectMS        int64 `json:"detect_ms"`
	OCRMS           int64 `json:"ocr_ms"`
	OCRBoxes        int   `json:"ocr_boxes"`
	TranslateMS     int64 `json:"translate_ms"`
	TranslateBlocks int   `json:"translate_blocks"`
	InpaintMS       int64 `json:"inpaint_ms"`
	RenderMS        int64 `json:"render_ms"`
	EncodeMS        int64 `json:"encode_ms"`
	WriteMS         int64 `json:"write_ms"`
	ValidateMS      int64 `json:"validate_ms"`
	LoadOCRMS       int64 `json:"load_ocr_ms"`
	LoadDetMS       int64 `json:"load_det_ms"`
	LoadMTMS        int64 `json:"load_mt_ms"`
	LoadLamaMS      int64 `json:"load_lama_ms"`
	TotalMS         int64 `json:"total_ms"`
}

func (b *bench) report(logf func(string, ...any), total time.Duration) {
	b.TotalMS = total.Milliseconds()
	pages := float64(b.Pages)

	logf("benchmark: pages=%d blocks=%d chars=%d total=%s",
		b.Pages, b.Blocks, b.Chars, total.Round(time.Millisecond))
	logf("  model load: ocr=%s det=%s mt=%s lama=%s",
		ms(b.LoadOCRMS), ms(b.LoadDetMS), ms(b.LoadMTMS), ms(b.LoadLamaMS))
	logf("  io: read=%s write=%s validate=%s decode=%s encode=%s",
		ms(b.ReadMS), ms(b.WriteMS), ms(b.ValidateMS), ms(b.DecodeMS), ms(b.EncodeMS))
	if b.Pages > 0 {
		logf("  detect: %s (%s/page)", ms(b.DetectMS), ms(int64(float64(b.DetectMS)/pages)))
	}
	if b.OCRBoxes > 0 {
		logf("  ocr: %s for %d boxes (%s/box, %.1f boxes/s)",
			ms(b.OCRMS), b.OCRBoxes, ms(b.OCRMS/int64(b.OCRBoxes)), rate(b.OCRBoxes, b.OCRMS))
	}
	if b.TranslateBlocks > 0 {
		logf("  translate: %s for %d blocks (%s/block, %.1f blocks/s)",
			ms(b.TranslateMS), b.TranslateBlocks, ms(b.TranslateMS/int64(b.TranslateBlocks)), rate(b.TranslateBlocks, b.TranslateMS))
	}
	if b.Pages > 0 {
		logf("  finish: inpaint=%s render=%s (%s/page total)",
			ms(b.InpaintMS), ms(b.RenderMS), ms(int64(float64(b.InpaintMS+b.RenderMS)/pages)))
	}
	if b.TotalMS > 0 && b.Pages > 0 {
		logf("  wall: %s/page, %.2f pages/min", ms(b.TotalMS/int64(b.Pages)), 60000/float64(b.TotalMS)*pages)
	}
}

func (b *bench) writeJSON(path string) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func ms(v int64) time.Duration { return time.Duration(v) * time.Millisecond }

func rate(n int, totalMS int64) float64 {
	if totalMS <= 0 {
		return 0
	}
	return float64(n) / (float64(totalMS) / 1000)
}
