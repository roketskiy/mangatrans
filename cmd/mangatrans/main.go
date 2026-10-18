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
	"runtime/pprof"
	"strings"
	"time"

	"mangatrans/internal/bubble"
	"mangatrans/internal/cbz"
	"mangatrans/internal/detect"
	"mangatrans/internal/infer"
	"mangatrans/internal/inpaint"
	"mangatrans/internal/llm"
	"mangatrans/internal/perf"
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
	if err := run(); err != nil {
		log.Printf("fatal: %v", err)
		os.Exit(1)
	}
}

func run() (err error) {
	perfStarted := perf.Now()
	in := flag.String("in", "", "input .cbz/.zip")
	out := flag.String("out", "", "output .cbz")
	work := flag.String("work", "", "scratch dir for debug images")
	limit := flag.Int("limit", 0, "process only the first N pages (0 = all)")
	conf := flag.Float64("conf", 0.35, "detection confidence threshold")
	mtModel := flag.String("mt-model", `E:\work\transl\models\Hy-MT2-1.8B-UD-Q4_K_XL.gguf`, "translation gguf")
	fontPath := flag.String("font", `C:\Windows\Fonts\msyhbd.ttc`, "font file for translations")
	directionFlag := flag.String("text-direction", "vertical", "translation layout: vertical, horizontal or auto")
	glossaryPath := flag.String("glossary", "", "term glossary: JSON map or source=target lines")
	benchPath := flag.String("bench-json", "", "write stage timings as JSON to this path")
	perfPath := flag.String("perf-json", "", "write monotonic operation/request timings and page outcomes, including failures")
	cpuProfile := flag.String("cpu-profile", "", "write CPU profile; use a separate diagnostic run")
	ocrWorkers := flag.Int("ocr-workers", 1, "concurrent OCR requests (1-4); ONNX sessions stay serial")
	batchPages := flag.Int("mt-batch-pages", 2, "maximum consecutive pages sharing one 12-block translation batch")
	flag.Parse()
	direction, directionErr := render.ParseDirection(*directionFlag)
	if directionErr != nil {
		return directionErr
	}
	trace := perf.New(*perfPath, perfStarted)
	defer func() {
		if v := recover(); v != nil {
			_ = trace.Finish(fmt.Errorf("panic: %v", v))
			panic(v)
		}
		if e := trace.Finish(err); e != nil {
			err = fmt.Errorf("write performance report: %w (pipeline error: %v)", e, err)
		}
	}()
	if *cpuProfile != "" {
		f, e := os.Create(*cpuProfile)
		if e != nil {
			return e
		}
		if e := pprof.StartCPUProfile(f); e != nil {
			f.Close()
			return e
		}
		defer func() { pprof.StopCPUProfile(); _ = f.Close() }()
	}
	if *in == "" || *out == "" {
		return fmt.Errorf("usage: mangatrans -in book.cbz -out book.zh.cbz [-work dir] [-limit N]")
	}
	if *ocrWorkers < 1 || *ocrWorkers > 4 || *batchPages < 1 || *batchPages > 4 {
		return fmt.Errorf("ocr-workers and mt-batch-pages must be in 1..4")
	}
	inAbs, err := filepath.Abs(*in)
	if err != nil {
		return fmt.Errorf("resolve input path: %w", err)
	}
	outAbs, err := filepath.Abs(*out)
	if err != nil {
		return fmt.Errorf("resolve output path: %w", err)
	}
	if strings.EqualFold(filepath.Clean(inAbs), filepath.Clean(outAbs)) {
		return fmt.Errorf("output path must differ from input path")
	}
	if st, err := os.Stat(outAbs); err == nil && st.IsDir() {
		return fmt.Errorf("output path is a directory")
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
	trace.FilesUsed(map[string]string{
		"llama_server": p.llamaExe, "ocr_model": p.ocrModel, "mmproj": p.ocrMMProj,
		"translation_model": p.mtModel, "detector": p.detModel, "inpaint_model": p.lapModel,
		"font": p.fontPath, "ort": filepath.Join(p.ortDir, "onnxruntime.dll"),
	})

	if *work != "" {
		_ = os.MkdirAll(*work, 0o755)
	}

	tRead := time.Now()
	endSpan := trace.Span("operation", "archive_read", -1, -1)
	pages, err := cbz.ReadCBZ(*in)
	endSpan(err)
	if err != nil {
		return fmt.Errorf("read input: %w", err)
	}
	b.ReadMS = time.Since(tRead).Milliseconds()
	if *limit > 0 && len(pages) > *limit {
		pages = pages[:*limit]
	}
	log.Printf("pages: %d", len(pages))
	b.Pages = len(pages)
	pageMetrics := make([]*perf.Page, len(pages))
	for i, pg := range pages {
		pageMetrics[i] = trace.Page(i, pg.Name)
	}

	// Keep every intermediate file for this run in one directory next to the
	// output and delete it on exit, including error and panic paths. Server
	// logs never go to the user's C: temp directory, and a killed run leaves
	// at most one small .mangatrans directory, never page images in C:.
	outDir := filepath.Dir(outAbs)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return fmt.Errorf("create output directory: %w", err)
	}
	runDir := filepath.Join(outDir, "."+filepath.Base(outAbs)+".mangatrans")
	// A forced kill or power loss can leave this directory behind. The UI
	// allows only one run at a time, so the next run may safely replace it.
	_ = os.RemoveAll(runDir)
	if err := os.MkdirAll(runDir, 0o755); err != nil {
		return fmt.Errorf("create run temp dir: %w", err)
	}
	defer os.RemoveAll(runDir)
	// Keep the run directory clean, but show the last server lines in the
	// main log before deleting it. Otherwise a start failure only points at a
	// log file that no longer exists by the time the user opens it.
	defer func() {
		if err == nil {
			return
		}
		for _, name := range []string{"ocr.log", "mt.log"} {
			if tail := readTail(filepath.Join(runDir, name), 24); tail != "" {
				log.Printf("--- %s (last lines) ---\n%s", name, tail)
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
	defer cancel()
	ctx = perf.WithRecorder(ctx, trace)

	// Stage A: OCR. One llama-server instance, then released so the
	// translation model can take the same VRAM.
	log.Printf("starting OCR server")
	tLoad := time.Now()
	endSpan = trace.Span("operation", "load_ocr", -1, -1)
	ocrSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.ocrModel, MMProj: p.ocrMMProj,
		LogPath: filepath.Join(runDir, "ocr.log"),
		Args:    []string{"-ngl", "99", "-c", "4096", "--temp", "0", "--jinja"},
	})
	endSpan(err)
	if err != nil {
		return fmt.Errorf("start OCR server: %w", err)
	}
	// Kill is idempotent. Register it so every error path also stops the
	// server, even before the normal OCR stage has finished.
	defer func() {
		end := trace.Span("operation", "cleanup_ocr", -1, -1)
		ocrSrv.Kill()
		end(nil)
	}()
	b.LoadOCRMS = time.Since(tLoad).Milliseconds()

	endSpan = trace.Span("operation", "init_runtime", -1, -1)
	rt, err := infer.Env(p.ortDir, p.cudaDir)
	endSpan(err)
	if err != nil {
		return fmt.Errorf("init onnxruntime: %w", err)
	}
	defer func() {
		end := trace.Span("operation", "cleanup_runtime", -1, -1)
		end(rt.Destroy())
	}()

	tLoad = time.Now()
	endSpan = trace.Span("operation", "load_detect", -1, -1)
	det, err := detect.Load(p.detModel, true, 3<<30)
	endSpan(err)
	if err != nil {
		return fmt.Errorf("load detector: %w", err)
	}
	b.LoadDetMS = time.Since(tLoad).Milliseconds()
	detOpen := true
	closeDetector := func() {
		if !detOpen {
			return
		}
		detOpen = false
		end := trace.Span("operation", "unload_detect", -1, -1)
		end(det.Close())
	}
	defer closeDetector()
	results := make([]*pageResult, 0, len(pages))

	detParams := detect.DefaultParams()
	detParams.ConfThreshold = float32(*conf)

	ocrStart := time.Now()
	for i, pg := range pages {
		log.Printf("progress ocr %d/%d", i+1, len(pages))
		tDecode := time.Now()
		endSpan = trace.Span("operation", "decode", i, -1)
		img, err := decodeImage(pg.Bytes)
		endSpan(err)
		if err != nil {
			pageMetrics[i].Outcome = "decode_failed"
			log.Printf("page %d (%s): decode failed: %v", i, pg.Name, err)
			continue
		}
		b.DecodeMS += time.Since(tDecode).Milliseconds()
		pageMetrics[i].Width, pageMetrics[i].Height = img.Bounds().Dx(), img.Bounds().Dy()
		tDetect := time.Now()
		endSpan = trace.Span("operation", "detect", i, -1)
		res, err := det.Run(img, detParams)
		endSpan(err)
		if err != nil {
			pageMetrics[i].Outcome = "detect_failed"
			log.Printf("page %d: detect failed: %v", i, err)
			continue
		}
		b.DetectMS += time.Since(tDetect).Milliseconds()
		pageMetrics[i].Detected = len(res.Boxes)
		endSpan = trace.Span("operation", "split_dialogues", i, -1)
		parentBoxes := make([]image.Rectangle, len(res.Boxes))
		for j, box := range res.Boxes {
			parentBoxes[j] = image.Rect(int(box.X), int(box.Y), int(box.X+box.W), int(box.Y+box.H))
		}
		regions := bubble.SplitTextRegions(img, res.Mask, parentBoxes)
		pageMetrics[i].OCRRegions = len(regions)
		endSpan(nil)

		pr := &pageResult{index: i, mask: res.Mask}
		endOCRPage := trace.Span("operation", "ocr_page", i, -1)
		ocrResults := recognizeRegions(perf.WithLocation(ctx, i, -1), img, regions, *ocrWorkers,
			func(ctx context.Context, crop image.Image) (string, error) { return ocrBox(ctx, ocrSrv, crop) })
		for j, got := range ocrResults {
			region := regions[j]
			if got.cropSkipped {
				pageMetrics[i].CropSkipped++
				continue
			}
			b.OCRMS += got.durationNS / 1e6
			b.OCRBoxes++
			if got.err != nil {
				log.Printf("page %d box %d: ocr failed: %v", i, j, got.err)
				pageMetrics[i].OCRFailed++
				continue
			}
			text := cleanOCR(got.text)
			if text == "" || isNoise(text) {
				pageMetrics[i].OCRRejected++
				continue
			}
			pr.blocks = append(pr.blocks, render.Block{
				Box:        region.Box,
				Area:       region.Area,
				SourceArea: region.CropMask,
				Vertical:   direction.IsVertical(region.Box.Dy() > region.Box.Dx()),
			})
			pr.ocr = append(pr.ocr, text)
			pageMetrics[i].OCRAccepted++
		}
		endOCRPage(nil)
		log.Printf("page %d: %d detector boxes, %d dialogue regions, %d with text", i, len(res.Boxes), len(regions), len(pr.ocr))
		results = append(results, pr)
	}
	endSpan = trace.Span("operation", "unload_ocr", -1, -1)
	ocrSrv.Kill()
	endSpan(nil)
	closeDetector()
	log.Printf("progress ocr %d/%d", len(pages), len(pages))
	log.Printf("OCR done in %s", time.Since(ocrStart).Round(time.Second))

	// Stage B: translation, batched across pages so the model sees context.
	log.Printf("starting translation server")
	tLoad = time.Now()
	endSpan = trace.Span("operation", "load_translate", -1, -1)
	mtSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.mtModel,
		LogPath: filepath.Join(runDir, "mt.log"),
		Args:    []string{"-ngl", "99", "-c", "8192", "--jinja"},
	})
	endSpan(err)
	if err != nil {
		return fmt.Errorf("start translation server: %w", err)
	}
	b.LoadMTMS = time.Since(tLoad).Milliseconds()
	defer func() {
		end := trace.Span("operation", "cleanup_translate", -1, -1)
		mtSrv.Kill()
		end(nil)
	}()

	var glossary map[string]string
	if *glossaryPath != "" {
		glossary, err = translate.LoadGlossary(*glossaryPath)
		if err != nil {
			return fmt.Errorf("load glossary: %w", err)
		}
		log.Printf("glossary: %d terms", len(glossary))
	}

	tr := translate.New(mtSrv, glossary)
	const perBatch = 12
	mtStart := time.Now()
	byIndex := make(map[int]*pageResult, len(results))
	for _, pr := range results {
		byIndex[pr.index] = pr
	}
	for batchIndex, batch := range planTranslations(results, perBatch, *batchPages) {
		first := batch.locations[batch.blocks[0].ID]
		log.Printf("progress translate %d/%d", first.page+1, len(pages))
		tTr := time.Now()
		endSpan = trace.Span("operation", "translate_batch", -1, batchIndex)
		got, err := tr.TranslateBatch(perf.WithLocation(ctx, -1, batchIndex), batch.blocks)
		endSpan(err)
		b.TranslateMS += time.Since(tTr).Milliseconds()
		b.TranslateBlocks += len(batch.blocks)
		if err != nil {
			log.Printf("batch %d: translate failed: %v", batchIndex, err)
			continue
		}
		for id, txt := range got {
			if loc, ok := batch.locations[id]; ok && txt != "" {
				byIndex[loc.page].blocks[loc.block].Text = txt
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
			endSpan = trace.Span("operation", "translate_retry", pr.index, k)
			txt, err := tr.TranslateOne(perf.WithLocation(ctx, pr.index, k), pr.ocr[k])
			endSpan(err)
			b.TranslateMS += time.Since(tRetry).Milliseconds()
			if err != nil {
				log.Printf("page %d block %d: single retry failed: %v", pr.index, k+1, err)
				continue
			}
			if !translate.Untranslated(pr.ocr[k], txt) {
				pr.blocks[k].Text = txt
				retries++
			}
		}
	}
	log.Printf("single-line retries: %d", retries)
	for _, pr := range results {
		for _, blk := range pr.blocks {
			if blk.Text != "" {
				pageMetrics[pr.index].Translated++
				b.Blocks++
				b.Chars += len([]rune(blk.Text))
			}
		}
	}

	log.Printf("progress translate %d/%d", len(pages), len(pages))
	log.Printf("translation done in %s", time.Since(mtStart).Round(time.Second))
	skipped := 0
	for _, pr := range results {
		n := 0
		for _, b := range pr.blocks {
			if b.Text != "" {
				n++
			}
		}
		skipped += len(pr.blocks) - n
		log.Printf("page %d: translated %d/%d blocks", pr.index, n, len(pr.blocks))
	}
	if skipped > 0 {
		log.Printf("warning: %d text blocks kept the original text", skipped)
	}
	endSpan = trace.Span("operation", "unload_translate", -1, -1)
	mtSrv.Kill()
	endSpan(nil)

	// Stage C: erase original text, then draw translations.
	log.Printf("starting inpainting")
	log.Printf("progress inpaint 0/%d", len(pages))
	tLoad = time.Now()
	endSpan = trace.Span("operation", "load_inpaint", -1, -1)
	lap, err := inpaint.Load(p.lapModel, true, 3<<30)
	endSpan(err)
	if err != nil {
		return fmt.Errorf("load inpaint model: %w", err)
	}
	b.LoadLamaMS = time.Since(tLoad).Milliseconds()
	defer func() {
		end := trace.Span("operation", "cleanup_inpaint", -1, -1)
		end(lap.Close())
	}()

	renderOpt := render.Options{
		FontPath:  p.fontPath,
		MinSize:   1.2,
		MaxSize:   16.0,
		Padding:   3,
		TextColor: color.RGBA{A: 255},
	}

	outPages := make([]cbz.Page, 0, len(pages))

	finishStart := time.Now()
	reportPage := func(i int) {
		log.Printf("progress inpaint %d/%d", i+1, len(pages))
	}
	for i, pg := range pages {
		pr := byIndex[i]
		if pr == nil {
			outPages = append(outPages, pg)
			reportPage(i)
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
			pageMetrics[i].Outcome = "no_translation"
			pr.mask, pr.ocr, pr.blocks = nil, nil, nil
			outPages = append(outPages, pg)
			reportPage(i)
			continue
		}
		endSpan = trace.Span("operation", "redecode", i, -1)
		img, err := decodeImage(pg.Bytes)
		endSpan(err)
		if err != nil {
			pageMetrics[i].Outcome = "decode_failed"
			pr.mask, pr.ocr, pr.blocks = nil, nil, nil
			outPages = append(outPages, pg)
			reportPage(i)
			continue
		}
		// Build the inpaint mask from the original OCR boxes, before bubble
		// fitting changes anything. The mask must cover every original glyph,
		// including those outside the bubble's largest inner rectangle.
		endSpan = trace.Span("operation", "mask_prepare", i, -1)
		mask := restrictMask(pr.mask, active)
		endSpan(nil)
		if *work != "" {
			writePNG(filepath.Join(*work, fmt.Sprintf("mask%02d.png", i)), mask)
			dumps := make([]blockDump, 0, len(pr.blocks))
			for k, b := range pr.blocks {
				dumps = append(dumps, blockDump{Box: b.Box, Vertical: b.Vertical, OCR: pr.ocr[k], ZH: b.Text})
			}
			writeBlocks(filepath.Join(*work, fmt.Sprintf("blocks%02d.json", i)), dumps)
		}
		tInpaint := time.Now()
		endSpan = trace.Span("operation", "inpaint", i, -1)
		clean, err := lap.Inpaint(perf.WithLocation(ctx, i, -1), img, mask)
		endSpan(err)
		pr.mask, pr.ocr, pr.blocks = nil, nil, nil
		b.InpaintMS += time.Since(tInpaint).Milliseconds()
		if err != nil {
			pageMetrics[i].Outcome = "inpaint_failed"
			// Never draw a translation over the un-erased original: that
			// would leave two overlapping texts. Keep the page as it was.
			log.Printf("page %d: inpaint failed, keeping original page: %v", i+1, err)
			outPages = append(outPages, pg)
			reportPage(i)
			continue
		}
		// Original glyph strokes can connect to the outline and split a bubble's
		// white interior. Fit against the erased page, while retaining the source
		// boxes for ownership, sizing, and text colour.
		endSpan = trace.Span("operation", "layout_prepare", i, -1)
		finder := bubble.Find(clean, 210)
		boxes := make([]image.Rectangle, len(active))
		for k := range active {
			boxes[k] = active[k].Box
		}
		areas := finder.FitAreas(boxes)
		for k := range active {
			if active[k].Area == nil {
				active[k].Area = areas[k]
			}
			sample := active[k].Box
			if active[k].Area != nil {
				sample = active[k].Area.Bounds()
				perf.Count(perf.WithLocation(ctx, i, k), "layout_shaped_blocks", 1)
			} else {
				perf.Count(perf.WithLocation(ctx, i, k), "layout_box_fallbacks", 1)
			}
			active[k].TextColor = textColor(img, sample)
		}
		endSpan(nil)
		tRender := time.Now()
		endSpan = trace.Span("operation", "render", i, -1)
		final, err := render.Render(clean, active, renderOpt)
		endSpan(err)
		b.RenderMS += time.Since(tRender).Milliseconds()
		if err != nil {
			pageMetrics[i].Outcome = "render_failed"
			// The original text was already erased in memory; revert to the
			// input page so the reader does not get a blank or half-drawn page.
			log.Printf("page %d: render failed, keeping original page: %v", i+1, err)
			outPages = append(outPages, pg)
			reportPage(i)
			continue
		}
		tEncode := time.Now()
		endSpan = trace.Span("operation", "encode", i, -1)
		data, err := encodeJPEG(final, 92)
		endSpan(err)
		b.EncodeMS += time.Since(tEncode).Milliseconds()
		if err != nil {
			log.Printf("page %d: encode failed, keeping original page: %v", i+1, err)
			pageMetrics[i].Outcome = "encode_failed"
			outPages = append(outPages, pg)
			reportPage(i)
			continue
		}
		outPages = append(outPages, cbz.Page{Name: pg.Name, Bytes: data})
		pageMetrics[i].Outcome = "rendered"
		pageMetrics[i].Rendered = len(active)
		if *work != "" {
			_ = os.WriteFile(filepath.Join(*work, fmt.Sprintf("p%02d.jpg", i)), data, 0o644)
		}
		reportPage(i)
	}
	log.Printf("finish done in %s", time.Since(finishStart).Round(time.Second))

	tWrite := time.Now()
	validateMS := int64(0)
	checkCount := 0
	endSpan = trace.Span("operation", "archive_write_including_validate", -1, -1)
	err = cbz.WriteCBZValidated(*out, outPages, func(part string) error {
		endValidate := trace.Span("nested", "archive_validate", -1, -1)
		var validationErr error
		defer func() { endValidate(validationErr) }()
		tValidate := time.Now()
		defer func() {
			validateMS = time.Since(tValidate).Milliseconds()
		}()
		// Read the completed .part back and decode every page before the
		// rename. A truncated zip or bad encoder never replaces a good file.
		check, err := cbz.ReadCBZ(part)
		if err != nil {
			validationErr = fmt.Errorf("read back: %w", err)
			return validationErr
		}
		if len(check) != len(outPages) {
			validationErr = fmt.Errorf("%d pages, expected %d", len(check), len(outPages))
			return validationErr
		}
		bad := 0
		for _, p := range check {
			if _, err := decodeImage(p.Bytes); err != nil {
				bad++
			}
		}
		if bad > 0 {
			validationErr = fmt.Errorf("%d of %d pages do not decode", bad, len(check))
			return validationErr
		}
		checkCount = len(check)
		return nil
	})
	endSpan(err)
	b.WriteMS = time.Since(tWrite).Milliseconds() - validateMS
	b.ValidateMS = validateMS
	if err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	log.Printf("progress save %d/%d", len(outPages), len(outPages))
	log.Printf("wrote %s (%d pages)", *out, len(outPages))
	log.Printf("validate output: %d pages, all decode", checkCount)
	log.Printf("progress done %d/%d", len(outPages), len(outPages))
	b.report(log.Printf, time.Since(started))
	if *benchPath != "" {
		if err := b.writeJSON(*benchPath); err != nil {
			log.Printf("write benchmark json: %v", err)
		} else {
			log.Printf("benchmark json: %s", *benchPath)
		}
	}
	return nil
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
