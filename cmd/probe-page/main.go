// Command probe-page runs ONE page through the full translation pipeline and
// dumps every stage's image plus a machine-readable record, so a bad page can
// be analysed without re-running the whole book.
//
//	probe-page -in page.png|book.cbz [-page 0] -out dir [-settings s.json] [-conf 0.35]
//
// Outputs in -out:
//
//	01-source.png   decoded input page
//	02-detect.png   detector line boxes over the source
//	03-regions.png  merged dialogue regions (green = translated, orange = rejected, red = failed)
//	04-ocr/NN.png   the exact crop sent to the OCR model
//	05-mask.png     erasure mask actually applied (translated blocks only)
//	06-clean.png    page after inpainting
//	07-layout.png   translated block boxes (yellow) and fitted layout areas (cyan) on the clean page
//	08-final.png    rendered result
//	record.json     per-stage timings, boxes, OCR/translation text, fitted areas, text colours
//	ocr.log, mt.log llama-server logs
//
// The pipeline mirrors cmd/mangatrans stage for stage. When the engine changes,
// update this tool in the same commit.
package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"mangatrans/internal/bubble"
	"mangatrans/internal/cbz"
	"mangatrans/internal/dbtext"
	"mangatrans/internal/infer"
	"mangatrans/internal/inpaint"
	"mangatrans/internal/llm"
	"mangatrans/internal/pageimage"
	"mangatrans/internal/render"
	"mangatrans/internal/translate"
	"mangatrans/settings"
)

// regionRecord is one merged dialogue region's full trail: box, OCR verdict,
// translation, and the layout decision the renderer made for it.
type regionRecord struct {
	Index    int     `json:"index"`
	Box      []int   `json:"box"`
	Status   string  `json:"status"` // accepted | noise | empty | ocr_failed | crop_skipped
	OCR      string  `json:"ocr,omitempty"`
	ZH       string  `json:"zh,omitempty"`
	Area     []int   `json:"area_bounds,omitempty"` // fitted layout area; absent = box fallback
	Vertical bool    `json:"vertical"`
	SizeHint float64 `json:"size_hint,omitempty"`
	Color    string  `json:"text_color,omitempty"`
	Note     string  `json:"note,omitempty"`
}

type record struct {
	Source     string           `json:"source"`
	Page       int              `json:"page"`
	Width      int              `json:"width"`
	Height     int              `json:"height"`
	DetBoxes   int              `json:"detector_boxes"`
	Regions    int              `json:"dialogue_regions"`
	Accepted   int              `json:"accepted_blocks"`
	Translated int              `json:"translated_blocks"`
	Rejected   int              `json:"rejected_blocks"`
	TimingMS   map[string]int64 `json:"timing_ms"`
	Note       string           `json:"note,omitempty"`
	Blocks     []regionRecord   `json:"blocks"`
}

func main() {
	if err := run(); err != nil {
		log.Printf("fatal: %v", err)
		os.Exit(1)
	}
}

func run() error {
	in := flag.String("in", "", "input page image (.png/.jpg/.webp) or .cbz/.zip")
	page := flag.Int("page", 0, "page index when -in is an archive")
	out := flag.String("out", "", "output directory for stage images and record.json")
	settingsPath := flag.String("settings", "", "settings JSON file (same format as the engine)")
	conf := flag.Float64("conf", 0, "detection confidence override (0 = keep the settings value)")
	workers := flag.Int("ocr-workers", 1, "concurrent OCR requests (1-4)")
	flag.Parse()
	if *in == "" || *out == "" {
		return fmt.Errorf("usage: probe-page -in page.png|book.cbz [-page 0] -out dir")
	}
	if *workers < 1 || *workers > 4 {
		return fmt.Errorf("ocr-workers must be in 1..4")
	}
	cfg := settings.Default()
	if *settingsPath != "" {
		loaded, err := settings.Load(*settingsPath)
		if err != nil {
			return err
		}
		cfg = loaded
	}
	if *conf > 0 {
		cfg.Recognition.Confidence = *conf
	}
	if math.IsNaN(cfg.Recognition.Confidence) || cfg.Recognition.Confidence <= 0 || cfg.Recognition.Confidence > 1 {
		return fmt.Errorf("confidence must be in (0,1]")
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(*out, 0755); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(*out, "04-ocr"), 0755); err != nil {
		return err
	}

	p := struct {
		llamaExe, ocrModel, ocrMMProj, mtModel, detModel, lapModel, ortDir, cudaDir string
	}{
		llamaExe:  envOr("MANGATRANS_LLAMA", `C:\Users\Public\Temp\opencode\llamacpp\llama\llama-server.exe`),
		ocrModel:  envOr("MANGATRANS_OCR_MODEL", `E:\work\transl\models\PaddleOCR-VL-manga-Q4_K_M.gguf`),
		ocrMMProj: envOr("MANGATRANS_MMPROJ", `E:\work\transl\models\mmproj-Q8_0.gguf`),
		mtModel:   mtModelPath(cfg),
		detModel:  envOr("MANGATRANS_DET_MODEL", `E:\work\transl\models\manga_det_v0.2.onnx`),
		lapModel:  envOr("MANGATRANS_LAMA_MODEL", `E:\work\transl\.cache\onnx-probe\lama-manga.onnx`),
		ortDir:    envOr("MANGATRANS_ORT_DIR", `E:\work\transl\runtime\ort-cuda13`),
		cudaDir:   envOr("MANGATRANS_CUDA_DIR", `E:\Python312\Lib\site-packages\torch\lib`),
	}
	for name, path := range map[string]string{
		"llama": p.llamaExe, "ocr model": p.ocrModel, "mmproj": p.ocrMMProj,
		"translation model": p.mtModel, "detector": p.detModel, "inpaint model": p.lapModel,
	} {
		if st, err := os.Stat(path); err != nil || st.IsDir() {
			return fmt.Errorf("%s not found: %s", name, path)
		}
	}

	// Load the single page: a plain image or one entry of an archive.
	var sourceBytes []byte
	sourceName := *in
	if isArchive(*in) {
		pages, err := cbz.ReadCBZ(*in)
		if err != nil {
			return fmt.Errorf("read input: %w", err)
		}
		if *page < 0 || *page >= len(pages) {
			return fmt.Errorf("page %d out of range, archive has %d pages", *page, len(pages))
		}
		sourceBytes, sourceName = pages[*page].Bytes, pages[*page].Name
	} else {
		data, err := os.ReadFile(*in)
		if err != nil {
			return err
		}
		sourceBytes = data
	}
	rec := record{Source: sourceName, Page: *page, TimingMS: map[string]int64{}}
	fail := func(format string, args ...any) error {
		rec.Note = fmt.Sprintf(format, args...)
		return saveRecord(*out, rec)
	}

	var img image.Image
	started := time.Now()
	img, err := pageimage.Decode(sourceBytes)
	rec.TimingMS["decode"] = time.Since(started).Milliseconds()
	if err != nil {
		rec.Width, rec.Height = 0, 0
		return fail("decode failed: %v", err)
	}
	rec.Width, rec.Height = img.Bounds().Dx(), img.Bounds().Dy()
	savePNG(filepath.Join(*out, "01-source.png"), img)
	log.Printf("page %s: %dx%d", sourceName, rec.Width, rec.Height)

	rt, err := infer.Env(p.ortDir, p.cudaDir)
	if err != nil {
		return fmt.Errorf("init onnxruntime: %w", err)
	}
	defer rt.Destroy()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()

	log.Printf("starting OCR server")
	ocrSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.ocrModel, MMProj: p.ocrMMProj,
		LogPath: filepath.Join(*out, "ocr.log"),
		Args:    []string{"-ngl", "99", "-c", "4096", "--temp", "0", "--jinja"},
	})
	if err != nil {
		return fmt.Errorf("start OCR server: %w", err)
	}
	defer ocrSrv.Kill()

	// Stage 1: text detection, identical to the engine (model load included in
	// the timing because reloads dominate single-page analyses).
	var res *dbtext.Result
	stage := func(name string, fn func()) {
		started := time.Now()
		fn()
		rec.TimingMS[name] = time.Since(started).Milliseconds()
		log.Printf("stage %-9s %s", name, time.Since(started).Round(time.Millisecond))
	}
	stage("detect", func() {
		det, loadErr := dbtext.Load(p.detModel, true, 3<<30)
		if loadErr != nil {
			rec.Note = "load detector: " + loadErr.Error()
			return
		}
		defer det.Close()
		params := dbtext.DefaultParams()
		params.BoxThresh = float32(cfg.Recognition.Confidence)
		if cfg.Recognition.TileSize > 0 {
			res, err = det.RunTiled(img, params, cfg.Recognition.TileSize)
		} else {
			res, err = det.Run(img, params)
		}
		if err != nil {
			rec.Note = "detect failed: " + err.Error()
		}
	})
	if res == nil {
		return saveRecord(*out, rec)
	}
	rec.DetBoxes = len(res.Boxes)
	boxes := make([]image.Rectangle, len(res.Boxes))
	for j, box := range res.Boxes {
		boxes[j] = image.Rect(int(box.X), int(box.Y), int(box.X+box.W), int(box.Y+box.H))
	}
	savePNG(filepath.Join(*out, "02-detect.png"), overlayMulti(img, boxes,
		pinkRects(len(boxes), color.RGBA{R: 255, G: 0, B: 230, A: 255})))

	// Stage 2: column merge into dialogue regions, then OCR each region crop.
	parents := bubble.MergeDialogueColumns(boxes)
	regions := make([]bubble.TextRegion, len(parents))
	for j, box := range parents {
		regions[j] = bubble.TextRegion{Box: box}
	}
	rec.Regions = len(regions)
	log.Printf("%d detector boxes -> %d dialogue regions", len(boxes), len(regions))

	type ocrOutcome struct {
		text    string
		err     error
		skipped bool
	}
	outcomes := make([]ocrOutcome, len(regions))
	stage("ocr", func() {
		jobs := make(chan int, len(regions))
		for j := range regions {
			jobs <- j
		}
		close(jobs)
		done := make(chan struct{})
		for range min(*workers, len(regions)) {
			go func() {
				defer func() { done <- struct{}{} }()
				for j := range jobs {
					crop := regions[j].Crop(img, 6)
					if crop == nil {
						outcomes[j].skipped = true
						continue
					}
					savePNG(filepath.Join(*out, "04-ocr", fmt.Sprintf("%02d.png", j)), crop)
					outcomes[j].text, outcomes[j].err = ocrBox(ctx, ocrSrv, crop)
				}
			}()
		}
		for range min(*workers, len(regions)) {
			<-done
		}
	})

	// Accept/reject exactly like the engine: clean, drop empty and noise-only
	// text. Blocks and acceptedIdx keep the block -> region mapping for later
	// stages and the record.
	var blocks []render.Block
	var acceptedIdx []int
	for j, got := range outcomes {
		box := regions[j].Box
		entry := regionRecord{
			Index:    j,
			Box:      rectSlice(box),
			Vertical: render.Vertical.IsVertical(box.Dy() > box.Dx()),
		}
		switch {
		case got.skipped:
			entry.Status = "crop_skipped"
		case got.err != nil:
			entry.Status = "ocr_failed"
			entry.Note = got.err.Error()
		default:
			text := cleanOCR(got.text)
			entry.OCR = text
			switch {
			case text == "":
				entry.Status = "empty"
			case isNoise(text):
				entry.Status = "noise"
			default:
				entry.Status = "accepted"
				entry.SizeHint = estimateFontSize(res.Mask, box, len([]rune(text)))
				blocks = append(blocks, render.Block{
					Box:      box,
					Vertical: entry.Vertical,
					Anchor:   box,
					SizeHint: entry.SizeHint,
				})
				acceptedIdx = append(acceptedIdx, j)
			}
		}
		rec.Blocks = append(rec.Blocks, entry)
		if entry.Status == "accepted" {
			rec.Accepted++
		} else {
			rec.Rejected++
		}
	}
	savePNG(filepath.Join(*out, "03-regions.png"), overlayVerdicts(img, rec.Blocks))
	ocrSrv.Kill()
	log.Printf("%d dialogue regions, %d accepted", len(regions), rec.Accepted)
	if rec.Accepted == 0 {
		return fail("no region produced usable text")
	}

	// Stage 3: translation in batches of 12, exactly like the engine.
	log.Printf("starting translation server")
	var mtSrv *llm.Server
	stage("translate_load", func() {
		mtSrv, err = llm.Start(ctx, llm.Config{
			Exe: p.llamaExe, Model: p.mtModel,
			LogPath: filepath.Join(*out, "mt.log"),
			Args:    []string{"-ngl", "99", "-c", "8192", "--jinja"},
		})
	})
	if err != nil {
		return fmt.Errorf("start translation server: %w", err)
	}
	tr := translate.New(mtSrv, nil)
	tr.BatchAttempts = cfg.Translation.BatchAttempts
	stage("translate", func() {
		const perBatch = 12
		for start := 0; start < len(blocks); start += perBatch {
			end := min(start+perBatch, len(blocks))
			batch := make([]translate.Block, 0, end-start)
			for k := start; k < end; k++ {
				batch = append(batch, translate.Block{
					Key:  translate.TargetKey{Page: 0, Region: k},
					Text: rec.Blocks[acceptedIdx[k]].OCR,
				})
			}
			got, batchErr := tr.TranslateBatch(ctx, start/perBatch, batch)
			if batchErr != nil {
				log.Printf("translate batch %d: %v", start/perBatch, batchErr)
			}
			for _, b := range batch {
				if text, ok := got[b.Key]; ok && text != "" {
					blocks[b.Key.Region].Text = text
					rec.Blocks[acceptedIdx[b.Key.Region]].ZH = text
				}
			}
		}
	})
	mtSrv.Kill()
	for _, b := range blocks {
		if b.Text != "" {
			rec.Translated++
		}
	}
	if rec.Translated < rec.Accepted {
		log.Printf("warning: %d of %d blocks kept the original text", rec.Accepted-rec.Translated, rec.Accepted)
	}

	// Stage 4: erase exactly the translated boxes, fit layout areas on the
	// clean page, colour each block, render.
	var clean image.Image
	stage("inpaint", func() {
		lap, loadErr := inpaint.Load(p.lapModel, true, 3<<30)
		if loadErr != nil {
			rec.Note = "load inpaint model: " + loadErr.Error()
			return
		}
		defer lap.Close()
		lap.ContextPadding = cfg.Repair.ContextPadding
		lap.TileSize = cfg.Repair.TileSize
		radius := cfg.Repair.MaskDilation
		lap.MaskRadius = &radius
		active := translatedBlocks(blocks)
		if len(active) == 0 {
			rec.Note = "nothing translated; nothing erased"
			return
		}
		if len(active) < len(blocks) {
			log.Printf("warning: erasing %d of %d blocks (untranslated blocks keep their text)", len(active), len(blocks))
		}
		mask := restrictMask(res.Mask, active)
		lap.MaskLimit = maskOwnership(res.Mask.Bounds(), active)
		savePNG(filepath.Join(*out, "05-mask.png"), mask)
		var inpaintErr error
		clean, inpaintErr = lap.Inpaint(ctx, img, mask)
		if inpaintErr != nil {
			rec.Note = "inpaint failed: " + inpaintErr.Error()
			return
		}
		savePNG(filepath.Join(*out, "06-clean.png"), clean)
	})
	if clean == nil {
		return saveRecord(*out, rec)
	}

	stage("layout", func() {
		// Original glyph strokes can split a bubble's white interior, so the
		// engine fits areas against the erased page. Mirror that here.
		finder := bubble.Find(clean, 210)
		fitBoxes := make([]image.Rectangle, len(blocks))
		for k := range blocks {
			fitBoxes[k] = blocks[k].Box
		}
		areas := finder.FitAreas(fitBoxes)
		for k := range blocks {
			if blocks[k].Area == nil {
				blocks[k].Area = areas[k]
			}
			sample := blocks[k].Box
			if blocks[k].Area != nil {
				sample = blocks[k].Area.Bounds()
			}
			blocks[k].TextColor = textColor(clean, sample)
			if cfg.Layout.Color != "auto" {
				var r, g, b uint8
				fmt.Sscanf(cfg.Layout.Color, "#%02x%02x%02x", &r, &g, &b)
				blocks[k].TextColor = color.RGBA{R: r, G: g, B: b, A: 255}
			}
			entry := &rec.Blocks[acceptedIdx[k]]
			if blocks[k].Area != nil {
				entry.Area = rectSlice(blocks[k].Area.Bounds())
			}
			entry.Color = fmt.Sprintf("#%02x%02x%02x", blocks[k].TextColor.R, blocks[k].TextColor.G, blocks[k].TextColor.B)
		}
		savePNG(filepath.Join(*out, "07-layout.png"), overlayAreas(clean, blocks))
	})

	active := translatedBlocks(blocks)
	if len(active) > 0 {
		renderOpt := render.Options{
			FontPath:  fontPath(cfg),
			MinSize:   1.2,
			MaxSize:   18.0,
			Padding:   3,
			TextColor: color.RGBA{A: 255},
		}
		renderOpt.FontScale = cfg.Layout.FontScale
		renderOpt.Spacing = cfg.Layout.Spacing
		renderOpt.AutoStroke = cfg.Layout.StrokeWidth < 0
		if cfg.Layout.MinFontSize > 0 {
			renderOpt.MinSize = cfg.Layout.MinFontSize
			renderOpt.MaxSize = max(18, cfg.Layout.MinFontSize/cfg.Layout.FontScale)
		}
		if cfg.Layout.StrokeWidth >= 0 {
			renderOpt.StrokeWidth = cfg.Layout.StrokeWidth
		}
		stage("render", func() {
			final, renderErr := render.Render(clean, active, renderOpt)
			if renderErr != nil {
				rec.Note = "render failed: " + renderErr.Error()
				return
			}
			savePNG(filepath.Join(*out, "08-final.png"), final)
		})
	}
	return saveRecord(*out, rec)
}

func isArchive(path string) bool {
	return strings.EqualFold(filepath.Ext(path), ".zip") || strings.EqualFold(filepath.Ext(path), ".cbz")
}

// mtModelPath and fontPath mirror the engine's fallback order: a built-in
// default, overridden when the settings file carries a non-empty value.
func mtModelPath(cfg settings.Config) string {
	if cfg.Translation.ModelPath != "" {
		return cfg.Translation.ModelPath
	}
	return envOr("MANGATRANS_MT_MODEL", `E:\work\transl\models\Hy-MT2-1.8B-UD-Q4_K_XL.gguf`)
}

func fontPath(cfg settings.Config) string {
	if cfg.Layout.FontPath != "" {
		return cfg.Layout.FontPath
	}
	return `C:\Windows\Fonts\msyhbd.ttc`
}

// --- helpers copied from cmd/mangatrans; keep in sync ---

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func cleanOCR(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "`")
	for _, p := range []string{"OCR:", "OCR：", "文字：", "文本：", "识别结果："} {
		s = strings.TrimSpace(strings.TrimPrefix(s, p))
	}
	s = strings.ReplaceAll(s, "\r", "")
	return strings.TrimSpace(s)
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
			src := full.PixOffset(r.Min.X, y)
			dst := out.PixOffset(r.Min.X, y)
			copy(out.Pix[dst:dst+r.Dx()], full.Pix[src:src+r.Dx()])
		}
	}
	return out
}

func maskOwnership(bounds image.Rectangle, blocks []render.Block) *image.Gray {
	m := image.NewGray(bounds)
	for _, b := range blocks {
		r := b.Box.Inset(-8).Intersect(bounds)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if b.SourceArea == nil || b.SourceArea.AlphaAt(x, y).A == 255 {
					m.SetGray(x, y, color.Gray{Y: 255})
				}
			}
		}
	}
	return m
}

// textColor picks black or white from the most frequent luminance of the
// sample box: the dominant luminance is the background, so the text takes the
// opposite colour.
func textColor(img image.Image, box image.Rectangle) color.RGBA {
	if img == nil {
		return color.RGBA{A: 255}
	}
	r := box.Intersect(img.Bounds())
	if r.Empty() {
		return color.RGBA{A: 255}
	}
	var hist [256]int
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			c := color.RGBAModel.Convert(img.At(x, y)).(color.RGBA)
			lum := (299*uint32(c.R) + 587*uint32(c.G) + 114*uint32(c.B)) / 1000
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

func translatedBlocks(blocks []render.Block) []render.Block {
	out := make([]render.Block, 0, len(blocks))
	for _, b := range blocks {
		if b.Text != "" {
			out = append(out, b)
		}
	}
	return out
}

// --- image output ---

func savePNG(path string, img image.Image) {
	if img == nil {
		return
	}
	f, err := os.Create(path)
	if err != nil {
		log.Printf("write %s: %v", path, err)
		return
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		log.Printf("encode %s: %v", path, err)
	}
}

func saveRecord(dir string, rec record) error {
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "record.json"), data, 0644)
}

func rectSlice(r image.Rectangle) []int {
	return []int{r.Min.X, r.Min.Y, r.Max.X, r.Max.Y}
}

func pinkRects(n int, c color.RGBA) []color.RGBA {
	out := make([]color.RGBA, n)
	for i := range out {
		out[i] = c
	}
	return out
}

// overlayMulti draws one-pixel rectangles over a copy of src. Thin borders stay
// readable at full-page zoom without hiding the text underneath.
func overlayMulti(src image.Image, rects []image.Rectangle, cols []color.RGBA) image.Image {
	b := src.Bounds()
	out := image.NewRGBA(b)
	draw.Draw(out, b, src, b.Min, draw.Src)
	for i, r := range rects {
		c := cols[i]
		r = r.Intersect(b)
		if r.Empty() {
			continue
		}
		for x := r.Min.X; x < r.Max.X; x++ {
			out.Set(x, r.Min.Y, c)
			out.Set(x, r.Max.Y-1, c)
		}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			out.Set(r.Min.X, y, c)
			out.Set(r.Max.X-1, y, c)
		}
	}
	return out
}

// overlayVerdicts colours merged regions by OCR outcome: green accepted,
// orange noise/empty/skipped, red failed.
func overlayVerdicts(src image.Image, entries []regionRecord) image.Image {
	green := color.RGBA{R: 0, G: 200, B: 0, A: 255}
	orange := color.RGBA{R: 255, G: 140, B: 0, A: 255}
	red := color.RGBA{R: 230, G: 0, B: 0, A: 255}
	boxes := make([]image.Rectangle, 0, len(entries))
	cols := make([]color.RGBA, 0, len(entries))
	for _, e := range entries {
		boxes = append(boxes, image.Rect(e.Box[0], e.Box[1], e.Box[2], e.Box[3]))
		switch e.Status {
		case "accepted":
			cols = append(cols, green)
		case "noise", "empty", "crop_skipped":
			cols = append(cols, orange)
		default:
			cols = append(cols, red)
		}
	}
	return overlayMulti(src, boxes, cols)
}

// overlayAreas draws translated block boxes in yellow and their fitted layout
// area bounds in cyan on the clean page.
func overlayAreas(src image.Image, blocks []render.Block) image.Image {
	yellow := color.RGBA{R: 255, G: 220, B: 0, A: 255}
	cyan := color.RGBA{R: 0, G: 210, B: 255, A: 255}
	boxes := make([]image.Rectangle, 0, 2*len(blocks))
	cols := make([]color.RGBA, 0, 2*len(blocks))
	for _, b := range blocks {
		boxes = append(boxes, b.Box)
		cols = append(cols, yellow)
		if b.Area != nil {
			boxes = append(boxes, b.Area.Bounds())
			cols = append(cols, cyan)
		}
	}
	return overlayMulti(src, boxes, cols)
}
