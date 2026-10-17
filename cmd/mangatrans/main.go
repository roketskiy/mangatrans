package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

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
	flag.Parse()
	if *in == "" || *out == "" {
		log.Fatal("usage: mangatrans -in book.cbz -out book.zh.cbz [-work dir] [-limit N]")
	}

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

	pages, err := cbz.ReadCBZ(*in)
	if err != nil {
		log.Fatalf("read input: %v", err)
	}
	if *limit > 0 && len(pages) > *limit {
		pages = pages[:*limit]
	}
	log.Printf("pages: %d", len(pages))

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Minute)
	defer cancel()

	// Stage A: OCR. One llama-server instance, then released so the
	// translation model can take the same VRAM.
	log.Printf("starting OCR server")
	ocrSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.ocrModel, MMProj: p.ocrMMProj,
		LogPath: filepath.Join(os.TempDir(), "mangatrans-ocr.log"),
		Args:    []string{"-ngl", "99", "-c", "4096", "--temp", "0", "--jinja"},
	})
	if err != nil {
		log.Fatalf("start OCR server: %v", err)
	}

	rt, err := infer.Env(p.ortDir, p.cudaDir)
	if err != nil {
		log.Fatalf("init onnxruntime: %v", err)
	}
	defer rt.Destroy()

	det, err := detect.Load(p.detModel, true, 3<<30)
	if err != nil {
		log.Fatalf("load detector: %v", err)
	}
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
		img, err := decodeImage(pg.Bytes)
		if err != nil {
			log.Printf("page %d (%s): decode failed: %v", i, pg.Name, err)
			continue
		}
		res, err := det.Run(img, detParams)
		if err != nil {
			log.Printf("page %d: detect failed: %v", i, err)
			continue
		}

		pr := &pageResult{index: i, name: pg.Name, img: img, mask: res.Mask}
		for j, box := range res.Boxes {
			crop := cropRect(img, box.X, box.Y, box.W, box.H, 6)
			if crop == nil {
				continue
			}
			text, err := ocrBox(ctx, ocrSrv, crop)
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
	mtSrv, err := llm.Start(ctx, llm.Config{
		Exe: p.llamaExe, Model: p.mtModel,
		LogPath: filepath.Join(os.TempDir(), "mangatrans-mt.log"),
		Args:    []string{"-ngl", "99", "-c", "8192", "--jinja"},
	})
	if err != nil {
		log.Fatalf("start translation server: %v", err)
	}
	defer mtSrv.Kill()

	tr := translate.New(mtSrv)
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
			got, err := tr.TranslateBatch(ctx, batch)
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
	lap, err := inpaint.Load(p.lapModel, true, 3<<30)
	if err != nil {
		log.Fatalf("load inpaint model: %v", err)
	}
	defer lap.Close()

	renderOpt := render.Options{
		FontPath:  p.fontPath,
		MinSize:   1.2,
		MaxSize:   24.0,
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
		for k := range active {
			active[k].TextColor = textColor(pr.img, active[k].Box)
		}
		mask := restrictMask(pr.mask, active)
		if *work != "" {
			writePNG(filepath.Join(*work, fmt.Sprintf("mask%02d.png", i)), mask)
			dumps := make([]blockDump, 0, len(pr.blocks))
			for k, b := range pr.blocks {
				dumps = append(dumps, blockDump{Box: b.Box, Vertical: b.Vertical, OCR: pr.ocr[k], ZH: b.Text})
			}
			writeBlocks(filepath.Join(*work, fmt.Sprintf("blocks%02d.json", i)), dumps)
		}
		clean, err := lap.Inpaint(pr.img, mask)
		if err != nil {
			log.Printf("page %d: inpaint failed: %v", i, err)
			clean = pr.img
		}
		final, err := render.Render(clean, active, renderOpt)
		if err != nil {
			log.Printf("page %d: render failed: %v", i, err)
			final = clean
		}
		data, err := encodeJPEG(final, 92)
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

	if err := cbz.WriteCBZ(*out, outPages); err != nil {
		log.Fatalf("write output: %v", err)
	}
	log.Printf("wrote %s (%d pages)", *out, len(outPages))
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
