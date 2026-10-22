package main

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"strings"

	"mangatrans/internal/cbz"
	"mangatrans/internal/pageimage"
	"mangatrans/internal/render"
	"mangatrans/internal/translate"
	"mangatrans/settings"
)

const inferenceCacheVersion = 14

type storedPage struct {
	Index  int            `json:"index"`
	Mask   *image.Gray    `json:"mask"`
	Blocks []render.Block `json:"blocks"`
	OCR    []string       `json:"ocr"`
}

type inferenceRecord struct {
	Version int          `json:"version"`
	Key     string       `json:"key"`
	Pages   []storedPage `json:"pages"`
}

func validatePaths(p paths) error {
	files := []string{p.llamaExe, p.ocrModel, p.ocrMMProj, p.mtModel, p.detModel, p.lapModel, p.fontPath, filepath.Join(p.ortDir, "onnxruntime.dll")}
	for _, path := range files {
		st, err := os.Stat(path)
		if err != nil {
			return fmt.Errorf("required file %q: %w", path, err)
		}
		if !st.Mode().IsRegular() || st.Size() == 0 {
			return fmt.Errorf("required file is empty or not regular: %q", path)
		}
	}
	return render.ValidateFont(p.fontPath)
}

func inferenceKey(p paths, c settings.Config, conf float64, pages []cbz.Page, glossaryPath string) (string, error) {
	h := sha256.New()
	enc := json.NewEncoder(h)
	if err := enc.Encode(struct {
		Version     int
		Recognition settings.Recognition
		Translation settings.Translation
		Batch       int
		Confidence  float64
	}{inferenceCacheVersion, c.Recognition, c.Translation, c.Processing.BatchPages, conf}); err != nil {
		return "", err
	}
	for _, path := range []string{p.llamaExe, p.ocrModel, p.ocrMMProj, p.mtModel, p.detModel, p.bubbleSegModel, glossaryPath} {
		if path == "" {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			if path == p.bubbleSegModel {
				enc.Encode("unavailable optional segmentation model: " + path)
				continue
			}
			return "", err
		}
		modelHash := sha256.New()
		_, err = io.Copy(modelHash, f)
		f.Close()
		if err != nil {
			return "", err
		}
		enc.Encode(hex.EncodeToString(modelHash.Sum(nil)))
	}
	for _, pg := range pages {
		sum := sha256.Sum256(pg.Bytes)
		enc.Encode(pg.Name)
		enc.Encode(hex.EncodeToString(sum[:]))
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func recordFor(key string, pages []*pageResult) inferenceRecord {
	r := inferenceRecord{Version: inferenceCacheVersion, Key: key}
	for _, p := range pages {
		r.Pages = append(r.Pages, storedPage{p.index, p.mask, p.blocks, p.ocr})
	}
	return r
}

func loadInference(path, key string, pageCount int) ([]*pageResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer z.Close()
	var r inferenceRecord
	decoder := json.NewDecoder(io.LimitReader(z, 512<<20))
	if err = decoder.Decode(&r); err != nil {
		return nil, err
	}
	var extra any
	if err = decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("invalid cache trailer: %v", err)
	}
	if r.Version != inferenceCacheVersion || r.Key != key || len(r.Pages) != pageCount {
		return nil, fmt.Errorf("stale or incomplete cache")
	}
	var pages []*pageResult
	areas := map[string]*image.Alpha{}
	seen := map[int]bool{}
	for _, p := range r.Pages {
		if p.Index < 0 || p.Index >= pageCount || seen[p.Index] || len(p.Blocks) != len(p.OCR) || !validGray(p.Mask) {
			return nil, fmt.Errorf("invalid cached page")
		}
		seen[p.Index] = true
		for k := range p.Blocks {
			b := &p.Blocks[k]
			if translate.Untranslated(p.OCR[k], b.Text) || b.Box.Empty() || !b.Box.Overlaps(p.Mask.Bounds()) {
				return nil, fmt.Errorf("invalid cached text or box")
			}
			for _, a := range []*image.Alpha{b.Area, b.SourceArea} {
				if a != nil && !validAlpha(a) {
					return nil, fmt.Errorf("invalid cached geometry")
				}
			}
			if b.Area != nil {
				data, _ := json.Marshal(b.Area)
				sum := sha256.Sum256(data)
				id := hex.EncodeToString(sum[:])
				if old := areas[id]; old != nil {
					b.Area = old
				} else {
					areas[id] = b.Area
				}
			}
		}
		pages = append(pages, &pageResult{index: p.Index, mask: p.Mask, blocks: p.Blocks, ocr: p.OCR})
	}
	return pages, nil
}

func validGray(m *image.Gray) bool   { return m != nil && validPixels(m.Rect, m.Stride, len(m.Pix)) }
func validAlpha(m *image.Alpha) bool { return validPixels(m.Rect, m.Stride, len(m.Pix)) }
func validPixels(r image.Rectangle, stride, n int) bool {
	return !r.Empty() && r.Dx() <= 65536 && r.Dy() <= 65536 && stride >= r.Dx() && int64(stride)*int64(r.Dy()-1)+int64(r.Dx()) <= int64(n)
}

func atomicJSON(path string, value any, compressed bool) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".mangatrans-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	var w io.Writer = f
	var z *gzip.Writer
	if compressed {
		z = gzip.NewWriter(f)
		w = z
	}
	err = json.NewEncoder(w).Encode(value)
	if z != nil {
		if e := z.Close(); err == nil {
			err = e
		}
	}
	if err == nil {
		err = f.Sync()
	}
	if e := f.Close(); err == nil {
		err = e
	}
	if err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func availableOutput(path, policy string) (string, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return path, nil
	} else if err != nil {
		return "", err
	}
	if policy != "rename" {
		return "", fmt.Errorf("output exists; confirm replacement in the desktop app or choose another path: %s", path)
	}
	ext := filepath.Ext(path)
	stem := strings.TrimSuffix(path, ext)
	for n := 2; n < 100000; n++ {
		candidate := fmt.Sprintf("%s (%d)%s", stem, n, ext)
		if _, err := os.Stat(candidate); os.IsNotExist(err) {
			return candidate, nil
		} else if err != nil {
			return "", err
		}
	}
	return "", fmt.Errorf("no available output filename")
}

func outputOptions(c settings.Config) pageimage.Options {
	quality := c.Output.JPEGQuality
	if c.Output.Format == "webp" {
		quality = c.Output.WebPQuality
	}
	return pageimage.Options{Format: c.Output.Format, Quality: quality, Lossless: c.Output.WebPLossless, MaxLongEdge: c.Output.MaxLongEdge}
}

func assignOutputNames(pages []cbz.Page) {
	// A uniform numeric prefix fixes natural ordering even when mixed original
	// and converted extensions would otherwise sort identical stems differently.
	for i := range pages {
		name := filepath.Base(strings.ReplaceAll(pages[i].Name, "\\", "/"))
		pages[i].Name = fmt.Sprintf("%08d_%s", i+1, name)
	}
}

func pageNames(pages []cbz.Page) []string {
	names := make([]string, len(pages))
	for i, p := range pages {
		names[i] = p.Name
	}
	return names
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
