package cbz

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Page is one image of the book, in reading order.
type Page struct {
	Name  string
	Bytes []byte
}

var imageExts = map[string]bool{
	".png": true, ".jpg": true, ".jpeg": true, ".webp": true, ".bmp": true, ".gif": true,
}

// ReadCBZ extracts the image entries of a cbz/zip archive in natural sort order.
func ReadCBZ(path string) ([]Page, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open archive: %w", err)
	}
	defer r.Close()

	var pages []Page
	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(f.Name))
		if !imageExts[ext] {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, fmt.Errorf("open %s: %w", f.Name, err)
		}
		data, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", f.Name, err)
		}
		pages = append(pages, Page{Name: f.Name, Bytes: data})
	}
	if len(pages) == 0 {
		return nil, fmt.Errorf("no images in %s", path)
	}

	sort.Slice(pages, func(i, j int) bool {
		return naturalLess(pages[i].Name, pages[j].Name)
	})
	return pages, nil
}

// WriteCBZ writes pages to a deflate-compressed zip.
func WriteCBZ(path string, pages []Page) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	zw := zip.NewWriter(f)
	for _, p := range pages {
		w, err := zw.Create(p.Name)
		if err != nil {
			return err
		}
		if _, err := w.Write(p.Bytes); err != nil {
			return err
		}
	}
	if err := zw.Close(); err != nil {
		return err
	}
	return f.Close()
}

// naturalLess compares names so that page2 sorts before page10.
func naturalLess(a, b string) bool {
	ai, bi := 0, 0
	for ai < len(a) && bi < len(b) {
		ca, cb := a[ai], b[bi]
		if isDigit(ca) && isDigit(cb) {
			// compare the full number runs
			as, ae := digitRun(a, ai)
			bs, be := digitRun(b, bi)
			an := strings.TrimLeft(a[as:ae], "0")
			bn := strings.TrimLeft(b[bs:be], "0")
			if len(an) != len(bn) {
				return len(an) < len(bn)
			}
			if an != bn {
				return an < bn
			}
			ai, bi = ae, be
			continue
		}
		if ca != cb {
			return ca < cb
		}
		ai++
		bi++
	}
	return len(a)-ai < len(b)-bi
}

func digitRun(s string, i int) (int, int) {
	start := i
	for i < len(s) && isDigit(s[i]) {
		i++
	}
	return start, i
}

func isDigit(c byte) bool { return '0' <= c && c <= '9' }
