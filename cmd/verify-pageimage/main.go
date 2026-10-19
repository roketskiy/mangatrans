package main

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/gen2brain/webp"
	"mangatrans/internal/pageimage"
)

type result struct {
	Archive     string `json:"archive"`
	Member      string `json:"member"`
	Format      string `json:"format"`
	Width       int    `json:"width"`
	Height      int    `json:"height"`
	Bytes       int    `json:"bytes"`
	ExactPixels bool   `json:"exact_pixels"`
}

type report struct {
	Passed      bool     `json:"passed"`
	GoVersion   string   `json:"go_version"`
	Platform    string   `json:"platform"`
	WebPBackend string   `json:"webp_backend"`
	Checks      []string `json:"checks"`
	Outputs     []result `json:"outputs"`
	Error       string   `json:"error,omitempty"`
}

func main() {
	dir := flag.String("out", ".cache/settings", "artifact directory")
	flag.Parse()
	// Bound even direct invocations, independent of the caller's process timeout.
	watchdog := time.AfterFunc(2*time.Minute, func() {
		fmt.Fprintln(os.Stderr, "codec verification exceeded two minutes")
		os.Exit(2)
	})
	defer watchdog.Stop()
	r := report{GoVersion: runtime.Version(), Platform: runtime.GOOS + "/" + runtime.GOARCH, WebPBackend: "dynamic"}
	if err := webp.Dynamic(); err != nil {
		r.WebPBackend = "pure-go"
	}
	err := run(*dir, &r)
	r.Passed = err == nil
	if err != nil {
		r.Error = err.Error()
	}
	data, marshalErr := json.MarshalIndent(r, "", "  ")
	if marshalErr != nil {
		panic(marshalErr)
	}
	if writeErr := os.WriteFile(filepath.Join(*dir, "codec-report.json"), append(data, '\n'), 0644); writeErr != nil {
		fmt.Fprintln(os.Stderr, writeErr)
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("PASS: %d archive outputs, %d checks; %s\n", len(r.Outputs), len(r.Checks), filepath.Join(*dir, "codec-report.json"))
}

func run(dir string, r *report) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	invalid := []pageimage.Options{
		{Format: ""}, {Format: "avif"}, {Format: "png", Quality: -1},
		{Format: "webp", Quality: 101}, {Format: "jpeg", MaxLongEdge: -1},
	}
	for _, opts := range invalid {
		if opts.Validate() == nil {
			return fmt.Errorf("invalid options accepted: %+v", opts)
		}
		if _, _, err := pageimage.Encode(image.NewNRGBA(image.Rect(0, 0, 1, 1)), opts); err == nil {
			return fmt.Errorf("Encode accepted invalid options: %+v", opts)
		}
	}
	for _, img := range []image.Image{nil, image.NewNRGBA(image.Rectangle{})} {
		if _, _, err := pageimage.Encode(img, pageimage.Options{Format: "png"}); err == nil {
			return fmt.Errorf("empty source accepted")
		}
	}
	for _, data := range [][]byte{nil, []byte("garbage"), []byte("RIFF1234WEBPVP8L")} {
		if _, err := pageimage.Decode(data); err == nil {
			return fmt.Errorf("invalid bytes accepted")
		}
	}
	r.Checks = append(r.Checks, "invalid options, empty source, invalid bytes rejected")
	parent := image.NewNRGBA(image.Rect(-13, 7, 110, 96))
	for y := parent.Rect.Min.Y; y < parent.Rect.Max.Y; y++ {
		for x := parent.Rect.Min.X; x < parent.Rect.Max.X; x++ {
			a := uint8((x + 3*y) & 255)
			if (x+y)%7 == 0 {
				a = 0
			}
			parent.SetNRGBA(x, y, color.NRGBA{uint8(x * 17), uint8(y * 23), uint8(x + y*5), a})
		}
	}
	source := parent.SubImage(image.Rect(-5, 13, 92, 74))
	var input bytes.Buffer
	zw := zip.NewWriter(&input)
	for _, opts := range []pageimage.Options{
		{Format: "png"}, {Format: "jpeg"}, {Format: "webp", Quality: 90}, {Format: "webp", Quality: 90, Lossless: true},
	} {
		data, ext, err := pageimage.Encode(source, opts)
		if err != nil {
			return err
		}
		if opts.Format == "jpeg" || (opts.Format == "webp" && !opts.Lossless) {
			defaultOpts, explicitOpts := opts, opts
			defaultOpts.Quality = 0
			explicitOpts.Quality = 92
			if opts.Format == "webp" {
				explicitOpts.Quality = 90
			}
			defaults, _, err := pageimage.Encode(source, defaultOpts)
			if err != nil {
				return err
			}
			explicit, _, err := pageimage.Encode(source, explicitOpts)
			if err != nil || !bytes.Equal(defaults, explicit) {
				return fmt.Errorf("incorrect default quality for %s: %v", opts.Format, err)
			}
		}
		name := "page-" + opts.Format
		if opts.Lossless {
			name += "-lossless"
		}
		if err := addMember(zw, name+ext, data); err != nil {
			return err
		}
		decoded, err := pageimage.Decode(data)
		if err != nil {
			return err
		}
		if opts.Format == "png" || opts.Lossless {
			if err := equalPixels(source, decoded, false); err != nil {
				return fmt.Errorf("nonzero-bounds %s: %w", name, err)
			}
		}
		if opts.Format == "webp" {
			if err := equalPixels(source, decoded, true); err != nil {
				return fmt.Errorf("WebP alpha: %w", err)
			}
			if hasChunk(data, "VP8L") != opts.Lossless || hasChunk(data, "VP8 ") == opts.Lossless {
				return fmt.Errorf("incorrect WebP compression mode")
			}
		}
		if _, err := pageimage.Decode(data[:len(data)/2]); err == nil {
			return fmt.Errorf("truncated %s accepted", name)
		}
	}
	if err := zw.Close(); err != nil {
		return err
	}
	for _, ext := range []string{"cbz", "zip"} {
		if err := os.WriteFile(filepath.Join(dir, "codec-input."+ext), input.Bytes(), 0644); err != nil {
			return err
		}
	}
	r.Checks = append(r.Checks, "PNG and lossless WebP preserve all NRGBA bytes including zero-alpha RGB", "cropped negative-origin image with padded stride", "lossy WebP preserves alpha", "VP8 and VP8L chunks verified", "truncated JPEG, PNG, WebP rejected", "default JPEG quality 92 and WebP quality 90 match explicit settings")
	if err := checkResize(r); err != nil {
		return err
	}
	if err := checkJPEG(r); err != nil {
		return err
	}
	for _, format := range []string{"jpeg", "png", "webp-lossy", "webp-lossless"} {
		for _, edge := range []int{0, 48} {
			if err := convertArchive(dir, format, edge, r); err != nil {
				return err
			}
		}
	}
	r.Checks = append(r.Checks, "CBZ and ZIP input round trips, original-size and limited-long-edge outputs")
	return nil
}

func checkResize(r *report) error {
	for i, size := range []image.Point{{97, 61}, {61, 97}, {61, 61}, {1, 97}, {97, 1}} {
		src := image.NewNRGBA(image.Rect(5, -7, size.X+5, size.Y-7))
		for _, edge := range []int{0, 1000} {
			if pageimage.Resize(src, edge) != src {
				return fmt.Errorf("no-op resize allocated or changed image")
			}
		}
		out := pageimage.Resize(src, 31).Bounds()
		if max(out.Dx(), out.Dy()) != 31 || min(out.Dx(), out.Dy()) < 1 {
			return fmt.Errorf("invalid resize dimensions %v", out)
		}
		want := []image.Point{{31, 19}, {19, 31}, {31, 31}, {1, 31}, {31, 1}}[i]
		if out.Size() != want {
			return fmt.Errorf("aspect ratio changed: got %v want %v", out.Size(), want)
		}
	}
	r.Checks = append(r.Checks, "landscape, portrait, square and thin resize without enlargement")
	return nil
}

func checkJPEG(r *report) error {
	for _, c := range []color.NRGBA{{47, 81, 233, 0}, {255, 0, 0, 128}} {
		src := image.NewNRGBA(image.Rect(5, -3, 21, 13))
		for y := -3; y < 13; y++ {
			for x := 5; x < 21; x++ {
				src.SetNRGBA(x, y, c)
			}
		}
		data, ext, err := pageimage.Encode(src, pageimage.Options{Format: "JPG"})
		if err != nil || ext != ".jpg" {
			return fmt.Errorf("JPEG alias or encode failed: %s, %v", ext, err)
		}
		got, err := jpeg.Decode(bytes.NewReader(data))
		if err != nil {
			return err
		}
		want := color.NRGBA{255, 127, 127, 255}
		if c.A == 0 {
			want = color.NRGBA{255, 255, 255, 255}
		}
		pixel := color.NRGBAModel.Convert(got.At(8, 8)).(color.NRGBA)
		if abs(int(pixel.R)-int(want.R)) > 2 || abs(int(pixel.G)-int(want.G)) > 2 || abs(int(pixel.B)-int(want.B)) > 2 {
			return fmt.Errorf("JPEG did not composite over white: got %v want %v", pixel, want)
		}
	}
	r.Checks = append(r.Checks, "JPEG opaque white background and semitransparent composition, JPG alias")
	return nil
}

func convertArchive(dir, format string, edge int, r *report) error {
	inputName := "codec-input.cbz"
	if edge != 0 {
		inputName = "codec-input.zip"
	}
	zr, err := zip.OpenReader(filepath.Join(dir, inputName))
	if err != nil {
		return err
	}
	defer zr.Close()
	name := fmt.Sprintf("codec-%s-%d.cbz", format, edge)
	var output bytes.Buffer
	zw := zip.NewWriter(&output)
	expected := make(map[string]image.Image)
	opts := pageimage.Options{Format: format, Quality: 90, MaxLongEdge: edge}
	if strings.HasPrefix(format, "webp") {
		opts.Format = "webp"
		opts.Lossless = format == "webp-lossless"
	}
	if format == "jpeg" {
		opts.Quality = 0
	}
	for _, file := range zr.File {
		rc, err := file.Open()
		if err != nil {
			return err
		}
		data, err := io.ReadAll(rc)
		closeErr := rc.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
		src, err := pageimage.Decode(data)
		if err != nil {
			return err
		}
		encoded, ext, err := pageimage.Encode(src, opts)
		if err != nil {
			return err
		}
		member := strings.TrimSuffix(file.Name, filepath.Ext(file.Name)) + ext
		if err := addMember(zw, member, encoded); err != nil {
			return err
		}
		expected[member] = pageimage.Resize(src, edge)
	}
	if err := zw.Close(); err != nil {
		return err
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, output.Bytes(), 0644); err != nil {
		return err
	}
	actual, err := zip.OpenReader(path)
	if err != nil {
		return err
	}
	defer actual.Close()
	if len(actual.File) != len(expected) {
		return fmt.Errorf("output member count mismatch")
	}
	for _, file := range actual.File {
		rc, err := file.Open()
		if err != nil {
			return err
		}
		data, err := io.ReadAll(rc)
		closeErr := rc.Close()
		if err != nil {
			return err
		}
		if closeErr != nil {
			return closeErr
		}
		cfg, detected, err := image.DecodeConfig(bytes.NewReader(data))
		if err != nil || detected != opts.Format {
			return fmt.Errorf("actual output format %q != %q: %v", detected, opts.Format, err)
		}
		want := expected[file.Name]
		if cfg.Width != want.Bounds().Dx() || cfg.Height != want.Bounds().Dy() {
			return fmt.Errorf("output dimensions mismatch: %s", file.Name)
		}
		decoded, err := pageimage.Decode(data)
		if err != nil {
			return err
		}
		exact := opts.Format == "png" || opts.Lossless
		if exact {
			if err := equalPixels(want, decoded, false); err != nil {
				return fmt.Errorf("%s/%s: %w", name, file.Name, err)
			}
		}
		r.Outputs = append(r.Outputs, result{name, file.Name, detected, cfg.Width, cfg.Height, len(data), exact})
	}
	return nil
}

func addMember(w *zip.Writer, name string, data []byte) error {
	member, err := w.Create(name)
	if err != nil {
		return err
	}
	_, err = member.Write(data)
	return err
}

func equalPixels(want, got image.Image, alphaOnly bool) error {
	a, b := want.Bounds(), got.Bounds()
	if a.Size() != b.Size() {
		return fmt.Errorf("image dimensions differ")
	}
	for y := 0; y < a.Dy(); y++ {
		for x := 0; x < a.Dx(); x++ {
			wa := color.NRGBAModel.Convert(want.At(a.Min.X+x, a.Min.Y+y)).(color.NRGBA)
			gb := color.NRGBAModel.Convert(got.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			if (alphaOnly && wa.A != gb.A) || (!alphaOnly && wa != gb) {
				return fmt.Errorf("pixel (%d,%d) differs: %v != %v", x, y, wa, gb)
			}
		}
	}
	return nil
}

func hasChunk(data []byte, name string) bool {
	for off := 12; off+8 <= len(data); {
		n := int(binary.LittleEndian.Uint32(data[off+4 : off+8]))
		if string(data[off:off+4]) == name {
			return true
		}
		off += 8 + n + n%2
	}
	return false
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
