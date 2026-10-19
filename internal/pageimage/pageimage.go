package pageimage

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"strings"

	"github.com/gen2brain/webp"
	xdraw "golang.org/x/image/draw"
	xwebp "golang.org/x/image/webp"
)

type Options struct {
	Format      string
	Quality     int // Zero selects JPEG 92 or WebP 90.
	Lossless    bool
	MaxLongEdge int // Zero preserves the original dimensions.
}

func (o Options) Validate() error {
	switch strings.ToLower(strings.TrimSpace(o.Format)) {
	case "jpg", "jpeg", "png", "webp":
	default:
		return fmt.Errorf("pageimage: unsupported output format %q", o.Format)
	}
	if o.Quality < 0 || o.Quality > 100 {
		return fmt.Errorf("pageimage: quality must be between 0 and 100")
	}
	if o.MaxLongEdge < 0 {
		return fmt.Errorf("pageimage: maximum long edge must not be negative")
	}
	return nil
}

func Decode(data []byte) (image.Image, error) {
	// The encoder's registered decoder uses YUV420 even for lossless WebP.
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		img, err := xwebp.Decode(bytes.NewReader(data))
		if err != nil {
			return nil, fmt.Errorf("pageimage: decode WebP: %w", err)
		}
		return img, nil
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("pageimage: decode: %w", err)
	}
	return img, nil
}

func Encode(img image.Image, opts Options) ([]byte, string, error) {
	if err := opts.Validate(); err != nil {
		return nil, "", err
	}
	if img == nil || img.Bounds().Empty() {
		return nil, "", fmt.Errorf("pageimage: source image is empty")
	}
	img = Resize(img, opts.MaxLongEdge)
	var buf bytes.Buffer
	var err error
	var ext string
	switch strings.ToLower(strings.TrimSpace(opts.Format)) {
	case "jpg", "jpeg":
		ext = ".jpg"
		quality := opts.Quality
		if quality == 0 {
			quality = 92
		}
		dst := image.NewRGBA(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
		draw.Draw(dst, dst.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
		draw.Draw(dst, dst.Bounds(), img, img.Bounds().Min, draw.Over)
		err = jpeg.Encode(&buf, dst, &jpeg.Options{Quality: quality})
	case "png":
		ext = ".png"
		err = png.Encode(&buf, normalized(img))
	case "webp":
		ext = ".webp"
		quality := opts.Quality
		if quality == 0 {
			quality = 90
		}
		err = webp.Encode(&buf, normalized(img), webp.Options{
			Quality: quality, Lossless: opts.Lossless, Method: webp.DefaultMethod, Exact: true,
		})
	}
	if err != nil {
		return nil, "", fmt.Errorf("pageimage: encode %s: %w", ext, err)
	}
	return buf.Bytes(), ext, nil
}

func normalized(img image.Image) *image.NRGBA {
	b := img.Bounds()
	dst := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	// A fixed straight-alpha conversion avoids codec-specific YCbCrA rounding.
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			dst.SetNRGBA(x, y, color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA))
		}
	}
	return dst
}

// Resize preserves aspect ratio without enlarging. Nonpositive limits are a no-op.
func Resize(img image.Image, maxLongEdge int) image.Image {
	if img == nil || maxLongEdge <= 0 {
		return img
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	long := max(w, h)
	if b.Empty() || long <= maxLongEdge {
		return img
	}
	scale := float64(maxLongEdge) / float64(long)
	w = max(1, int(math.Round(float64(w)*scale)))
	h = max(1, int(math.Round(float64(h)*scale)))
	dst := image.NewNRGBA(image.Rect(0, 0, w, h))
	xdraw.CatmullRom.Scale(dst, dst.Bounds(), img, b, draw.Src, nil)
	return dst
}
