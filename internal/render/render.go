package render

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"

	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers"
)

// pxToMM maps page pixels to canvas millimetres. 4 px per mm keeps the
// rasterised text layer at page resolution.
const pxToMM = 0.25

type Block struct {
	Box      image.Rectangle
	Text     string
	Vertical bool
}

type Options struct {
	FontPath    string
	FontIndex   int
	MinSize     float64
	MaxSize     float64
	Padding     float64
	TextColor   color.RGBA
	StrokeColor color.RGBA
	StrokeWidth float64
}

// Render draws translated text into each block's box on a copy of page.
//
// Each block is rendered into its own canvas sized exactly to the usable area,
// then composited at the box offset. Doing it this way avoids depending on
// DrawText's anchor semantics and on how canvas swaps axes for vertical text.
func Render(page image.Image, blocks []Block, opt Options) (image.Image, error) {
	if page == nil {
		return nil, fmt.Errorf("render page: nil image")
	}
	font, err := canvas.LoadFontCollection(opt.FontPath, opt.FontIndex, canvas.FontRegular)
	if err != nil {
		font, err = canvas.LoadFontFile(opt.FontPath, canvas.FontRegular)
	}
	if err != nil {
		return nil, fmt.Errorf("load font %q: %w", opt.FontPath, err)
	}

	bounds := page.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(out, out.Bounds(), page, bounds.Min, draw.Src)

	min, max := opt.MinSize, opt.MaxSize
	if min <= 0 {
		min = 1
	}
	if max < min {
		return nil, fmt.Errorf("invalid font size range: min %.2f max %.2f", min, max)
	}
	fill := canvas.RGBA(float64(opt.TextColor.R)/255, float64(opt.TextColor.G)/255,
		float64(opt.TextColor.B)/255, float64(opt.TextColor.A)/255)
	if opt.TextColor.A == 0 {
		fill = canvas.Black
	}

	for _, b := range blocks {
		if b.Text == "" {
			continue
		}
		usable := image.Rect(
			b.Box.Min.X+int(opt.Padding), b.Box.Min.Y+int(opt.Padding),
			b.Box.Max.X-int(opt.Padding), b.Box.Max.Y-int(opt.Padding),
		).Intersect(image.Rect(0, 0, w, h))
		if usable.Dx() < 8 || usable.Dy() < 8 {
			continue
		}

		aw := float64(usable.Dx()) * pxToMM
		ah := float64(usable.Dy()) * pxToMM

		layout := func(size float64) *canvas.Text {
			rt := canvas.NewRichText(font.Face(size, fill))
			if b.Vertical {
				rt.SetWritingMode(canvas.VerticalRL)
				rt.SetTextOrientation(canvas.Upright)
			}
			rt.WriteString(b.Text)
			return rt.ToText(aw, ah, canvas.Center, canvas.Center, nil)
		}

		// Largest size whose rendered extent fits. Overflow flags are the
		// reliable signal; Width/Height swap meaning between writing modes.
		lo, hi := min, max
		for n := 0; n < 20; n++ {
			mid := (lo + hi) / 2
			t := layout(mid)
			if !t.OverflowsX && !t.OverflowsY {
				lo = mid
			} else {
				hi = mid
			}
		}
		text := layout(lo)
		if text.Empty() {
			continue
		}

		// Rasterise this block alone into a canvas the size of the usable area,
		// then composite. Anchor the text box at the canvas top-left: the y
		// argument is the box bottom measured from the canvas bottom, so the
		// top edge sits at ah. ToText centres the ink within that box.
		c := canvas.New(aw, ah)
		ctx := canvas.NewContext(c)
		ctx.SetFillColor(fill)
		ctx.DrawText(0, ah, text)

		var buf bytes.Buffer
		if err := renderers.PNG(canvas.DPMM(1/pxToMM))(&buf, c); err != nil {
			return nil, fmt.Errorf("rasterise block at %v: %w", b.Box, err)
		}
		layer, err := png.Decode(&buf)
		if err != nil {
			return nil, fmt.Errorf("decode block layer: %w", err)
		}
		draw.Draw(out, usable, layer, image.Point{}, draw.Over)
	}

	return out, nil
}
