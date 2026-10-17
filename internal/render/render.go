package render

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"runtime/debug"
	"strings"

	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers"
)

// wrapForMode inserts line breaks so canvas has explicit columns/lines to lay
// out. canvas does not break CJK text without spaces, so a long translation
// would otherwise run off one end of the box (seen on page 6 of the test book).
func wrapForMode(s string, vertical bool, size, aw, ah float64) string {
	// canvas font sizes are points, while the box is millimetres.
	const ptToMM = 0.3527778
	limit := aw
	if vertical {
		limit = ah
	}
	per := int(limit / (size * ptToMM * 1.1))
	if per < 1 {
		per = 1
	}
	runes := []rune(s)
	if len(runes) <= per {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + len(runes)/per)
	for len(runes) > 0 {
		n := per
		if n > len(runes) {
			n = len(runes)
		}
		// Kinsoku: never start a column with closing punctuation, and never
		// end one with an opening bracket.
		for n < len(runes) && isClosing(runes[n]) {
			n++
		}
		if n > 1 && n < len(runes) && isOpening(runes[n-1]) {
			n--
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(string(runes[:n]))
		runes = runes[n:]
	}
	return b.String()
}

func isClosing(r rune) bool {
	return strings.ContainsRune("、。，．,.!?！？：；）」』】〕》〉…ー～", r)
}

func isOpening(r rune) bool {
	return strings.ContainsRune("（「『【〔《〈", r)
}

// normalize flattens line breaks. The layout engine wraps by column, so a
// model that copies the source's line structure would create too many narrow
// columns and overlap them in a tight bubble.
func normalize(s string) string {
	s = strings.ReplaceAll(s, "\r", "")
	s = strings.ReplaceAll(s, "\n", "")
	s = strings.ReplaceAll(s, "\t", "")
	return strings.TrimSpace(s)
}

// pxToMM maps page pixels to canvas millimetres. 4 px per mm keeps the
// rasterised text layer at page resolution.
const pxToMM = 0.25

type Block struct {
	Box      image.Rectangle
	Text     string
	Vertical bool
	// Anchor, when set, is the speech-bubble rectangle used for placement.
	// Box still controls the font size, so a longer translation cannot grow
	// past the original text footprint and leave the bubble.
	Anchor image.Rectangle
	// TextColor is chosen per block from the original glyph colour. The zero
	// value falls back to Options.TextColor.
	TextColor color.RGBA
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
func Render(page image.Image, blocks []Block, opt Options) (out image.Image, err error) {
	// A bad glyph, font, or degenerate block must not terminate the book.
	// The caller reverts to the input page when this returns an error.
	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = fmt.Errorf("render panic: %v\n%s", r, debug.Stack())
		}
	}()
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
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), page, bounds.Min, draw.Src)

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
		b.Text = normalize(b.Text)
		if b.Text == "" {
			continue
		}
		blockFill := fill
		if b.TextColor.A != 0 {
			blockFill = canvas.RGBA(float64(b.TextColor.R)/255, float64(b.TextColor.G)/255,
				float64(b.TextColor.B)/255, float64(b.TextColor.A)/255)
		}
		// Detected boxes hug the original glyphs and can reach the bubble
		// outline. A proportional inset keeps longer translations away from
		// the border, where they would spill onto the artwork.
		pad := opt.Padding
		if m := 0.07 * math.Min(float64(b.Box.Dx()), float64(b.Box.Dy())); m > pad {
			pad = m
		}
		usable := image.Rect(
			b.Box.Min.X+int(pad), b.Box.Min.Y+int(pad),
			b.Box.Max.X-int(pad), b.Box.Max.Y-int(pad),
		).Intersect(image.Rect(0, 0, w, h))
		if usable.Dx() < 8 || usable.Dy() < 8 {
			continue
		}

		aw := float64(usable.Dx()) * pxToMM
		ah := float64(usable.Dy()) * pxToMM

		layout := func(size float64) *canvas.Text {
			rt := canvas.NewRichText(font.Face(size, blockFill))
			if b.Vertical {
				rt.SetWritingMode(canvas.VerticalRL)
				rt.SetTextOrientation(canvas.Upright)
			}
			rt.WriteString(wrapForMode(b.Text, b.Vertical, size, aw, ah))
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
		ctx.SetFillColor(blockFill)
		ctx.DrawText(0, ah, text)

		var buf bytes.Buffer
		if err := renderers.PNG(canvas.DPMM(1/pxToMM))(&buf, c); err != nil {
			return nil, fmt.Errorf("rasterise block at %v: %w", b.Box, err)
		}
		layer, err := png.Decode(&buf)
		if err != nil {
			return nil, fmt.Errorf("decode block layer: %w", err)
		}
		anchor := b.Anchor
		if anchor.Empty() {
			anchor = b.Box
		}
		dst := image.Rect(
			anchor.Min.X+(anchor.Dx()-usable.Dx())/2,
			anchor.Min.Y+(anchor.Dy()-usable.Dy())/2,
			anchor.Min.X+(anchor.Dx()-usable.Dx())/2+usable.Dx(),
			anchor.Min.Y+(anchor.Dy()-usable.Dy())/2+usable.Dy(),
		)
		draw.Draw(rgba, dst, layer, image.Point{}, draw.Over)
	}

	return rgba, nil
}
