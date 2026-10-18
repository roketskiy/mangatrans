package inpaint

import (
	"context"
	"image"
	"image/color"
	"testing"
)

func TestInpaintLocalFastPathFillsDilatedTextFootprintWithoutSession(t *testing.T) {
	const w, h = 100, 100
	src := image.NewRGBA(image.Rect(0, 0, w, h))
	mask := image.NewGray(src.Bounds())
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
			if x < 4 || y < 4 || x >= w-4 || y >= h-4 {
				src.SetRGBA(x, y, color.RGBA{A: 255}) // distant black frame
			}
		}
	}
	// The visible black glyph is slightly wider than its detected mask, but
	// every extra black pixel remains inside the radius-6 dilated footprint.
	for y := 45; y < 55; y++ {
		for x := 43; x < 57; x++ {
			src.SetRGBA(x, y, color.RGBA{A: 255})
		}
	}
	for y := 47; y < 53; y++ {
		for x := 47; x < 53; x++ {
			mask.SetGray(x, y, color.Gray{Y: 255})
		}
	}
	got, err := (&Model{}).Inpaint(context.Background(), src, mask)
	if err != nil {
		t.Fatalf("local fast path unexpectedly needed model session: %v", err)
	}
	for y := 45; y < 55; y++ {
		for x := 43; x < 57; x++ {
			c := color.RGBAModel.Convert(got.At(x, y)).(color.RGBA)
			if c.R != 255 || c.G != 255 || c.B != 255 {
				t.Fatalf("dilated text pixel (%d,%d) remained dark: %#v", x, y, c)
			}
		}
	}
	for _, p := range []image.Point{{2, 2}, {97, 50}, {50, 2}} {
		c := color.RGBAModel.Convert(got.At(p.X, p.Y)).(color.RGBA)
		if c.R != 0 || c.G != 0 || c.B != 0 {
			t.Fatalf("outside-footprint frame pixel %v changed: %#v", p, c)
		}
	}
}

func TestSplitVerticalWindowsCoverRightBoundary(t *testing.T) {
	r := image.Rect(10, 5, 30, 95)
	parts := split(r)
	if len(parts) != 5 || parts[len(parts)-1].Max.X != r.Max.X {
		t.Fatalf("vertical split does not cover right edge: %#v", parts)
	}
	for _, p := range parts {
		if p.Min.X != r.Min.X || p.Max.X != r.Max.X || p.Empty() {
			t.Fatalf("invalid vertical split window %v", p)
		}
	}
}

func TestSolidBackgroundUsesUnmaskedPixels(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 32, 32))
	mask := make([]uint8, 32*32)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			im.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	for y := 10; y < 22; y++ {
		for x := 10; x < 22; x++ {
			mask[y*32+x] = 255
			im.SetRGBA(x, y, color.RGBA{A: 255})
		}
	}
	c, ok := solidBackground(im, mask, 32, im.Bounds())
	if !ok || c.R != 255 || c.G != 255 || c.B != 255 {
		t.Fatalf("white background around black masked center got (%v, %t)", c, ok)
	}
	fillMasked(im, mask, 32, im.Bounds(), c)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			got := im.RGBAAt(x, y)
			if mask[y*32+x] == 0 {
				if got != (color.RGBA{R: 255, G: 255, B: 255, A: 255}) {
					t.Fatalf("unmasked pixel (%d,%d) changed: %#v", x, y, got)
				}
			} else if got.R != 255 || got.G != 255 || got.B != 255 {
				t.Fatalf("masked pixel (%d,%d) was not filled white: %#v", x, y, got)
			}
		}
	}
}

func TestSolidBackgroundRejectsGrayTexture(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 20, 20))
	mask := make([]uint8, 400)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			v := uint8(160)
			if (x+y)%2 == 0 {
				v = 120
			}
			im.SetRGBA(x, y, color.RGBA{R: v, G: v, B: v, A: 255})
		}
	}
	if _, ok := solidBackground(im, mask, 20, im.Bounds()); ok {
		t.Fatal("textured gray background incorrectly accepted as solid")
	}
}

func TestLocalSolidBackgroundAcceptsWhiteRegionAwayFromDistantBorder(t *testing.T) {
	const w, h = 100, 100
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	mask := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{R: 255, G: 255, B: 255, A: 255}
			if x < 3 || y < 3 || x >= w-3 || y >= h-3 {
				c = color.RGBA{A: 255}
			}
			im.SetRGBA(x, y, c)
		}
	}
	component := makeComponent(t, mask, w, image.Rect(44, 44, 56, 56))
	// The direct local detector sees only a clean neighborhood; a page-wide
	// detector would include the distant dark frame and reject this fill.
	window := expand(component.bounds, 8, w, h)
	bg, ok := localSolidBackground(im, mask, w, window)
	if !ok {
		t.Fatalf("clean local white region rejected around %v", component.bounds)
	}
	if _, ok := solidBackground(im, mask, w, im.Bounds()); ok {
		t.Fatal("page-wide detector unexpectedly ignored distant dark frame")
	}
	componentMask := make([]uint8, len(mask))
	for _, i := range component.pixels {
		componentMask[i] = 255
	}
	fillComponent(im, componentMask, w, component, bg)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := im.RGBAAt(x, y)
			if mask[y*w+x] > 0 && (c.R != 255 || c.G != 255 || c.B != 255) {
				t.Fatalf("component mask pixel (%d,%d) not restored", x, y)
			}
			if (x < 3 || y < 3 || x >= w-3 || y >= h-3) && c.R != 0 {
				t.Fatalf("distant dark frame changed at (%d,%d)", x, y)
			}
		}
	}
}

func TestLocalSolidBackgroundRejectsGrayColorTextureAndInsufficientSamples(t *testing.T) {
	tests := []struct {
		name  string
		paint func(x, y int) color.RGBA
	}{
		{"gray", func(_, _ int) color.RGBA { return color.RGBA{R: 160, G: 160, B: 160, A: 255} }},
		{"color", func(_, _ int) color.RGBA { return color.RGBA{R: 240, G: 180, B: 180, A: 255} }},
		{"texture", func(x, y int) color.RGBA {
			v := uint8(255)
			if (x+y)%2 == 0 {
				v = 180
			}
			return color.RGBA{R: v, G: v, B: v, A: 255}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			im, mask, r := localFixture(40, tc.paint)
			if _, ok := localSolidBackground(im, mask, 40, r); ok {
				t.Fatalf("%s region incorrectly accepted", tc.name)
			}
		})
	}
	// Leave 63 unmasked pixels in an otherwise flat white window.
	im, mask, r := localFixture(12, func(_, _ int) color.RGBA {
		return color.RGBA{R: 255, G: 255, B: 255, A: 255}
	})
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			mask[y*12+x] = 255
		}
	}
	for i := 0; i < 63; i++ {
		mask[i] = 0
	}
	if _, ok := localSolidBackground(im, mask, 12, r); ok {
		t.Fatal("region with insufficient unmasked samples incorrectly accepted")
	}
}

func TestLocalComponentFillPreservesUnmaskedPixelsAndOtherComponents(t *testing.T) {
	const w, h = 40, 40
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	componentMask := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			im.SetRGBA(x, y, color.RGBA{R: 255, G: 255, B: 255, A: 255})
		}
	}
	for _, r := range []image.Rectangle{image.Rect(8, 8, 14, 14), image.Rect(25, 25, 29, 29)} {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				i := y*w + x
				componentMask[i] = 255
				im.SetRGBA(x, y, color.RGBA{A: 255})
			}
		}
	}
	regions := maskComponents(componentMask, w, h)
	if len(regions) != 2 {
		t.Fatalf("got %d components, want 2", len(regions))
	}
	before := make([]color.RGBA, w*h)
	for i := range before {
		before[i] = im.RGBAAt(i%w, i/w)
	}
	target := make(map[int]bool, len(regions[0].pixels))
	for _, i := range regions[0].pixels {
		target[i] = true
	}
	fill := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	fillComponent(im, componentMask, w, regions[0], fill)
	for i := range componentMask {
		if !target[i] && im.RGBAAt(i%w, i/w) != before[i] {
			t.Fatalf("non-target pixel %d changed", i)
		}
	}
	for _, i := range regions[1].pixels {
		if componentMask[i] == 0 {
			t.Fatalf("neighbor component pixel %d was cleared", i)
		}
	}
}

func localFixture(size int, paint func(x, y int) color.RGBA) (*image.RGBA, []uint8, image.Rectangle) {
	im := image.NewRGBA(image.Rect(0, 0, size, size))
	mask := make([]uint8, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			im.SetRGBA(x, y, paint(x, y))
		}
	}
	return im, mask, im.Bounds()
}

func makeComponent(t *testing.T, mask []uint8, w int, r image.Rectangle) maskComponent {
	t.Helper()
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			mask[y*w+x] = 255
		}
	}
	regions := maskComponents(mask, w, len(mask)/w)
	if len(regions) != 1 {
		t.Fatalf("got %d components, want 1", len(regions))
	}
	return regions[0]
}

func TestMergeOverlappingWindowsRespectsPatchLimit(t *testing.T) {
	in := []image.Rectangle{
		image.Rect(0, 0, 300, 300),
		image.Rect(250, 200, 450, 400), // overlaps; union is 450x400
		image.Rect(450, 0, 500, 300),   // adjacent, must remain separate
		image.Rect(450, 300, 600, 500), // overlaps but union would exceed 512 wide
	}
	out := mergeOverlapping(in)
	if len(out) != 3 {
		t.Fatalf("expected one legal overlap merge only; got %v", out)
	}
	for _, r := range out {
		if r.Dx() > side || r.Dy() > side {
			t.Fatalf("merged window exceeds model patch: %v", r)
		}
	}
	if out[0] != image.Rect(0, 0, 450, 400) {
		t.Fatalf("unexpected merged bound: %v", out[0])
	}
}
