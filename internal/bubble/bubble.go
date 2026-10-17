// Package bubble finds speech-bubble interiors as connected white regions. It
// lets the renderer lay text out in the bubble instead of the tight OCR box,
// which keeps longer translations inside the outline and allows a larger font.
package bubble

import (
	"image"
	"image/color"
)

// Finder holds the white-region labels of one page.
type Finder struct {
	W, H   int
	labels []int32
	parent []int32
	stats  map[int32]stat
}

type stat struct {
	rect image.Rectangle
	area int
}

// Find labels 4-connected white regions. Pixels at or above whiteMin on the
// 0..255 luminance scale are white. Ink is thickened by a few pixels first so
// one-pixel gaps in a bubble outline do not leak the interior into the page
// background.
func Find(img image.Image, whiteMin uint8) *Finder {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	f := &Finder{
		W:      w,
		H:      h,
		labels: make([]int32, w*h),
		parent: make([]int32, 0, w*h/4),
		stats:  make(map[int32]stat),
	}
	for i := range f.labels {
		f.labels[i] = -1
	}

	blocked := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !isWhite(img.At(b.Min.X+x, b.Min.Y+y), whiteMin) {
				blocked[y*w+x] = true
			}
		}
	}
	blocked = dilate(blocked, w, h, 2)

	// First pass: assign provisional labels and union with the left and top
	// neighbours. A union-find keeps this linear-ish on a 1 MP page.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if blocked[i] {
				continue
			}
			var left, up int32 = -1, -1
			if x > 0 {
				left = f.labels[i-1]
			}
			if y > 0 {
				up = f.labels[i-w]
			}
			switch {
			case left < 0 && up < 0:
				id := int32(len(f.parent))
				f.parent = append(f.parent, id)
				f.labels[i] = id
			case left >= 0 && up >= 0:
				f.labels[i] = f.union(left, up)
			case left >= 0:
				f.labels[i] = f.find(left)
			default:
				f.labels[i] = f.find(up)
			}
		}
	}

	// Second pass: resolve roots and accumulate bounding boxes.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if f.labels[i] < 0 {
				continue
			}
			r := f.find(f.labels[i])
			f.labels[i] = r
			s, ok := f.stats[r]
			if !ok {
				s.rect = image.Rect(x, y, x+1, y+1)
			} else {
				if x < s.rect.Min.X {
					s.rect.Min.X = x
				}
				if y < s.rect.Min.Y {
					s.rect.Min.Y = y
				}
				if x+1 > s.rect.Max.X {
					s.rect.Max.X = x + 1
				}
				if y+1 > s.rect.Max.Y {
					s.rect.Max.Y = y + 1
				}
			}
			s.area++
			f.stats[r] = s
		}
	}
	return f
}

func (f *Finder) find(i int32) int32 {
	for f.parent[i] != i {
		f.parent[i] = f.parent[f.parent[i]]
		i = f.parent[i]
	}
	return i
}

func (f *Finder) union(a, b int32) int32 {
	ra, rb := f.find(a), f.find(b)
	if ra == rb {
		return ra
	}
	f.parent[rb] = ra
	return ra
}

// Fit returns the bubble rectangle that contains textBox. It only accepts
// regions that look like a speech bubble: enclosed by ink, larger than the
// text, not the page background, and not an enormous white area of artwork.
func (f *Finder) Fit(textBox image.Rectangle) (image.Rectangle, bool) {
	if f == nil || textBox.Empty() {
		return image.Rectangle{}, false
	}
	textBox = textBox.Intersect(image.Rect(0, 0, f.W, f.H))
	if textBox.Empty() {
		return image.Rectangle{}, false
	}

	counts := make(map[int32]int)
	for y := textBox.Min.Y; y < textBox.Max.Y; y++ {
		for x := textBox.Min.X; x < textBox.Max.X; x++ {
			if id := f.labels[y*f.W+x]; id >= 0 {
				counts[id]++
			}
		}
	}
	var best int32 = -1
	bestN := 0
	for id, n := range counts {
		if n > bestN {
			best, bestN = id, n
		}
	}
	if best < 0 {
		return image.Rectangle{}, false
	}
	s := f.stats[best]

	// Reject the page background and very large artwork regions.
	if s.rect.Min.X <= 2 || s.rect.Min.Y <= 2 || s.rect.Max.X >= f.W-2 || s.rect.Max.Y >= f.H-2 {
		return image.Rectangle{}, false
	}
	if s.rect.Dx()*s.rect.Dy() > f.W*f.H/5 {
		return image.Rectangle{}, false
	}
	// The text must sit inside the region, and the region must be bigger.
	outer := s.rect.Inset(-2)
	if textBox.Min.X < outer.Min.X || textBox.Min.Y < outer.Min.Y ||
		textBox.Max.X > outer.Max.X || textBox.Max.Y > outer.Max.Y {
		return image.Rectangle{}, false
	}
	if s.area < textBox.Dx()*textBox.Dy() {
		return image.Rectangle{}, false
	}
	// A region far larger than the text is usually several bubbles merged by
	// a shared white gap; expanding there would centre the text in the gutter.
	tbArea := textBox.Dx() * textBox.Dy()
	if s.rect.Dx()*s.rect.Dy() > 8*tbArea {
		return image.Rectangle{}, false
	}
	if s.rect.Dx() > 5*textBox.Dx() || s.rect.Dy() > 5*textBox.Dy() {
		return image.Rectangle{}, false
	}
	// Extremely elongated regions are usually panel gutters, not bubbles.
	dx, dy := s.rect.Dx(), s.rect.Dy()
	if dx > 6*dy || dy > 6*dx {
		return image.Rectangle{}, false
	}
	return s.rect.Inset(3), true
}

// Debug returns the region label under a point, for probes.
func (f *Finder) Debug(x, y int) (image.Rectangle, int, bool) {
	if x < 0 || y < 0 || x >= f.W || y >= f.H {
		return image.Rectangle{}, 0, false
	}
	id := f.labels[y*f.W+x]
	if id < 0 {
		return image.Rectangle{}, 0, false
	}
	s := f.stats[id]
	return s.rect, s.area, true
}

func isWhite(c color.Color, min uint8) bool {
	r, g, b, _ := c.RGBA()
	// RGBA returns 16-bit values; approximate luminance in 0..255.
	lum := (299*(r>>8) + 587*(g>>8) + 114*(b>>8)) / 1000
	return lum >= uint32(min)
}

// dilate grows a boolean mask with a square structuring element, using one row
// pass and one column pass.
func dilate(src []bool, w, h, r int) []bool {
	tmp := make([]bool, len(src))
	out := make([]bool, len(src))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			lo, hi := x-r, x+r
			if lo < 0 {
				lo = 0
			}
			if hi >= w {
				hi = w - 1
			}
			for i := lo; i <= hi; i++ {
				if src[y*w+i] {
					tmp[y*w+x] = true
					break
				}
			}
		}
	}
	for y := 0; y < h; y++ {
		lo, hi := y-r, y+r
		if lo < 0 {
			lo = 0
		}
		if hi >= h {
			hi = h - 1
		}
		for x := 0; x < w; x++ {
			for i := lo; i <= hi; i++ {
				if tmp[i*w+x] {
					out[y*w+x] = true
					break
				}
			}
		}
	}
	return out
}
