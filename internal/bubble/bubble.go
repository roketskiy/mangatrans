// Package bubble finds speech-bubble interiors as connected white regions. It
// lets the renderer lay text out in the bubble instead of the tight OCR box,
// which keeps longer translations inside the outline and allows a larger font.
package bubble

import (
	"fmt"
	"image"
	"image/color"
	"sort"
)

// Finder holds the white-region labels of one page.
type Finder struct {
	W, H   int
	origin image.Point
	labels []int32
	parent []int32
	stats  map[int32]stat
	Reason string
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
		origin: b.Min,
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

// FitAreas returns one safe placement mask per text box. Masks for boxes in
// the same connected component are partitioned by distance to their original
// boxes. A nil entry means the component was not trustworthy enough to expand.
// Bounds use the source image's coordinate system.
func (f *Finder) FitAreas(boxes []image.Rectangle) []*image.Alpha {
	areas := make([]*image.Alpha, len(boxes))
	if f == nil {
		return areas
	}
	groups := make(map[int32][]int)
	for i, box := range boxes {
		id, ok := f.componentFor(box)
		if ok {
			groups[id] = append(groups[id], i)
		}
	}
	for id, indices := range groups {
		componentBoxes := make([]image.Rectangle, 0, len(indices))
		for _, i := range indices {
			componentBoxes = append(componentBoxes, boxes[i].Sub(f.origin))
		}
		mask, bounds, ok := f.interior(id, componentBoxes)
		if !ok {
			continue
		}
		// Erode the filled shape so placements stay away from the outline.
		mask = erode(mask, bounds.Dx(), bounds.Dy(), 2)
		for _, i := range indices {
			r := image.Rectangle{Min: f.origin.Add(bounds.Min), Max: f.origin.Add(bounds.Max)}
			areas[i] = image.NewAlpha(r)
		}
		assigned := make(map[int]int, len(indices))
		for y := 0; y < bounds.Dy(); y++ {
			for x := 0; x < bounds.Dx(); x++ {
				if !mask[y*bounds.Dx()+x] {
					continue
				}
				owner := indices[0]
				bestDist, bestCenter := rectDistance2(boxes[owner].Sub(f.origin), x+bounds.Min.X, y+bounds.Min.Y), centerDistance2(boxes[owner].Sub(f.origin), x+bounds.Min.X, y+bounds.Min.Y)
				for _, i := range indices[1:] {
					box := boxes[i].Sub(f.origin)
					d, c := rectDistance2(box, x+bounds.Min.X, y+bounds.Min.Y), centerDistance2(box, x+bounds.Min.X, y+bounds.Min.Y)
					if d < bestDist || d == bestDist && (c < bestCenter || c == bestCenter && i < owner) {
						owner, bestDist, bestCenter = i, d, c
					}
				}
				p := image.Pt(f.origin.X+bounds.Min.X+x, f.origin.Y+bounds.Min.Y+y)
				areas[owner].SetAlpha(p.X, p.Y, color.Alpha{A: 255})
				assigned[owner]++
			}
		}
		for _, i := range indices {
			if assigned[i] == 0 {
				areas[i] = nil
			}
		}
	}
	// Fallback text remains on the page, so reserve its footprint in every
	// expanded area regardless of which component group was processed first.
	for _, area := range areas {
		if area == nil {
			continue
		}
		for j, box := range boxes {
			if areas[j] != nil {
				continue
			}
			reserve := box.Inset(-3).Intersect(image.Rectangle{Min: f.origin, Max: f.origin.Add(image.Pt(f.W, f.H))})
			r := area.Bounds().Intersect(reserve)
			for y := r.Min.Y; y < r.Max.Y; y++ {
				for x := r.Min.X; x < r.Max.X; x++ {
					area.SetAlpha(x, y, color.Alpha{})
				}
			}
		}
	}
	for i, area := range areas {
		if area != nil {
			areas[i] = trimAlpha(area)
		}
	}
	return areas
}

func trimAlpha(src *image.Alpha) *image.Alpha {
	var bounds image.Rectangle
	for y := src.Bounds().Min.Y; y < src.Bounds().Max.Y; y++ {
		for x := src.Bounds().Min.X; x < src.Bounds().Max.X; x++ {
			if src.AlphaAt(x, y).A != 0 {
				bounds = bounds.Union(image.Rect(x, y, x+1, y+1))
			}
		}
	}
	if bounds.Empty() {
		return nil
	}
	out := image.NewAlpha(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			out.SetAlpha(x, y, src.AlphaAt(x, y))
		}
	}
	return out
}

// componentFor applies Fit's existing trust checks while returning its chosen
// component. The old Fit method remains unchanged for comparison.
func (f *Finder) componentFor(textBox image.Rectangle) (int32, bool) {
	if textBox.Empty() {
		return -1, false
	}
	textBox = textBox.Sub(f.origin).Intersect(image.Rect(0, 0, f.W, f.H))
	if textBox.Empty() {
		return -1, false
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
		if n > bestN || n == bestN && (best < 0 || id < best) {
			best, bestN = id, n
		}
	}
	if best < 0 {
		return -1, false
	}
	s := f.stats[best]
	if s.rect.Min.X <= 2 || s.rect.Min.Y <= 2 || s.rect.Max.X >= f.W-2 || s.rect.Max.Y >= f.H-2 {
		return -1, false
	}
	if s.area*2 < textBox.Dx()*textBox.Dy() {
		return -1, false
	}
	tbArea := textBox.Dx() * textBox.Dy()
	if s.rect.Dx()*s.rect.Dy() > 8*tbArea || s.rect.Dx() > 5*textBox.Dx() || s.rect.Dy() > 5*textBox.Dy() {
		return -1, false
	}
	if s.rect.Dx() > 6*s.rect.Dy() || s.rect.Dy() > 6*s.rect.Dx() {
		return -1, false
	}
	return best, true
}

// interior fills only enclosed holes of the component; open concavities remain
// outside because they are reachable from the component bounding-box edge.
func (f *Finder) interior(id int32, textBoxes []image.Rectangle) ([]bool, image.Rectangle, bool) {
	r := f.stats[id].rect
	w, h := r.Dx(), r.Dy()
	if w < 5 || h < 5 {
		return nil, image.Rectangle{}, false
	}
	inside := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			inside[y*w+x] = f.labels[(r.Min.Y+y)*f.W+r.Min.X+x] == id
		}
	}
	outside := make([]bool, len(inside))
	queue := make([]int, 0, 2*(w+h))
	add := func(i int) {
		if !inside[i] && !outside[i] {
			outside[i] = true
			queue = append(queue, i)
		}
	}
	for x := 0; x < w; x++ {
		add(x)
		add((h-1)*w + x)
	}
	for y := 0; y < h; y++ {
		add(y * w)
		add(y*w + w - 1)
	}
	for head := 0; head < len(queue); head++ {
		i := queue[head]
		x, y := i%w, i/w
		if x > 0 {
			add(i - 1)
		}
		if x+1 < w {
			add(i + 1)
		}
		if y > 0 {
			add(i - w)
		}
		if y+1 < h {
			add(i + w)
		}
	}
	textFootprint := make([]bool, len(inside))
	for _, box := range textBoxes {
		box = box.Inset(-4).Intersect(r)
		for y := box.Min.Y; y < box.Max.Y; y++ {
			for x := box.Min.X; x < box.Max.X; x++ {
				textFootprint[(y-r.Min.Y)*w+x-r.Min.X] = true
			}
		}
	}
	for i := range inside {
		inside[i] = inside[i] || !outside[i] && textFootprint[i]
	}
	return inside, r, true
}

func erode(src []bool, w, h, radius int) []bool {
	out := make([]bool, len(src))
	for y := radius; y < h-radius; y++ {
		for x := radius; x < w-radius; x++ {
			good := true
			for dy := -radius; dy <= radius && good; dy++ {
				for dx := -radius; dx <= radius; dx++ {
					if !src[(y+dy)*w+x+dx] {
						good = false
						break
					}
				}
			}
			out[y*w+x] = good
		}
	}
	return out
}

func rectDistance2(r image.Rectangle, x, y int) int {
	dx, dy := 0, 0
	if x < r.Min.X {
		dx = r.Min.X - x
	} else if x >= r.Max.X {
		dx = x - r.Max.X + 1
	}
	if y < r.Min.Y {
		dy = r.Min.Y - y
	} else if y >= r.Max.Y {
		dy = y - r.Max.Y + 1
	}
	return dx*dx + dy*dy
}
func centerDistance2(r image.Rectangle, x, y int) int {
	dx, dy := 2*x-r.Min.X-r.Max.X, 2*y-r.Min.Y-r.Max.Y
	return dx*dx + dy*dy
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
	if f == nil {
		return image.Rectangle{}, false
	}
	if textBox.Empty() {
		return f.reject("empty")
	}
	textBox = textBox.Intersect(image.Rect(0, 0, f.W, f.H))
	if textBox.Empty() {
		return f.reject("outside page")
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
		return f.reject("no white region")
	}
	s := f.stats[best]

	// Reject the page background and very large artwork regions.
	if s.rect.Min.X <= 2 || s.rect.Min.Y <= 2 || s.rect.Max.X >= f.W-2 || s.rect.Max.Y >= f.H-2 {
		return f.reject("touches page border")
	}
	// The text must sit inside the region, and the region must be bigger.
	// A dense vertical block can cover most of the bubble, so allow the white
	// area to be somewhat smaller than the OCR box.
	if s.area*2 < textBox.Dx()*textBox.Dy() {
		return f.reject(fmt.Sprintf("white area %d < half of box %d", s.area, textBox.Dx()*textBox.Dy()))
	}
	// A region far larger than the text is usually several bubbles merged by
	// a shared white gap; expanding there would centre the text in the gutter.
	tbArea := textBox.Dx() * textBox.Dy()
	if s.rect.Dx()*s.rect.Dy() > 8*tbArea {
		return f.reject("region 8x larger than text")
	}
	if s.rect.Dx() > 5*textBox.Dx() || s.rect.Dy() > 5*textBox.Dy() {
		return f.reject("dimension ratio")
	}
	// Extremely elongated regions are usually panel gutters, not bubbles.
	dx, dy := s.rect.Dx(), s.rect.Dy()
	if dx > 6*dy || dy > 6*dx {
		return f.reject("elongated")
	}
	f.Reason = "ok"
	return s.rect.Inset(2), true
}

// reject records why a region was refused, for probes.
func (f *Finder) reject(reason string) (image.Rectangle, bool) {
	f.Reason = reason
	return image.Rectangle{}, false
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

// MergeDialogueColumns merges line-level detector boxes into dialogue blocks
// for vertical Japanese text: columns whose vertical ranges overlap and whose
// horizontal gap is within one column width belong to the same speech, while
// separate bubbles keep a clear column gap and stay apart. Lines no neighbour
// joins remain their own block.
func MergeDialogueColumns(boxes []image.Rectangle) []image.Rectangle {
	if len(boxes) < 2 {
		return boxes
	}
	parent := make([]int, len(boxes))
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(i int) int {
		for parent[i] != i {
			parent[i] = parent[parent[i]]
			i = parent[i]
		}
		return i
	}
	for i := 0; i < len(boxes); i++ {
		for j := i + 1; j < len(boxes); j++ {
			a, b := boxes[i], boxes[j]
			overlap := min(a.Max.Y, b.Max.Y) - max(a.Min.Y, b.Min.Y)
			if overlap*2 < min(a.Dy(), b.Dy()) {
				continue
			}
			gap := max(a.Min.X, b.Min.X) - min(a.Max.X, b.Max.X)
			if gap > max(16, min(a.Dx(), b.Dx())) {
				continue
			}
			ra, rb := find(i), find(j)
			if ra != rb {
				parent[rb] = ra
			}
		}
	}
	slots := make(map[int]int)
	out := make([]image.Rectangle, 0, len(boxes))
	for i, box := range boxes {
		root := find(i)
		if s, ok := slots[root]; ok {
			out[s] = out[s].Union(box)
			continue
		}
		slots[root] = len(out)
		out = append(out, box)
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if absInt(a.Min.Y-b.Min.Y) > 20 {
			return a.Min.Y < b.Min.Y
		}
		return a.Min.X > b.Min.X
	})
	return out
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
