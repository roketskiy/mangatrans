package bubble

import (
	"image"
	"image/color"
	"sort"
)

// Balloon is one segmented speech-bubble instance in absolute page
// coordinates, typically produced by a YOLO-seg model.
type Balloon struct {
	Box  image.Rectangle
	Mask *image.Alpha // 255 inside the balloon
}

// minShare is the fraction of a box's text pixels below which a balloon does
// not claim a share when splitting a box across instances.
const minShare = 0.15

// wholeShare lets a balloon own the whole box once it holds this much of the
// text; small stray counts elsewhere then never manufacture a split.
const wholeShare = 0.8

// SplitWithBubbles assigns each detector box to segmented balloon instances
// before falling back to the geometric splitters. A box whose text pixels sit
// mostly inside one balloon stays whole and inherits that balloon as its
// layout area, which covers dark and irregular bubbles the white-interior
// heuristic cannot see. A box spanning several balloons splits at the
// instance boundaries, one region per balloon. Boxes no balloon explains go
// through SplitTextRegions' geometric path.
func SplitWithBubbles(source image.Image, textMask *image.Gray, balloons []Balloon, boxes []image.Rectangle) []TextRegion {
	areas := Find(source, 210).FitAreas(boxes)
	// Layout masks shrink a little so glyphs keep a margin from the outline;
	// membership still uses the full masks.
	inner := make([]Balloon, len(balloons))
	for i, bl := range balloons {
		inner[i] = Balloon{Box: bl.Box, Mask: erodeAlpha(bl.Mask, 2)}
	}
	var out []TextRegion
	for parent, box := range boxes {
		if regions, ok := balloonRegions(source, textMask, balloons, inner, box, parent); ok {
			out = append(out, regions...)
			continue
		}
		out = append(out, legacyRegions(source, textMask, areas, boxes, parent)...)
	}
	return out
}

// balloonRegions groups one box's text pixels by the balloon instance holding
// them. It reports false when the box is unexplained: no balloons, too little
// in-balloon text, or a split the shares do not back.
func balloonRegions(source image.Image, textMask *image.Gray, balloons, inner []Balloon, box image.Rectangle, parent int) ([]TextRegion, bool) {
	if textMask == nil || len(balloons) == 0 {
		return nil, false
	}
	r := box.Inset(-8).Intersect(source.Bounds())
	if r.Empty() {
		return nil, false
	}
	counts := make([]int, len(balloons))
	total := 0
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if textMask.GrayAt(x, y).Y == 0 {
				continue
			}
			total++
			for k := range balloons {
				if image.Pt(x, y).In(balloons[k].Box) && balloons[k].Mask.AlphaAt(x, y).A == 255 {
					counts[k]++
					break
				}
			}
		}
	}
	if total < 64 {
		return nil, false
	}

	claimed := 0
	for _, c := range counts {
		claimed += c
	}
	order := make([]int, len(balloons))
	for k := range order {
		order[k] = k
	}
	sort.Slice(order, func(i, j int) bool { return counts[order[i]] > counts[order[j]] })
	if float64(counts[order[0]]) >= float64(total)*wholeShare {
		// One balloon explains the box. Interior dialogue boundaries still
		// come from the text-mask gap splitter; every child then inherits the
		// balloon shape, which covers dark and irregular interiors the white
		// heuristic cannot see.
		regions, ok := maskGapRegions(source, textMask, box, parent)
		if ok {
			for i := range regions {
				regions[i].Area = inner[order[0]].Mask
			}
			return regions, true
		}
		region := TextRegion{Box: box, Area: inner[order[0]].Mask, Parent: parent}
		return []TextRegion{region}, true
	}
	if claimed*10 < total*8 {
		// Much of the text sits outside every balloon; trust the geometric
		// splitters instead of carving along balloon edges blindly.
		return nil, false
	}
	var holders []int
	for _, k := range order {
		if counts[k] >= 24 && float64(counts[k]) >= float64(total)*minShare {
			holders = append(holders, k)
		}
	}
	if len(holders) < 2 {
		return nil, false
	}

	segs := make([]maskSegment, len(holders))
	for i, k := range holders {
		segs[i] = maskSegment{bounds: balloons[k].Box}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				if textMask.GrayAt(x, y).Y == 0 {
					continue
				}
				if image.Pt(x, y).In(balloons[k].Box) && balloons[k].Mask.AlphaAt(x, y).A == 255 {
					segs[i].pixels = append(segs[i].pixels, image.Pt(x, y))
				}
			}
		}
		if len(segs[i].pixels) == 0 {
			return nil, false
		}
	}
	crop := box.Inset(-8).Intersect(source.Bounds())
	masks := growSegmentMask(segs, crop)
	children := make([]TextRegion, len(holders))
	for i, k := range holders {
		children[i] = TextRegion{
			Box:      segs[i].bounds,
			Area:     inner[k].Mask,
			CropMask: masks[i],
			Parent:   parent,
		}
	}
	// Reading order for vertical text runs right to left; horizontal reads
	// top to bottom, which the bounds order already is.
	sort.SliceStable(children, func(i, j int) bool {
		a, b := children[i].Box, children[j].Box
		if min(a.Max.Y, b.Max.Y)-max(a.Min.Y, b.Min.Y) > min(a.Dy(), b.Dy())/3 {
			return a.Min.X > b.Min.X
		}
		return a.Min.Y < b.Min.Y
	})
	return children, true
}

// erodeAlpha shrinks the opaque region by radius pixels with a two-pass
// chamfer distance transform, so rendered glyphs keep a margin from the
// balloon outline.
func erodeAlpha(mask *image.Alpha, radius int) *image.Alpha {
	b := mask.Bounds()
	w, h := b.Dx(), b.Dy()
	const big = 1 << 30
	dist := make([]int, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask.AlphaAt(b.Min.X+x, b.Min.Y+y).A == 255 {
				dist[y*w+x] = big
			}
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if dist[i] == 0 {
				continue
			}
			d := dist[i]
			if x > 0 {
				d = min(d, dist[i-1]+1)
			}
			if y > 0 {
				d = min(d, dist[i-w]+1)
			}
			if x > 0 && y > 0 {
				d = min(d, dist[i-w-1]+2)
			}
			if x < w-1 && y > 0 {
				d = min(d, dist[i-w+1]+2)
			}
			dist[i] = d
		}
	}
	for y := h - 1; y >= 0; y-- {
		for x := w - 1; x >= 0; x-- {
			i := y*w + x
			if dist[i] == 0 {
				continue
			}
			d := dist[i]
			if x < w-1 {
				d = min(d, dist[i+1]+1)
			}
			if y < h-1 {
				d = min(d, dist[i+w]+1)
			}
			if x < w-1 && y < h-1 {
				d = min(d, dist[i+w+1]+2)
			}
			if x > 0 && y < h-1 {
				d = min(d, dist[i+w-1]+2)
			}
			dist[i] = d
		}
	}
	out := image.NewAlpha(b)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if dist[y*w+x] > radius {
				out.SetAlpha(b.Min.X+x, b.Min.Y+y, color.Alpha{A: 255})
			}
		}
	}
	return out
}
