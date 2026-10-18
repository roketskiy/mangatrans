package bubble

import (
	"image"
	"image/color"
	"sort"
)

// maskSegment is one dialogue slice cut from a detector box. Pixels keeps the
// exact text-mask pixels owned by this segment, in absolute coordinates.
type maskSegment struct {
	bounds image.Rectangle
	pixels []image.Point
}

// maskBand is a contiguous run of active projection columns.
type maskBand struct{ start, end int }

// splitBoxByMaskGaps cuts a detector box into dialogue segments using the
// detector's text mask. Adjacent dialogues that share one box leave a fully
// blank band between their text pixels, so a projection over the reading axis
// separates them without relying on bubble colour or outline shape. The box
// aspect ratio picks the axis: tall boxes hold vertical columns separated
// across X, wide boxes hold horizontal rows separated across Y.
//
// Returns nil when the pixels do not form at least two credible segments.
func splitBoxByMaskGaps(textMask *image.Gray, box image.Rectangle, sourceVertical bool) []maskSegment {
	if textMask == nil {
		return nil
	}
	vertical := box.Dy() > box.Dx()
	project, inside := projectMask(textMask, box, vertical)
	total := 0
	for _, v := range project {
		total += v
	}
	if total < 64 {
		return nil
	}

	// A gap must be quiet, wide relative to the busiest band, and flanked by
	// real text on both sides. Character strokes inside one dialogue never
	// leave such a band on the reading axis, while separate dialogues do.
	maxColumn := 0
	for _, v := range project {
		maxColumn = max(maxColumn, v)
	}
	quiet := max(1, maxColumn/50)
	const minGap = 6
	const minBand = 16

	var bands []maskBand
	start := -1
	for i, v := range project {
		if v > quiet {
			if start < 0 {
				start = i
			}
		} else if start >= 0 {
			if i-start >= minBand {
				bands = append(bands, maskBand{start, i})
			}
			start = -1
		}
	}
	if start >= 0 && len(project)-start >= minBand {
		bands = append(bands, maskBand{start, len(project)})
	}
	if len(bands) < 2 {
		return nil
	}

	// Keep the widest bands when there are too many to be dialogues.
	const maxSegments = 4
	if len(bands) > maxSegments {
		sorted := append([]maskBand(nil), bands...)
		sort.SliceStable(sorted, func(i, j int) bool {
			a, b := sorted[i], sorted[j]
			return projectVolume(project, a) > projectVolume(project, b)
		})
		keep := sorted[:maxSegments]
		sort.Slice(keep, func(i, j int) bool { return keep[i].start < keep[j].start })
		bands = keep
	}

	segments := make([]maskSegment, 0, len(bands))
	for _, b := range bands {
		seg := maskSegment{}
		for i := b.start; i < b.end; i++ {
			for _, p := range inside[i] {
				seg.pixels = append(seg.pixels, p)
			}
		}
		if len(seg.pixels) == 0 {
			continue
		}
		r := image.Rect(seg.pixels[0].X, seg.pixels[0].Y, seg.pixels[0].X+1, seg.pixels[0].Y+1)
		for _, p := range seg.pixels {
			r = r.Union(image.Rect(p.X, p.Y, p.X+1, p.Y+1))
		}
		seg.bounds = r
		segments = append(segments, seg)
	}
	if len(segments) < 2 {
		return nil
	}
	// Every survivor needs a credible share of the text; otherwise a stray
	// band past the dialogues would steal recognition work.
	minPixels := total / 8
	kept := segments[:0]
	for _, seg := range segments {
		if len(seg.pixels) >= minPixels {
			kept = append(kept, seg)
		}
	}
	if len(kept) < 2 {
		return nil
	}
	return kept
}

func projectVolume(project []int, b maskBand) int {
	volume := 0
	for i := b.start; i < b.end; i++ {
		volume += project[i]
	}
	return volume
}

// projectMask accumulates text pixels over the reading axis. vertical buckets
// by X and returns per-bucket pixel lists; horizontal buckets by Y.
func projectMask(textMask *image.Gray, box image.Rectangle, vertical bool) ([]int, [][]image.Point) {
	projArea := box.Intersect(textMask.Rect)
	if projArea.Empty() {
		return nil, nil
	}
	var n int
	if vertical {
		n = projArea.Dx()
	} else {
		n = projArea.Dy()
	}
	project := make([]int, n)
	inside := make([][]image.Point, n)
	for y := projArea.Min.Y; y < projArea.Max.Y; y++ {
		for x := projArea.Min.X; x < projArea.Max.X; x++ {
			if textMask.GrayAt(x, y).Y == 0 {
				continue
			}
			var slot int
			if vertical {
				slot = x - projArea.Min.X
			} else {
				slot = y - projArea.Min.Y
			}
			project[slot]++
			inside[slot] = append(inside[slot], image.Pt(x, y))
		}
	}
	return project, inside
}

// growSegmentMask builds a source-pixel owner mask for one segment: segment
// pixels stay owned, and every other pixel inside the grown crop joins the
// nearest segment through BFS, so OCR crops never show a sibling's glyphs.
func growSegmentMask(segments []maskSegment, crop image.Rectangle) []*image.Alpha {
	masks := make([]*image.Alpha, len(segments))
	for i := range masks {
		masks[i] = image.NewAlpha(crop)
	}
	w := crop.Dx()
	owners := make([]int, w*crop.Dy())
	type seed struct{ idx, owner int }
	var queue []seed
	for si, seg := range segments {
		for _, p := range seg.pixels {
			if !p.In(crop) {
				continue
			}
			idx := (p.Y-crop.Min.Y)*w + p.X - crop.Min.X
			if owners[idx] == 0 {
				owners[idx] = si + 1
				masks[si].SetAlpha(p.X, p.Y, color.Alpha{A: 255})
				queue = append(queue, seed{idx, si + 1})
			}
		}
	}
	for head := 0; head < len(queue); head++ {
		cur := queue[head]
		x, y := cur.idx%w, cur.idx/w
		for _, n := range []image.Point{{X: x - 1, Y: y}, {X: x + 1, Y: y}, {X: x, Y: y - 1}, {X: x, Y: y + 1}} {
			if n.X < 0 || n.X >= w || n.Y < 0 || n.Y >= crop.Dy() {
				continue
			}
			j := n.Y*w + n.X
			if owners[j] == 0 {
				owners[j] = cur.owner
				queue = append(queue, seed{j, cur.owner})
			}
		}
	}
	for y := crop.Min.Y; y < crop.Max.Y; y++ {
		for x := crop.Min.X; x < crop.Max.X; x++ {
			if o := owners[(y-crop.Min.Y)*w+x-crop.Min.X]; o > 0 {
				masks[o-1].SetAlpha(x, y, color.Alpha{A: 255})
			}
		}
	}
	return masks
}
