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

// maskBand is a contiguous run of active projection slots.
type maskBand struct{ start, end int }

// splitBoxByMaskGaps cuts a detector box into dialogue segments using the
// detector's text mask. Adjacent dialogues that share one box leave a fully
// blank band between their text pixels, so a projection over a reading axis
// separates them without relying on bubble colour or outline shape. Both axes
// are tried: two vertical bubbles side by side form a wide box, so the box
// aspect ratio alone can pick the wrong axis. The axis exposing the most
// credible segments wins; ties keep the box's own reading axis.
//
// Segments come back in reading order: right to left for vertical text, top
// to bottom otherwise. Returns nil when the pixels do not form at least two
// credible segments.
func splitBoxByMaskGaps(textMask *image.Gray, box image.Rectangle) []maskSegment {
	if textMask == nil {
		return nil
	}
	boxVertical := box.Dy() >= box.Dx()
	var best []maskSegment
	for _, vertical := range []bool{boxVertical, !boxVertical} {
		segs := splitByAxis(textMask, box, vertical)
		if betterSegments(segs, best) {
			best = segs
		}
	}
	if len(best) < 2 {
		return nil
	}
	if boxVertical {
		// Vertical Japanese reads right to left across the segments.
		for i, j := 0, len(best)-1; i < j; i, j = i+1, j-1 {
			best[i], best[j] = best[j], best[i]
		}
	}
	return best
}

// betterSegments reports whether the candidate beats the incumbent. More
// segments win because a missed split is worse than an extra candidate on a
// genuinely conjoined box; balanced pixel shares break ties toward the box's
// own axis, which runs first.
func betterSegments(candidate, incumbent []maskSegment) bool {
	if len(candidate) == 0 {
		return false
	}
	if len(incumbent) == 0 {
		return true
	}
	if len(candidate) != len(incumbent) {
		return len(candidate) > len(incumbent)
	}
	return balance(candidate) < balance(incumbent)
}

// balance scores how unevenly the text mass is distributed over segments.
func balance(segs []maskSegment) float64 {
	total := 0
	for _, s := range segs {
		total += len(s.pixels)
	}
	mean := float64(total) / float64(len(segs))
	var variance float64
	for _, s := range segs {
		d := float64(len(s.pixels)) - mean
		variance += d * d
	}
	return variance / float64(len(segs))
}

// splitByAxis projects the mask over one reading axis and cuts at quiet
// bands. A gap must be quiet, wide, and flanked by real text on both sides;
// strokes inside one dialogue never leave such a band, while separate
// dialogues do.
func splitByAxis(textMask *image.Gray, box image.Rectangle, vertical bool) []maskSegment {
	project, inside := projectMask(textMask, box, vertical)
	if len(project) == 0 {
		return nil
	}
	total := 0
	maxSlot := 0
	for _, v := range project {
		total += v
		maxSlot = max(maxSlot, v)
	}
	if total < 64 {
		return nil
	}
	quiet := max(1, maxSlot/50)
	// Real dialogue gaps run 15px or more on manga pages, while gaps between
	// text columns inside one dialogue stay under 5px of pure silence.
	const minGap = 10
	const minBand = 16

	// Collect active runs first, then merge runs separated by a silence
	// narrower than minGap, and only then drop runs too narrow to be a
	// dialogue. Order matters: merging before the width filter keeps tight
	// column groups together instead of dropping them one by one.
	var runs []maskBand
	start := -1
	for i, v := range project {
		if v > quiet {
			if start < 0 {
				start = i
			}
		} else if start >= 0 {
			runs = append(runs, maskBand{start, i})
			start = -1
		}
	}
	if start >= 0 {
		runs = append(runs, maskBand{start, len(project)})
	}
	var bands []maskBand
	for _, r := range runs {
		if len(bands) > 0 && r.start-bands[len(bands)-1].end < minGap {
			bands[len(bands)-1].end = r.end
		} else {
			bands = append(bands, r)
		}
	}
	filtered := bands[:0]
	for _, b := range bands {
		if b.end-b.start >= minBand {
			filtered = append(filtered, b)
		}
	}
	bands = filtered
	if len(bands) < 2 {
		return nil
	}

	// Keep the busiest bands when there are too many to be dialogues.
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

	var segments []maskSegment
	for _, b := range bands {
		seg := maskSegment{}
		for i := b.start; i < b.end; i++ {
			seg.pixels = append(seg.pixels, inside[i]...)
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

// growSegmentMask builds a source-pixel owner mask for each segment: segment
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
