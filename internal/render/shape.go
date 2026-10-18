package render

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"

	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers"
)

const shapePtToMM = 0.3527778

type shapeLane struct {
	// rect is a wholly writable rectangle in page coordinates.
	rect image.Rectangle
}

type shapeInterval struct{ start, end int }

type shapePlacement struct {
	lane  shapeLane
	text  *canvas.Text
	chunk string
}

// renderShape finds one common font size for the full translation, partitions
// the text over safe rectangular lanes, and rasterises each lane separately.
// Since every glyph canvas is contained by a lane whose every pixel is safe,
// the rendered ink cannot cross the supplied alpha mask. Lane windows also
// avoid the rectangles already handed to sibling dialogues sharing this
// bubble shape; occupied grows by this dialogue's window on success.
func renderShape(dst *image.RGBA, b Block, mask *image.Alpha, pageBounds image.Rectangle, font *canvas.Font, fill color.Color, min, max float64, occupied *[]image.Rectangle) error {
	runes := []rune(b.Text)
	if len(runes) == 0 {
		return nil
	}
	mb := opaqueBounds(mask).Intersect(pageBounds)
	if mb.Empty() {
		return fmt.Errorf("safe area is empty")
	}
	if !hasOpaquePixel(mask, mb) {
		return fmt.Errorf("safe area contains no fully writable pixels")
	}

	// The lane grid changes discretely with font size, so feasibility is not
	// strictly monotone. Try practical sizes from largest down in half-point
	// steps and always check the caller's exact minimum as the last candidate.
	anchor := b.Anchor
	if anchor.Empty() {
		anchor = mb
	}
	var placements []shapePlacement
	for size := max; size > min; size -= 0.5 {
		if p, ok := shapeLayout(mask, mb, runes, b.Vertical, font, fill, size, anchor, *occupied); ok {
			placements = p
			break
		}
	}
	if len(placements) == 0 {
		var ok bool
		placements, ok = shapeLayout(mask, mb, runes, b.Vertical, font, fill, min, anchor, *occupied)
		if !ok || len(placements) == 0 {
			return fmt.Errorf("translation does not fit safe area at minimum size %.2f", min)
		}
	}
	for _, p := range placements {
		*occupied = append(*occupied, p.lane.rect)
	}

	for _, p := range placements {
		r := p.lane.rect
		aw, ah := float64(r.Dx())*pxToMM, float64(r.Dy())*pxToMM
		c := canvas.New(aw, ah)
		ctx := canvas.NewContext(c)
		ctx.SetFillColor(fill)
		ctx.DrawText(0, ah, p.text)
		var buf bytes.Buffer
		if err := renderers.PNG(canvas.DPMM(1/pxToMM))(&buf, c); err != nil {
			return fmt.Errorf("rasterise shape lane %v: %w", r, err)
		}
		layer, err := png.Decode(&buf)
		if err != nil {
			return fmt.Errorf("decode shape lane: %w", err)
		}
		d := image.Rect(r.Min.X-pageBounds.Min.X, r.Min.Y-pageBounds.Min.Y,
			r.Max.X-pageBounds.Min.X, r.Max.Y-pageBounds.Min.Y)
		draw.Draw(dst, d, layer, image.Point{}, draw.Over)
	}
	return nil
}

func hasOpaquePixel(mask *image.Alpha, r image.Rectangle) bool {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if mask.AlphaAt(x, y).A == 255 {
				return true
			}
		}
	}
	return false
}

func opaqueBounds(mask *image.Alpha) image.Rectangle {
	var bounds image.Rectangle
	for y := mask.Bounds().Min.Y; y < mask.Bounds().Max.Y; y++ {
		for x := mask.Bounds().Min.X; x < mask.Bounds().Max.X; x++ {
			if mask.AlphaAt(x, y).A != 255 {
				continue
			}
			p := image.Rect(x, y, x+1, y+1)
			if bounds.Empty() {
				bounds = p
			} else {
				bounds = bounds.Union(p)
			}
		}
	}
	return bounds
}

// shapeLayout fits one independently recognized dialogue into safe lanes.
// Dialogue boundaries are decided before OCR; this function only wraps the
// corresponding translation, preserving its rune order. Lanes must overlap
// the anchor rectangle when possible, keeping each dialogue near its own
// original text instead of drifting across a shared bubble shape. Windows
// also dodge the occupied rectangles of sibling dialogues already placed on
// this bubble; the constraint relaxes stepwise so a crowded balloon still
// lays out rather than failing outright.
func shapeLayout(mask *image.Alpha, bounds image.Rectangle, runes []rune, vertical bool, font *canvas.Font, fill color.Color, size float64, anchor image.Rectangle, occupied []image.Rectangle) ([]shapePlacement, bool) {
	fontPx := size * shapePtToMM / pxToMM
	thickness := int(math.Ceil(fontPx * 1.55))
	if thickness < 4 {
		thickness = 4
	}
	lanes := scanShapeLanes(mask, bounds, vertical, thickness)
	if len(lanes) == 0 {
		return nil, false
	}
	// Estimate per-lane capacities cheaply, then validate candidate windows
	// with actual canvas overflow measurements.
	caps := make([]int, len(lanes))
	for i, lane := range lanes {
		long := lane.rect.Dx()
		if vertical {
			long = lane.rect.Dy()
		}
		caps[i] = int(float64(long) / math.Max(fontPx*0.88, 1))
		if caps[i] < 1 {
			caps[i] = 1
		}
	}
	// Find the shortest sequence of lanes that actually fits, preferring the
	// sequence nearest the anchor centre when widths tie. Constraint passes
	// relax stepwise: anchored and disjoint from siblings first, then merely
	// anchored, then anywhere.
	start, end, bestCount := -1, -1, len(lanes)+1
	bestDist := math.Inf(1)
	center := float64(anchor.Min.X+anchor.Max.X) / 2
	if !vertical {
		center = float64(anchor.Min.Y+anchor.Max.Y) / 2
	}
	for _, relax := range []int{0, 1, 2} {
		if start >= 0 {
			break
		}
		for i := range lanes {
			estimate := 0
			for j := i; j < len(lanes); j++ {
				estimate += caps[j]
				if estimate < len(runes) {
					continue
				}
				window := windowRect(lanes[i : j+1])
				if relax < 1 && window.Intersect(anchor).Empty() {
					continue
				}
				if relax < 2 && touchesAny(window, occupied) {
					continue
				}
				if _, ok := fitLaneWindow(lanes[i:j+1], runes, vertical, size, font, fill); !ok {
					continue
				}
				count := j - i + 1
				first, last := lanes[i].rect, lanes[j].rect
				mid := center
				if vertical {
					mid = float64(first.Min.X+last.Max.X) / 2
				} else {
					mid = float64(first.Min.Y+last.Max.Y) / 2
				}
				dist := math.Abs(mid - center)
				if count < bestCount || (count == bestCount && dist < bestDist) {
					start, end, bestCount, bestDist = i, j+1, count, dist
				}
				break
			}
		}
	}
	if start < 0 {
		return nil, false
	}

	placements, ok := fitLaneWindow(lanes[start:end], runes, vertical, size, font, fill)
	if !ok {
		return nil, false
	}
	return rebalancePlacements(placements, vertical, size, font, fill), true
}

// touchesAny reports whether the window overlaps an occupied rectangle.
func touchesAny(window image.Rectangle, occupied []image.Rectangle) bool {
	for _, r := range occupied {
		if !window.Intersect(r).Empty() {
			return true
		}
	}
	return false
}

// windowRect is the union rectangle of a run of consecutive lanes.
func windowRect(lanes []shapeLane) image.Rectangle {
	r := lanes[0].rect
	for _, lane := range lanes[1:] {
		r = r.Union(lane.rect)
	}
	return r
}

func fitLaneWindow(lanes []shapeLane, runes []rune, vertical bool, size float64, font *canvas.Font, fill color.Color) ([]shapePlacement, bool) {
	var out []shapePlacement
	pos := 0
	for _, lane := range lanes {
		if pos == len(runes) {
			break
		}
		text, n := bestLaneText(lane, vertical, size, font, fill, runes[pos:])
		if n == 0 {
			continue
		}
		chunk := string(runes[pos : pos+n])
		out = append(out, shapePlacement{lane: lane, text: text, chunk: chunk})
		pos += n
	}
	return out, pos == len(runes)
}

// rebalancePlacements moves boundary runes from fuller lanes to the next lane
// when both measured layouts still fit. This avoids leaving a one-glyph final
// column beside a mostly empty neighboring column while preserving order.
func rebalancePlacements(placements []shapePlacement, vertical bool, size float64, font *canvas.Font, fill color.Color) []shapePlacement {
	if len(placements) < 2 {
		return placements
	}
	maxMoves := 0
	for _, p := range placements {
		maxMoves += len([]rune(p.chunk))
	}
	moves := 0
	for pass := 0; pass < len(placements)*2 && moves < maxMoves; pass++ {
		changed := false
		for i := 0; i+1 < len(placements) && moves < maxMoves; i++ {
			left := []rune(placements[i].chunk)
			right := []rune(placements[i+1].chunk)
			for len(left) > len(right)+1 && moves < maxMoves {
				moved := left[len(left)-1]
				if isClosing(moved) || (len(left) > 1 && isOpening(left[len(left)-2])) {
					break
				}
				newLeft, newRight := left[:len(left)-1], append([]rune{moved}, right...)
				if len(newLeft) == 0 || isClosing(newRight[0]) {
					break
				}
				leftText, leftOK := laneText(placements[i].lane, vertical, size, font, fill, newLeft)
				rightText, rightOK := laneText(placements[i+1].lane, vertical, size, font, fill, newRight)
				if !leftOK || !rightOK {
					break
				}
				placements[i].chunk = string(newLeft)
				placements[i].text = leftText
				placements[i+1].chunk = string(newRight)
				placements[i+1].text = rightText
				left, right = newLeft, newRight
				moves++
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	return placements
}

// scanShapeLanes samples non-overlapping glyph-width strips. A run is admitted
// only when its entire rectangle is opaque, giving antialiased glyphs room
// inside the parent's already-inset mask.
func scanShapeLanes(mask *image.Alpha, bounds image.Rectangle, vertical bool, thickness int) []shapeLane {
	var lanes []shapeLane
	if vertical {
		first := bounds.Min.X + (bounds.Dx()%thickness)/2
		for x := first; x+thickness <= bounds.Max.X; x += thickness {
			runs := safeRuns(mask, bounds, image.Rect(x, bounds.Min.Y, x+thickness, bounds.Max.Y), true)
			for _, run := range runs {
				lanes = append(lanes, shapeLane{image.Rect(x, run.start, x+thickness, run.end)})
			}
		}
		// Right to left columns, top to bottom within disconnected portions.
		for i, j := 0, len(lanes)-1; i < j; i, j = i+1, j-1 {
			lanes[i], lanes[j] = lanes[j], lanes[i]
		}
		// Re-sort runs sharing a strip top-to-bottom after reversing strip order.
		for i := 0; i < len(lanes); {
			j := i + 1
			for j < len(lanes) && lanes[j].rect.Min.X == lanes[i].rect.Min.X {
				j++
			}
			for a, z := i, j-1; a < z; a, z = a+1, z-1 {
				lanes[a], lanes[z] = lanes[z], lanes[a]
			}
			i = j
		}
	} else {
		first := bounds.Min.Y + (bounds.Dy()%thickness)/2
		for y := first; y+thickness <= bounds.Max.Y; y += thickness {
			runs := safeRuns(mask, bounds, image.Rect(bounds.Min.X, y, bounds.Max.X, y+thickness), false)
			for _, run := range runs {
				lanes = append(lanes, shapeLane{image.Rect(run.start, y, run.end, y+thickness)})
			}
		}
	}
	return lanes
}

// safeRuns returns maximal intervals on the long axis for which all pixels
// across the supplied strip are fully writable.
func safeRuns(mask *image.Alpha, bounds, strip image.Rectangle, vertical bool) []shapeInterval {
	var runs []shapeInterval
	start := -1
	lo, hi := bounds.Min.X, bounds.Max.X
	if vertical {
		lo, hi = bounds.Min.Y, bounds.Max.Y
	}
	for n := lo; n < hi; n++ {
		safe := true
		if vertical {
			for x := strip.Min.X; x < strip.Max.X && safe; x++ {
				if mask.AlphaAt(x, n).A != 255 {
					safe = false
				}
			}
		} else {
			for y := strip.Min.Y; y < strip.Max.Y && safe; y++ {
				if mask.AlphaAt(n, y).A != 255 {
					safe = false
				}
			}
		}
		if safe && start < 0 {
			start = n
		} else if !safe && start >= 0 {
			runs = append(runs, shapeInterval{start, n})
			start = -1
		}
	}
	if start >= 0 {
		runs = append(runs, shapeInterval{start, hi})
	}
	return runs
}

func bestLaneText(lane shapeLane, vertical bool, size float64, font *canvas.Font, fill color.Color, runes []rune) (*canvas.Text, int) {
	lo, hi := 0, len(runes)
	var best *canvas.Text
	for lo < hi {
		mid := lo + (hi-lo+1)/2
		if t, ok := laneText(lane, vertical, size, font, fill, runes[:mid]); ok {
			lo, best = mid, t
		} else {
			hi = mid - 1
		}
	}
	if lo == 0 {
		return nil, 0
	}
	// Keep closing punctuation with the preceding lane where measured capacity
	// permits, and avoid ending a lane on an opening bracket.
	if lo < len(runes) && isClosing(runes[lo]) {
		if t, ok := laneText(lane, vertical, size, font, fill, runes[:lo+1]); ok {
			lo, best = lo+1, t
		} else {
			// Move a preceding ordinary rune with the closing punctuation rather
			// than accepting a lane whose next lane would start on punctuation.
			for lo > 1 && isClosing(runes[lo]) {
				lo--
				if !isOpening(runes[lo-1]) {
					break
				}
			}
			if lo <= 1 {
				return nil, 0
			}
			best, _ = laneText(lane, vertical, size, font, fill, runes[:lo])
		}
	}
	if lo > 1 && lo < len(runes) && isOpening(runes[lo-1]) {
		lo--
		best, _ = laneText(lane, vertical, size, font, fill, runes[:lo])
	}
	return best, lo
}

func laneText(lane shapeLane, vertical bool, size float64, font *canvas.Font, fill color.Color, runes []rune) (*canvas.Text, bool) {
	r := lane.rect
	aw, ah := float64(r.Dx())*pxToMM, float64(r.Dy())*pxToMM
	rt := canvas.NewRichText(font.Face(size, fill))
	if vertical {
		rt.SetWritingMode(canvas.VerticalRL)
		rt.SetTextOrientation(canvas.Upright)
	}
	// A single lane is one column/row. Splitting happens only at measured
	// capacity boundaries; source paragraphs were flattened before this stage.
	rt.WriteString(string(runes))
	t := rt.ToText(aw, ah, canvas.Center, canvas.Center, nil)
	return t, !t.Empty() && !t.OverflowsX && !t.OverflowsY
}
