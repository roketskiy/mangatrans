package bubble

import (
	"image"
	"image/color"
	"testing"
)

func grayMask(size image.Point, draw func(m *image.Gray)) *image.Gray {
	m := image.NewGray(image.Rect(0, 0, size.X, size.Y))
	draw(m)
	return m
}

func fillColumnStrip(m *image.Gray, x0, x1, y0, y1 int) {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			m.SetGray(x, y, color.Gray{Y: 255})
		}
	}
}

// Two vertical dialogues share one detector box with a fully blank band
// between them; the splitter must hand back one segment per dialogue.
func TestSplitBoxByMaskGapsSeparatesTwoVerticalDialogues(t *testing.T) {
	m := grayMask(image.Pt(160, 200), func(m *image.Gray) {
		fillColumnStrip(m, 10, 40, 10, 190)
		fillColumnStrip(m, 120, 150, 10, 190)
	})
	box := image.Rect(0, 0, 160, 200)
	segs := splitBoxByMaskGaps(m, box, true)
	if len(segs) != 2 {
		t.Fatalf("got %d segments, want 2", len(segs))
	}
	a, b := segs[0].bounds, segs[1].bounds
	if a.Min.X >= b.Min.X {
		a, b = b, a
	}
	if a.Max.X > 60 || b.Min.X < 100 {
		t.Fatalf("segments did not follow the text clusters: %v %v", a, b)
	}
	if a.Dy() < 150 || b.Dy() < 150 {
		t.Fatalf("segment lost vertical extent: %v %v", a, b)
	}
}

// Columns of one dialogue sit close together with only narrow gaps between
// their strokes; the splitter must keep them together.
func TestSplitBoxByMaskGapsKeepsOneDialogueTogether(t *testing.T) {
	m := grayMask(image.Pt(120, 200), func(m *image.Gray) {
		for x := 20; x < 100; x += 16 {
			fillColumnStrip(m, x, x+10, 10, 190)
		}
	})
	box := image.Rect(0, 0, 120, 200)
	if segs := splitBoxByMaskGaps(m, box, true); len(segs) != 0 {
		t.Fatalf("got %d segments, want no split", len(segs))
	}
}

// Three conjoined dialogues must all come apart.
func TestSplitBoxByMaskGapsSeparatesThreeDialogues(t *testing.T) {
	m := grayMask(image.Pt(180, 240), func(m *image.Gray) {
		fillColumnStrip(m, 10, 35, 10, 230)
		fillColumnStrip(m, 80, 105, 10, 230)
		fillColumnStrip(m, 150, 175, 10, 230)
	})
	box := image.Rect(0, 0, 180, 240)
	segs := splitBoxByMaskGaps(m, box, true)
	if len(segs) != 3 {
		t.Fatalf("got %d segments, want 3", len(segs))
	}
}

// Wide boxes hold horizontal lines separated across Y.
func TestSplitBoxByMaskGapsSeparatesHorizontalDialogues(t *testing.T) {
	m := grayMask(image.Pt(200, 140), func(m *image.Gray) {
		fillColumnStrip(m, 10, 190, 15, 45)
		fillColumnStrip(m, 10, 190, 95, 125)
	})
	box := image.Rect(0, 0, 200, 140)
	segs := splitBoxByMaskGaps(m, box, false)
	if len(segs) != 2 {
		t.Fatalf("got %d segments, want 2", len(segs))
	}
	a, b := segs[0].bounds, segs[1].bounds
	if a.Min.Y > b.Min.Y {
		a, b = b, a
	}
	if a.Max.Y > 60 || b.Min.Y < 80 {
		t.Fatalf("segments did not follow the rows: %v %v", a, b)
	}
}

// A lone dialogue or empty pixels never produce segments.
func TestSplitBoxByMaskGapsRejectsTrivialInput(t *testing.T) {
	box := image.Rect(0, 0, 100, 100)
	if segs := splitBoxByMaskGaps(nil, box, true); segs != nil {
		t.Fatalf("nil mask gave %v", segs)
	}
	empty := grayMask(image.Pt(100, 100), func(m *image.Gray) {})
	if segs := splitBoxByMaskGaps(empty, box, true); segs != nil {
		t.Fatalf("empty mask gave %v", segs)
	}
}

// A faint stray band next to one real dialogue must not manufacture a split.
func TestSplitBoxByMaskGapsIgnoresTinySideBand(t *testing.T) {
	m := grayMask(image.Pt(200, 200), func(m *image.Gray) {
		fillColumnStrip(m, 20, 55, 10, 190)
		fillColumnStrip(m, 62, 90, 10, 190)
		fillColumnStrip(m, 150, 155, 80, 90)
	})
	box := image.Rect(0, 0, 200, 200)
	if segs := splitBoxByMaskGaps(m, box, true); len(segs) != 0 {
		t.Fatalf("got %d segments, want no split", len(segs))
	}
}

// Grown masks give every crop pixel an owner and keep sibling glyphs out.
func TestGrowSegmentMaskKeepsSiblingsWhite(t *testing.T) {
	m := grayMask(image.Pt(120, 160), func(m *image.Gray) {
		fillColumnStrip(m, 10, 30, 10, 150)
		fillColumnStrip(m, 90, 110, 10, 150)
	})
	box := image.Rect(0, 0, 120, 160)
	segs := splitBoxByMaskGaps(m, box, true)
	if len(segs) != 2 {
		t.Fatalf("got %d segments, want 2", len(segs))
	}
	crop := box
	masks := growSegmentMask(segs, crop)
	if len(masks) != 2 {
		t.Fatalf("got %d masks, want 2", len(masks))
	}
	for i, mask := range masks {
		other := segs[1-i]
		for y := crop.Min.Y; y < crop.Max.Y; y++ {
			for x := crop.Min.X; x < crop.Max.X; x++ {
				if image.Pt(x, y).In(other.bounds) && mask.AlphaAt(x, y).A == 255 {
					t.Fatalf("mask %d claims sibling pixel (%d,%d)", i, x, y)
				}
			}
		}
		if mask.AlphaAt(segs[i].bounds.Min.X+1, segs[i].bounds.Min.Y+1).A != 255 {
			t.Fatalf("mask %d lost its own pixels", i)
		}
	}
}
