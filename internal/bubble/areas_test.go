package bubble

import (
	"image"
	"image/color"
	"testing"
)

func TestFitAreasPreservesConcavityAndFillsEnclosedHole(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 100, 80)) // black is blocked
	fillGray(img, image.Rect(20, 12, 80, 68), color.Gray{Y: 255})
	// Open notch from the top edge: this is exterior and must stay excluded.
	fillGray(img, image.Rect(45, 12, 55, 38), color.Gray{Y: 0})
	// Enclosed black text-shaped hole: this belongs to the bubble interior.
	fillGray(img, image.Rect(32, 43, 42, 53), color.Gray{Y: 0})
	// Enclosed artwork island outside OCR: never expose it as text-safe area.
	fillGray(img, image.Rect(60, 43, 70, 53), color.Gray{Y: 0})
	f := Find(img, 210)
	areas := f.FitAreas([]image.Rectangle{image.Rect(24, 36, 50, 58)})
	if areas[0] == nil {
		t.Fatal("expected a trusted bubble area")
	}
	if got := areas[0].AlphaAt(50, 22).A; got != 0 {
		t.Fatalf("open notch included: alpha=%d", got)
	}
	if got := areas[0].AlphaAt(36, 48).A; got == 0 {
		t.Fatal("enclosed interior hole was not filled")
	}
	if got := areas[0].AlphaAt(64, 48).A; got != 0 {
		t.Fatalf("artwork hole outside OCR footprint included: alpha=%d", got)
	}
}

func TestFitAreasPartitionsSharedComponentStably(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 100, 80))
	fillGray(img, image.Rect(20, 15, 80, 65), color.Gray{Y: 255})
	f := Find(img, 210)
	boxes := []image.Rectangle{image.Rect(26, 28, 46, 48), image.Rect(54, 28, 74, 48)}
	first := f.FitAreas(boxes)
	second := f.FitAreas(boxes)
	for i := range first {
		if first[i] == nil || second[i] == nil {
			t.Fatalf("area %d unexpectedly nil", i)
		}
		if first[i].AlphaAt(boxes[i].Min.X+2, boxes[i].Min.Y+2).A == 0 {
			t.Fatalf("area %d lost its original text position", i)
		}
		for y := 15; y < 65; y++ {
			for x := 20; x < 80; x++ {
				if first[i].AlphaAt(x, y).A != second[i].AlphaAt(x, y).A {
					t.Fatalf("area %d changed at %v", i, image.Pt(x, y))
				}
				if first[i].AlphaAt(x, y).A != 0 && first[1-i].AlphaAt(x, y).A != 0 {
					t.Fatalf("areas overlap at %v", image.Pt(x, y))
				}
			}
		}
	}
}

func TestFitAreasReservesFallbackTextAndUsesAbsoluteCoordinates(t *testing.T) {
	bounds := image.Rect(10, 20, 110, 100)
	img := image.NewGray(bounds)
	fillGray(img, image.Rect(30, 35, 90, 85), color.Gray{Y: 255})
	f := Find(img, 210)
	large := image.Rect(38, 48, 58, 68)
	tiny := image.Rect(70, 52, 74, 56) // fails Fit's size-ratio trust checks
	areas := f.FitAreas([]image.Rectangle{large, tiny})
	if areas[0] == nil || areas[1] != nil {
		t.Fatalf("areas=(%v,%v), want trusted expansion plus fallback", areas[0] != nil, areas[1] != nil)
	}
	if got := areas[0].AlphaAt(tiny.Min.X, tiny.Min.Y).A; got != 0 {
		t.Fatalf("expanded area overlaps fallback text: alpha=%d", got)
	}
	if got := areas[0].AlphaAt(large.Min.X+2, large.Min.Y+2).A; got == 0 {
		t.Fatal("expanded area lost its source text position")
	}
	if areas[0].Bounds().Min.X < bounds.Min.X || areas[0].Bounds().Min.Y < bounds.Min.Y {
		t.Fatalf("returned non-absolute bounds %v", areas[0].Bounds())
	}
}

func TestFitAreasRejectsBoundaryLeakage(t *testing.T) {
	img := image.NewGray(image.Rect(0, 0, 100, 80))
	fillGray(img, image.Rect(0, 10, 35, 60), color.Gray{Y: 255})
	f := Find(img, 210)
	if got := f.FitAreas([]image.Rectangle{image.Rect(8, 25, 20, 37)})[0]; got != nil {
		t.Fatalf("boundary-connected region returned area %v", got.Bounds())
	}
}

func fillGray(img *image.Gray, r image.Rectangle, c color.Gray) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			img.SetGray(x, y, c)
		}
	}
}
