package bubble

import (
	"image"
	"image/color"
	"testing"
)

func balloonAt(bounds image.Rectangle, fill image.Rectangle) Balloon {
	m := image.NewAlpha(bounds)
	for y := fill.Min.Y; y < fill.Max.Y; y++ {
		for x := fill.Min.X; x < fill.Max.X; x++ {
			m.SetAlpha(x, y, color.Alpha{A: 255})
		}
	}
	return Balloon{Box: bounds, Mask: m}
}

func textStrip(m *image.Gray, x0, x1, y0, y1 int) {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			m.SetGray(x, y, color.Gray{Y: 255})
		}
	}
}

// A box spanning two balloons splits at the instance boundary, and each child
// inherits its own balloon as the layout area.
func TestSplitWithBubblesSeparatesBoxAcrossTwoBalloons(t *testing.T) {
	page := image.Rect(0, 0, 200, 100)
	bA := balloonAt(image.Rect(0, 0, 100, 100), image.Rect(10, 10, 90, 90))
	bB := balloonAt(image.Rect(100, 0, 200, 100), image.Rect(110, 10, 190, 90))
	mask := image.NewGray(page)
	textStrip(mask, 20, 80, 20, 80)
	textStrip(mask, 120, 180, 20, 80)
	white := image.NewRGBA(page)

	regions := SplitWithBubbles(white, mask, []Balloon{bA, bB}, []image.Rectangle{page})
	if len(regions) != 2 {
		t.Fatalf("got %d regions, want 2", len(regions))
	}
	// Vertical text reads right to left: the right balloon comes first.
	if regions[0].Box.Min.X < 100 {
		t.Fatalf("first region is not the right balloon: %v", regions[0].Box)
	}
	if regions[0].Area != bB.Mask || regions[1].Area != bA.Mask {
		t.Fatalf("children did not inherit their own balloon masks")
	}
	if regions[0].CropMask == nil || regions[1].CropMask == nil {
		t.Fatalf("split children must own their source pixels")
	}
}

// A box fully inside one balloon stays whole and inherits that balloon.
func TestSplitWithBubblesKeepsSingleBalloonBoxWhole(t *testing.T) {
	page := image.Rect(0, 0, 200, 100)
	bA := balloonAt(image.Rect(0, 0, 200, 100), image.Rect(10, 10, 190, 90))
	mask := image.NewGray(page)
	textStrip(mask, 30, 170, 30, 70)
	white := image.NewRGBA(page)

	regions := SplitWithBubbles(white, mask, []Balloon{bA}, []image.Rectangle{image.Rect(20, 20, 180, 80)})
	if len(regions) != 1 {
		t.Fatalf("got %d regions, want 1", len(regions))
	}
	if regions[0].Area != bA.Mask {
		t.Fatalf("region did not inherit the balloon mask")
	}
	if regions[0].CropMask != nil {
		t.Fatalf("unsplit region needs no source-owner mask")
	}
}

// Text mostly outside every balloon falls through to the geometric path.
func TestSplitWithBubblesFallsBackWhenBalloonsDoNotExplainText(t *testing.T) {
	page := image.Rect(0, 0, 200, 100)
	bA := balloonAt(image.Rect(0, 0, 100, 100), image.Rect(10, 10, 90, 90))
	mask := image.NewGray(page)
	textStrip(mask, 20, 180, 40, 60) // half the text sits right of the balloon
	white := image.NewRGBA(page)

	regions := SplitWithBubbles(white, mask, []Balloon{bA}, []image.Rectangle{image.Rect(10, 30, 190, 70)})
	if len(regions) != 1 {
		t.Fatalf("got %d regions, want the single fallback region", len(regions))
	}
	if regions[0].Area == bA.Mask {
		t.Fatalf("fallback region must not claim the balloon as its area")
	}
}
