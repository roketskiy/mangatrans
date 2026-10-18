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
	for i, region := range regions {
		if region.Area == nil {
			t.Fatalf("child %d has no balloon mask", i)
		}
		atRight := region.Area.AlphaAt(150, 50).A == 255
		atLeft := region.Area.AlphaAt(50, 50).A == 255
		if i == 0 && (!atRight || atLeft) {
			t.Fatalf("child 0 is not the right balloon mask")
		}
		if i == 1 && (!atLeft || atRight) {
			t.Fatalf("child 1 is not the left balloon mask")
		}
	}
	if regions[0].CropMask == nil || regions[1].CropMask == nil {
		t.Fatalf("split children must own their source pixels")
	}
}

// Layout masks are eroded so glyphs stay clear of the balloon outline.
func TestErodeAlphaShrinksByRadius(t *testing.T) {
	bounds := image.Rect(0, 0, 40, 40)
	balloon := balloonAt(bounds, image.Rect(5, 5, 35, 35))
	inner := erodeAlpha(balloon.Mask, 2)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			want := x >= 5+2 && x < 35-2 && y >= 5+2 && y < 35-2
			if got := inner.AlphaAt(x, y).A == 255; got != want {
				t.Fatalf("pixel (%d,%d) = %v, want %v", x, y, got, want)
			}
		}
	}
	// A balloon thinner than twice the radius erodes to nothing.
	thin := balloonAt(image.Rect(0, 0, 20, 20), image.Rect(8, 0, 12, 20))
	if got := erodeAlpha(thin.Mask, 2); got.AlphaAt(10, 10).A != 0 {
		t.Fatalf("thin balloon survived erosion")
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
	area := regions[0].Area
	if area == nil {
		t.Fatalf("region did not inherit a balloon mask")
	}
	// The inherited mask is the balloon eroded by the layout margin.
	if area.AlphaAt(30, 30).A != 255 || area.AlphaAt(11, 50).A != 0 {
		t.Fatalf("area mask is not the eroded balloon shape")
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
