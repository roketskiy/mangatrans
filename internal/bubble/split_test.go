package bubble

import (
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"
)

func TestDiagonalDialogueIsSplitBeforeOCR(t *testing.T) {
	read := func(name string) image.Image {
		f, err := os.Open("testdata/" + name)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		im, err := png.Decode(f)
		if err != nil {
			t.Fatal(err)
		}
		return im
	}
	source := read("connected-diagonal.png")
	mask := read("connected-diagonal-mask.png").(*image.Gray)
	parent := image.Rect(37, 16, 235, 224)
	regions := SplitTextRegions(source, mask, []image.Rectangle{parent})
	if len(regions) != 2 {
		t.Fatalf("got %d OCR regions, want two independent dialogues", len(regions))
	}
	// The upper-right thank-you and lower-left reply must have separate OCR
	// images, in reading order, with the original glyph pixels retained once.
	if regions[0].Box.Min.X < 150 || regions[0].Box.Min.Y > 30 || regions[1].Box.Max.X > 140 || regions[1].Box.Min.Y < 70 {
		t.Fatalf("wrong dialogue grouping/order: %v / %v", regions[0].Box, regions[1].Box)
	}
	for y := parent.Min.Y - 8; y < parent.Max.Y+8; y++ {
		for x := parent.Min.X - 8; x < parent.Max.X+8; x++ {
			owners := 0
			for _, region := range regions {
				if region.CropMask.AlphaAt(x, y).A == 255 {
					owners++
				}
			}
			if owners != 1 {
				t.Fatalf("source pixel (%d,%d) has %d OCR owners", x, y, owners)
			}
		}
	}
	for _, region := range regions {
		if region.Parent != 0 || region.Area == nil || region.Box.Empty() {
			t.Fatalf("incomplete child: %+v", region)
		}
	}
}

func TestOrdinaryOvalDoesNotSplit(t *testing.T) {
	mask := image.NewAlpha(image.Rect(0, 0, 200, 160))
	for y := 0; y < 160; y++ {
		for x := 0; x < 200; x++ {
			dx, dy := x-100, y-80
			if dx*dx*75*75+dy*dy*95*95 <= 95*95*75*75 {
				mask.SetAlpha(x, y, color.Alpha{A: 255})
			}
		}
	}
	if len(SplitArea(mask, mask.Bounds())) != 1 {
		t.Fatal("ordinary oval split into artificial dialogues")
	}
}
