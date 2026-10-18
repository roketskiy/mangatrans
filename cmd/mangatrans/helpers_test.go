package main

import (
	"image"
	"image/color"
	"testing"
)

// Ten em-square glyphs of ink at a realistic coverage ratio must estimate to
// roughly the true type size.
func TestEstimateFontSizeTracksInkDensity(t *testing.T) {
	const size = 16
	m := image.NewGray(image.Rect(0, 0, 400, 40))
	// Ten glyphs, each a filled size/2 x size/2 square (25% coverage).
	for g := 0; g < 10; g++ {
		for y := 0; y < size/2; y++ {
			for x := 0; x < size/2; x++ {
				m.SetGray(g*size+x+10, 10+y, color.Gray{Y: 255})
			}
		}
	}
	got := estimateFontSize(m, m.Bounds(), 10)
	if got < size*0.75 || got > size*1.3 {
		t.Fatalf("estimated %.1f, want within [%0.1f, %.1f]", got, size*0.75, size*1.3)
	}
}

func TestEstimateFontSizeRejectsEmptyInput(t *testing.T) {
	m := image.NewGray(image.Rect(0, 0, 40, 40))
	if got := estimateFontSize(nil, m.Bounds(), 10); got != 0 {
		t.Fatalf("nil mask gave %.1f", got)
	}
	if got := estimateFontSize(m, m.Bounds(), 10); got != 0 {
		t.Fatalf("empty mask gave %.1f", got)
	}
	if got := estimateFontSize(m, m.Bounds(), 0); got != 0 {
		t.Fatalf("no chars gave %.1f", got)
	}
}
