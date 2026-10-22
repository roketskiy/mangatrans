package dbtext

import (
	"image"
	"image/draw"

	"mangatrans/internal/infer"
)

// RunTiled keeps full-page context and re-checks source-pixel tiles with 25%
// overlap, so small lines the downscaled full-page pass loses still get one
// native-resolution look. The result is the union: full-page boxes stay, tile
// boxes are added when they cover a new line, and the erasure mask takes the
// maximum of every pass.
func (m *Model) RunTiled(img image.Image, p Params, tileSize int) (*Result, error) {
	full, err := m.Run(img, p)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if tileSize <= 0 || (w <= tileSize && h <= tileSize) {
		return full, nil
	}
	var regions []image.Rectangle
	step := max(1, tileSize*3/4)
	for y := 0; y < h; y += step {
		bottom := min(y+tileSize, h)
		for x := 0; x < w; x += step {
			right := min(x+tileSize, w)
			regions = append(regions, image.Rect(max(0, right-tileSize), max(0, bottom-tileSize), right, bottom))
			if right == w {
				break
			}
		}
		if bottom == h {
			break
		}
	}
	for _, r := range regions {
		crop := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
		draw.Draw(crop, crop.Bounds(), img, r.Min.Add(b.Min), draw.Src)
		got, err := m.Run(crop, p)
		if err != nil {
			return nil, err
		}
		for _, box := range got.Boxes {
			box.X += float32(r.Min.X)
			box.Y += float32(r.Min.Y)
			// A line touching a crop edge is truncated; the overlapping crop
			// or the full-page pass owns it.
			if (r.Min.X > 0 && box.X < float32(r.Min.X+8)) || (r.Min.Y > 0 && box.Y < float32(r.Min.Y+8)) ||
				(r.Max.X < w && box.X+box.W > float32(r.Max.X-8)) || (r.Max.Y < h && box.Y+box.H > float32(r.Max.Y-8)) {
				continue
			}
			if duplicateOf(box, full.Boxes) {
				continue
			}
			full.Boxes = append(full.Boxes, box)
		}
		for y := 0; y < r.Dy(); y++ {
			for x := 0; x < r.Dx(); x++ {
				if v := got.Mask.GrayAt(x, y); v.Y > full.Mask.GrayAt(x+r.Min.X, y+r.Min.Y).Y {
					full.Mask.SetGray(x+r.Min.X, y+r.Min.Y, v)
				}
			}
		}
	}
	sortBoxes(full.Boxes)
	full.Scores = make([]float32, len(full.Boxes))
	return full, nil
}

func duplicateOf(box infer.Rect, existing []infer.Rect) bool {
	for _, old := range existing {
		x0, y0 := maxf32(old.X, box.X), maxf32(old.Y, box.Y)
		x1, y1 := minf32(old.X+old.W, box.X+box.W), minf32(old.Y+old.H, box.Y+box.H)
		if x1 > x0 && y1 > y0 && (x1-x0)*(y1-y0) >= 0.7*minf32(old.Area(), box.Area()) {
			return true
		}
	}
	return false
}

func minf32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
