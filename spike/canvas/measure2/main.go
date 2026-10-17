package main

import (
	"fmt"
	"image"

	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers/rasterizer"
)

// Visual probe: draw the usable-area rectangle and the text into the same
// canvas, then measure where the ink actually lands relative to the box.
func main() {
	f, _ := canvas.LoadFontFile(`C:\Windows\Fonts\msyh.ttc`, canvas.FontRegular)
	face := f.Face(8, canvas.Black)

	// Page: 100x100mm canvas. Usable area: x 50..90, y 10..50 (in mm from
	// top-left, i.e. y-from-bottom 50..90).
	c := canvas.New(100, 100)
	ctx := canvas.NewContext(c)
	ctx.SetFillColor(image.White)
	ctx.DrawPath(0, 0, canvas.Rectangle(100, 100))

	// The usable-area outline in dark gray.
	ctx.SetFillColor(canvas.RGBA(0.7, 0.7, 0.7, 1))
	x0, yTopMM := 50.0, 10.0
	wBox, hBox := 40.0, 40.0
	yBottomMM := 100 - yTopMM
	ctx.DrawPath(x0, yBottomMM-hBox, canvas.Rectangle(wBox, hBox))

	// Text centered in that box via ToText, anchored at box top-left.
	ctx.SetFillColor(canvas.Black)
	rt := canvas.NewRichText(face)
	rt.SetWritingMode(canvas.VerticalRL)
	rt.SetTextOrientation(canvas.Upright)
	rt.WriteString("你好世界测试文字")
	t := rt.ToText(wBox, hBox, canvas.Center, canvas.Center, nil)
	ib := t.OutlineBounds()
	fmt.Printf("text W=%.2f H=%.2f ink x %.2f..%.2f y %.2f..%.2f\n",
		t.Width, t.Height, ib.X0, ib.X1, ib.Y0, ib.Y1)
	ctx.DrawText(x0, yBottomMM, t)

	img := rasterizer.Draw(c, canvas.DPMM(4), canvas.DefaultColorSpace)
	bbox, boxBBox := darkBBoxes(img, 179)
	fmt.Printf("image %dx%d px; text-ink bbox x %d..%d y %d..%d (mm %.1f..%.1f, %.1f..%.1f)\n",
		img.Bounds().Dx(), img.Bounds().Dy(),
		bbox[0], bbox[1], bbox[2], bbox[3],
		float64(bbox[0])/4, float64(bbox[1])/4, float64(bbox[2])/4, float64(bbox[3])/4)
	fmt.Printf("gray box bbox x %d..%d y %d..%d (mm %.1f..%.1f, %.1f..%.1f)\n",
		boxBBox[0], boxBBox[1], boxBBox[2], boxBBox[3],
		float64(boxBBox[0])/4, float64(boxBBox[1])/4, float64(boxBBox[2])/4, float64(boxBBox[3])/4)
}

// darkBBoxes returns the bbox of near-black pixels (text) and of mid-gray
// pixels (the box outline).
func darkBBoxes(img *image.RGBA, grayLevel uint8) ([4]int, [4]int) {
	textB := [4]int{1 << 30, -1, 1 << 30, -1}
	boxB := [4]int{1 << 30, -1, 1 << 30, -1}
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			c := img.RGBAAt(x, y)
			if c.R < 60 {
				extend(&textB, x, y)
			} else if c.R > 150 && c.R < 220 {
				extend(&boxB, x, y)
			}
		}
	}
	return textB, boxB
}

func extend(b *[4]int, x, y int) {
	if x < b[0] {
		b[0] = x
	}
	if x > b[1] {
		b[1] = x
	}
	if y < b[2] {
		b[2] = y
	}
	if y > b[3] {
		b[3] = y
	}
}
