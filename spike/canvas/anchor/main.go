package main

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"path/filepath"

	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers"
	"github.com/tdewolff/canvas/renderers/rasterizer"
)

// Probe where DrawText actually puts the box for each candidate anchor rule.
func main() {
	out := `E:\.cache\render-out`
	os.MkdirAll(out, 0o755)

	f, err := canvas.LoadFontFile(`C:\Windows\Fonts\msyh.ttc`, canvas.FontRegular)
	if err != nil {
		panic(err)
	}

	c := canvas.New(100, 100)
	ctx := canvas.NewContext(c)
	ctx.SetFillColor(color.RGBA{255, 255, 255, 255})
	ctx.DrawPath(0, 0, canvas.Rectangle(100, 100))
	ctx.SetFillColor(color.RGBA{0, 0, 0, 255})

	face := f.Face(8, canvas.Black)
	// A box we can locate: one line, left-top aligned inside a 40x20 area.
	t := canvas.NewTextBox(face, "测试", 0, 0, canvas.Left, canvas.Top, nil)
	fmt.Printf("text W=%.1f H=%.1f\n", t.Width, t.Height)

	// Hypothesis A: (x,y) is the top-left of the text box in mm, y down from
	// the canvas top means passing y=100-20 places the top edge 20mm from top.
	ctx.DrawText(30, 80, t) // expect top edge at y=20mm from top if y is measured from bottom

	img := rasterizer.Draw(c, canvas.DPMM(4), canvas.DefaultColorSpace)
	// 100mm at 4 dpmm = 400px. Find dark pixel bbox.
	bbox := darkBBox(img)
	fmt.Printf("canvas 100x100mm -> image %dx%d\n", img.Bounds().Dx(), img.Bounds().Dy())
	fmt.Printf("dark bbox: x %d..%d, y %d..%d\n", bbox[0], bbox[1], bbox[2], bbox[3])
	fmt.Println("expected if anchor is top-left with y-from-top: x 120..280, y 80..112")
	fmt.Println("expected if anchor is top-left with y-from-bottom: x 120..280, y 288..320")

	png := filepath.Join(out, "anchor.png")
	renderers.Write(png, c, canvas.DPMM(4))
}

func darkBBox(img *image.RGBA) [4]int {
	x0, y0, x1, y1 := 1<<30, 1<<30, -1, -1
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			c := img.RGBAAt(x, y)
			if c.R < 128 {
				if x < x0 {
					x0 = x
				}
				if x > x1 {
					x1 = x
				}
				if y < y0 {
					y0 = y
				}
				if y > y1 {
					y1 = y
				}
			}
		}
	}
	return [4]int{x0, x1, y0, y1}
}
