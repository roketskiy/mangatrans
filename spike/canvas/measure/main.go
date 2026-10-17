package main

import (
	"fmt"

	"github.com/tdewolff/canvas"
)

// Print the reported extents of vertical and horizontal text to pin down the
// semantics of Width/Height/OutlineBounds per writing mode.
func main() {
	f, err := canvas.LoadFontFile(`C:\Windows\Fonts\msyh.ttc`, canvas.FontRegular)
	if err != nil {
		panic(err)
	}
	face := f.Face(8, canvas.Black)

	for _, c := range []struct {
		name string
		vert bool
	}{
		{"horizontal", false},
		{"vertical", true},
	} {
		rt := canvas.NewRichText(face)
		if c.vert {
			rt.SetWritingMode(canvas.VerticalRL)
			rt.SetTextOrientation(canvas.Upright)
		}
		rt.WriteString("你好世界测试文字")
		t := rt.ToText(60, 80, canvas.Center, canvas.Center, nil)
		ib := t.OutlineBounds()
		fmt.Printf("%s: W=%.2f H=%.2f overX=%v overY=%v | ink x %.2f..%.2f y %.2f..%.2f (%.2f x %.2f)\n",
			c.name, t.Width, t.Height, t.OverflowsX, t.OverflowsY,
			ib.X0, ib.X1, ib.Y0, ib.Y1, ib.X1-ib.X0, ib.Y1-ib.Y0)
	}
}
