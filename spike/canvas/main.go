package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tdewolff/canvas"
	"github.com/tdewolff/canvas/renderers"
)

const outDir = `E:\work\transl\.cache\canvas-out`

func main() {
	os.MkdirAll(outDir, 0o755)

	fontPath, index := findCJKFont()
	f, err := canvas.LoadFontCollection(fontPath, index, canvas.FontRegular)
	if err != nil {
		f, err = canvas.LoadFontFile(fontPath, canvas.FontRegular)
		if err != nil {
			fmt.Println("FONT_FAIL", err)
			os.Exit(3)
		}
	}
	face := f.Face(7.0, canvas.Black)

	c := canvas.New(150, 110)
	ctx := canvas.NewContext(c)
	ctx.SetFillColor(canvas.White)
	ctx.DrawPath(0, 0, canvas.Rectangle(150, 110))
	ctx.SetFillColor(canvas.Black)

	// The writing mode must be set on the RichText before ToText lays it out.
	// Setting it on the returned Text has no effect, because layout already ran.
	rt := canvas.NewRichText(face)
	rt.SetWritingMode(canvas.VerticalRL)
	rt.SetTextOrientation(canvas.Upright)
	rt.WriteString("日本語の縦書きテスト。漫画の吹き出しの文字。")
	vert := rt.ToText(0, 90, canvas.Center, canvas.Center, nil)
	fmt.Printf("vert W=%.1f H=%.1f overX=%v overY=%v\n",
		vert.Width, vert.Height, vert.OverflowsX, vert.OverflowsY)
	ctx.DrawText(140, 105, vert)

	// A second column proves columns advance right to left.
	rt2 := canvas.NewRichText(face)
	rt2.SetWritingMode(canvas.VerticalRL)
	rt2.SetTextOrientation(canvas.Upright)
	rt2.WriteString("二列目。")
	v2 := rt2.ToText(0, 60, canvas.Center, canvas.Top, nil)
	ctx.DrawText(120, 105, v2)

	// Horizontal reference.
	h := canvas.NewTextBox(face, "横書きテスト", 0, 0, canvas.Left, canvas.Center, nil)
	ctx.DrawText(8, 12, h)

	png := filepath.Join(outDir, "vertical3.png")
	if err := renderers.Write(png, c, canvas.DPMM(4.0)); err != nil {
		fmt.Println("WRITE_FAIL", err)
		os.Exit(4)
	}
	st, _ := os.Stat(png)
	fmt.Printf("wrote %s %d bytes\n", png, st.Size())
}

func findCJKFont() (string, int) {
	cands := []struct {
		path  string
		index int
	}{
		{`C:\Windows\Fonts\msyh.ttc`, 0},
		{`C:\Windows\Fonts\msgothic.ttc`, 0},
		{`C:\Windows\Fonts\simsun.ttc`, 0},
	}
	for _, c := range cands {
		if _, err := os.Stat(c.path); err == nil {
			return c.path, c.index
		}
	}
	return "", 0
}
