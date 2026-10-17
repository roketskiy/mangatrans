package main

import (
	"fmt"
	"strings"

	"github.com/tdewolff/canvas"
)

func wrap(s string, vertical bool, size, aw, ah float64) string {
	limit := aw
	if vertical {
		limit = ah
	}
	per := int(limit / (size * 1.1))
	if per < 1 {
		per = 1
	}
	r := []rune(s)
	if len(r) <= per {
		return s
	}
	var b strings.Builder
	for i, ch := range r {
		if i > 0 && i%per == 0 {
			b.WriteByte('\n')
		}
		b.WriteRune(ch)
	}
	return b.String()
}

func main() {
	f, err := canvas.LoadFontCollection(`C:\Windows\Fonts\msyhbd.ttc`, 0, canvas.FontRegular)
	if err != nil {
		panic(err)
	}
	// Page 9 block 0: 80x163 px, padding 3 -> 18.5 x 39.25 mm.
	aw, ah := 18.5, 39.25
	text := "不过我已经不迷茫了，就出发吧"
	for _, size := range []float64{2, 3, 4, 5, 6, 7, 8, 9, 10} {
		rt := canvas.NewRichText(f.Face(size, canvas.Black))
		rt.SetWritingMode(canvas.VerticalRL)
		rt.SetTextOrientation(canvas.Upright)
		rt.WriteString(wrap(text, true, size, aw, ah))
		t := rt.ToText(aw, ah, canvas.Center, canvas.Center, nil)
		ib := t.OutlineBounds()
		fmt.Printf("size=%.1f overX=%v overY=%v W=%.2f H=%.2f inkW=%.2f inkH=%.2f cols=%d\n",
			size, t.OverflowsX, t.OverflowsY, t.Width, t.Height, ib.X1-ib.X0, ib.Y1-ib.Y0,
			strings.Count(wrap(text, true, size, aw, ah), "\n")+1)
	}
	fmt.Println("wrapped at size 6:", strings.ReplaceAll(wrap(text, true, 6, aw, ah), "\n", "|"))
}
