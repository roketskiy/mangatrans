package render

import (
	"image"
	"image/color"
	"os"
	"strings"
	"testing"

	"github.com/tdewolff/canvas"
)

const shapeTestFont = `C:\Windows\Fonts\msyh.ttc`

func TestShapeLayoutConservesRunes(t *testing.T) {
	font, err := canvas.LoadFontCollection(shapeTestFont, 0, canvas.FontRegular)
	if err != nil {
		t.Skipf("shape test font unavailable: %v", err)
	}
	for _, tc := range []struct {
		name     string
		vertical bool
		mask     *image.Alpha
	}{
		{name: "vertical lobes", vertical: true, mask: dumbbellMask(true)},
		{name: "horizontal lobes", mask: dumbbellMask(false)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			text := "连接的气泡需要完整文字。第二个气泡"
			placements, ok := shapeLayout(tc.mask, opaqueBounds(tc.mask), []rune(text), tc.vertical, font, color.Black, 4, opaqueBounds(tc.mask), nil)
			if !ok {
				t.Fatal("shape layout failed")
			}
			var got strings.Builder
			for _, p := range placements {
				got.WriteString(p.chunk)
			}
			if got.String() != text {
				t.Fatalf("layout changed text: got %q, want %q", got.String(), text)
			}
		})
	}
}

func TestShapeLayoutBalancesAdjacentLanes(t *testing.T) {
	font, err := canvas.LoadFontCollection(shapeTestFont, 0, canvas.FontRegular)
	if err != nil {
		t.Skipf("shape test font unavailable: %v", err)
	}
	mask := image.NewAlpha(image.Rect(0, 0, 42, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 42; x++ {
			mask.SetAlpha(x, y, color.Alpha{A: 255})
		}
	}
	text := "天地玄黄宇宙洪荒日月盈"
	placements, ok := shapeLayout(mask, mask.Bounds(), []rune(text), true, font, color.Black, 8, mask.Bounds(), nil)
	if !ok {
		t.Fatal("shape layout failed")
	}
	if len(placements) != 2 {
		t.Fatalf("got %d lanes, want 2", len(placements))
	}
	left, right := []rune(placements[0].chunk), []rune(placements[1].chunk)
	if got := len(left) - len(right); got < -1 || got > 1 {
		t.Fatalf("unbalanced chunks: %q (%d) and %q (%d)", placements[0].chunk, len(left), placements[1].chunk, len(right))
	}
	if placements[0].chunk+placements[1].chunk != text {
		t.Fatalf("rebalancing changed text: %q", placements[0].chunk+placements[1].chunk)
	}
}

func TestBestLaneTextDoesNotLeaveClosingPunctuationForNextLane(t *testing.T) {
	font, err := canvas.LoadFontCollection(shapeTestFont, 0, canvas.FontRegular)
	if err != nil {
		t.Skipf("shape test font unavailable: %v", err)
	}
	lane := shapeLane{rect: image.Rect(0, 0, 18, 100)}
	plain := []rune(strings.Repeat("天", 40))
	_, capacity := bestLaneText(lane, true, 8, font, color.Black, plain)
	if capacity < 2 || capacity >= len(plain) {
		t.Fatalf("unexpected test lane capacity: %d", capacity)
	}
	text := append([]rune(strings.Repeat("天", capacity)), []rune("，地")...)
	_, n := bestLaneText(lane, true, 8, font, color.Black, text)
	if n == 0 || (n < len(text) && isClosing(text[n])) {
		t.Fatalf("lane ended before a closing punctuation mark: consumed %d of %d runes", n, len(text))
	}
}

func TestRenderShapeKeepsInkInsideMask(t *testing.T) {
	if _, err := os.Stat(shapeTestFont); err != nil {
		t.Skip("Microsoft YaHei font is not installed")
	}
	background := color.RGBA{R: 210, G: 210, B: 210, A: 255}
	for _, tc := range []struct {
		name     string
		vertical bool
	}{
		{name: "vertical", vertical: true},
		{name: "horizontal"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			page := image.NewRGBA(image.Rect(0, 0, 220, 180))
			for y := 0; y < 180; y++ {
				for x := 0; x < 220; x++ {
					page.SetRGBA(x, y, background)
				}
			}
			area := dumbbellMask(tc.vertical)
			out, err := Render(page, []Block{{
				Box: image.Rect(15, 20, 205, 160), Text: "Connected bubbles need enough room for every glyph。第二个气泡",
				Vertical: tc.vertical, Area: area,
			}}, Options{FontPath: shapeTestFont, FontIndex: 0, MinSize: 2, MaxSize: 12, TextColor: color.RGBA{A: 255}})
			if err != nil {
				t.Fatal(err)
			}
			changed := 0
			for y := 0; y < 180; y++ {
				for x := 0; x < 220; x++ {
					if out.At(x, y) != background {
						changed++
						if area.AlphaAt(x, y).A != 255 {
							t.Fatalf("ink outside safe mask at (%d,%d)", x, y)
						}
					}
				}
			}
			if changed == 0 {
				t.Fatal("shape render produced no ink")
			}
		})
	}
}

func TestRenderShapeRejectsEmptyArea(t *testing.T) {
	if _, err := os.Stat(shapeTestFont); err != nil {
		t.Skip("Microsoft YaHei font is not installed")
	}
	page := image.NewRGBA(image.Rect(0, 0, 40, 40))
	_, err := Render(page, []Block{{Box: image.Rect(5, 5, 35, 35), Text: "text", Area: image.NewAlpha(image.Rect(0, 0, 40, 40))}},
		Options{FontPath: shapeTestFont, MinSize: 2, MaxSize: 12})
	if err == nil {
		t.Fatal("empty shape should return an error so callers can preserve the source page")
	}
}

// dumbbellMask makes two broad lobes connected by a thin neck. Some scan
// strips therefore contain separate writable runs while the overall shape
// remains connected, which catches layouts that only use one largest blob.
func dumbbellMask(vertical bool) *image.Alpha {
	m := image.NewAlpha(image.Rect(0, 0, 220, 180))
	for y := 0; y < 180; y++ {
		for x := 0; x < 220; x++ {
			ax, ay := x, y
			if vertical {
				ax, ay = y, x
			}
			left := square(ax-58)*square(62)+square(ay-90)*square(50) <= square(50)*square(62)
			right := square(ax-162)*square(62)+square(ay-90)*square(50) <= square(50)*square(62)
			neck := ax >= 58 && ax <= 162 && ay >= 87 && ay <= 93
			if left || right || neck {
				m.SetAlpha(x, y, color.Alpha{A: 255})
			}
		}
	}
	return m
}

func square(n int) int { return n * n }
