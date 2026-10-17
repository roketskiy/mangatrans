package inpaint

import (
	"fmt"
	"image"
	"image/color"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"

	"mangatrans/internal/infer"
)

const side = 512

type Model struct {
	sess *ort.AdvancedSession
	img  *ort.Tensor[float32]
	mask *ort.Tensor[float32]
	out  *ort.Tensor[float32]
}

func Load(path string, cuda bool, vram int64) (*Model, error) {
	opts, err := infer.CUDAOptions(cuda, vram)
	if err != nil {
		return nil, fmt.Errorf("create inpaint session options: %w", err)
	}
	defer opts.Destroy()
	img, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, side, side))
	if err != nil {
		return nil, fmt.Errorf("allocate inpaint image tensor: %w", err)
	}
	mask, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, side, side))
	if err != nil {
		img.Destroy()
		return nil, fmt.Errorf("allocate inpaint mask tensor: %w", err)
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, side, side))
	if err != nil {
		img.Destroy()
		mask.Destroy()
		return nil, fmt.Errorf("allocate inpaint output tensor: %w", err)
	}
	sess, err := ort.NewAdvancedSession(path, []string{"image", "mask"}, []string{"output"}, []ort.Value{img, mask}, []ort.Value{out}, opts)
	if err != nil {
		img.Destroy()
		mask.Destroy()
		out.Destroy()
		return nil, fmt.Errorf("create inpaint session: %w", err)
	}
	return &Model{sess: sess, img: img, mask: mask, out: out}, nil
}

func (m *Model) Close() error {
	m.img.Destroy()
	m.mask.Destroy()
	m.out.Destroy()
	return m.sess.Destroy()
}

func (m *Model) Inpaint(src image.Image, textMask *image.Gray) (image.Image, error) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), src, b.Min, draw.Src)
	if textMask == nil {
		return rgba, nil
	}
	mask := make([]uint8, w*h)
	mb := textMask.Bounds()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mb.Min.X+x < mb.Max.X && mb.Min.Y+y < mb.Max.Y && mb.Min.X+x >= mb.Min.X && mb.Min.Y+y >= mb.Min.Y && textMask.GrayAt(mb.Min.X+x, mb.Min.Y+y).Y > 0 {
				mask[y*w+x] = 255
			}
		}
	}
	if !any(mask) {
		return rgba, nil
	}
	dilated := dilate(mask, w, h, 4)
	components := components(dilated, w, h)
	for _, r := range components {
		r = expand(r, 64, w, h)
		windows := []image.Rectangle{r}
		if r.Dx() > 3*r.Dy() || r.Dy() > 3*r.Dx() {
			windows = split(r)
		}
		for _, win := range windows {
			if solidBackground(rgba, mask, w, win) {
				fillMasked(rgba, mask, w, win, colorAt(rgba, win))
				continue
			}
			if err := m.inpaintWindow(rgba, mask, w, win); err != nil {
				return nil, err
			}
		}
	}
	return rgba, nil
}

func (m *Model) inpaintWindow(src *image.RGBA, mask []uint8, w int, r image.Rectangle) error {
	// The model input is a fixed 512x512 patch. Sample it with one mapping
	// (patch coords -> page coords) and reuse the exact inverse when writing
	// back, so read and write can never disagree on where a pixel lives.
	patch := image.NewRGBA(image.Rect(0, 0, side, side))
	draw.ApproxBiLinear.Scale(patch, patch.Bounds(), src, r, draw.Src, nil)
	md := m.mask.GetData()
	id := m.img.GetData()
	plane := side * side
	// pageX(x) maps patch x to page x, pageY likewise.
	pageX := func(px int) int {
		return r.Min.X + px*r.Dx()/side
	}
	pageY := func(py int) int {
		return r.Min.Y + py*r.Dy()/side
	}
	for py := 0; py < side; py++ {
		oy := pageY(py)
		if oy >= r.Max.Y {
			oy = r.Max.Y - 1
		}
		for px := 0; px < side; px++ {
			ox := pageX(px)
			if ox >= r.Max.X {
				ox = r.Max.X - 1
			}
			c := patch.RGBAAt(px, py)
			i := py*side + px
			id[i] = float32(c.R) / 255
			id[plane+i] = float32(c.G) / 255
			id[2*plane+i] = float32(c.B) / 255
			if mask[oy*w+ox] > 0 {
				md[i] = 1
			} else {
				md[i] = 0
			}
		}
	}
	// LaMa treats nonzero mask pixels as holes: koharu-slim/crates/koharu-ml/src/lama/mod.rs:183-186 thresholds the mask with gt(1.0).
	if err := m.sess.Run(); err != nil {
		return fmt.Errorf("run inpaint model for window %v: %w", r, err)
	}
	od := m.out.GetData()
	for py := 0; py < side; py++ {
		y := pageY(py)
		if y < r.Min.Y || y >= r.Max.Y {
			continue
		}
		for px := 0; px < side; px++ {
			x := pageX(px)
			if x < r.Min.X || x >= r.Max.X {
				continue
			}
			i := y*w + x
			if mask[i] == 0 {
				continue
			}
			edge := edgeDistance(mask, w, x, y)
			a := math.Min(1, float64(edge)/2)
			si := py*side + px
			p := src.PixOffset(x, y)
			for c := 0; c < 3; c++ {
				v := uint8(clamp(float64(od[c*plane+si]) * 255))
				src.Pix[p+c] = uint8(float64(src.Pix[p+c])*(1-a) + float64(v)*a)
			}
		}
	}
	return nil
}

func any(a []uint8) bool {
	for _, v := range a {
		if v > 0 {
			return true
		}
	}
	return false
}
func dilate(a []uint8, w, h, r int) []uint8 {
	t := make([]uint8, len(a))
	o := make([]uint8, len(a))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for q := max(0, x-r); q <= min(w-1, x+r); q++ {
				if a[y*w+q] > 0 {
					t[y*w+x] = 255
					break
				}
			}
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for q := max(0, y-r); q <= min(h-1, y+r); q++ {
				if t[q*w+x] > 0 {
					o[y*w+x] = 255
					break
				}
			}
		}
	}
	return o
}
func components(a []uint8, w, h int) []image.Rectangle {
	seen := make([]bool, len(a))
	var out []image.Rectangle
	for i, v := range a {
		if v == 0 || seen[i] {
			continue
		}
		q := []int{i}
		seen[i] = true
		x0, x1, y0, y1 := i%w, i%w, i/w, i/w
		for len(q) > 0 {
			n := q[len(q)-1]
			q = q[:len(q)-1]
			x, y := n%w, n/w
			x0 = min(x0, x)
			x1 = max(x1, x)
			y0 = min(y0, y)
			y1 = max(y1, y)
			for _, z := range []int{n - 1, n + 1, n - w, n + w} {
				if z >= 0 && z < len(a) && !seen[z] && a[z] > 0 && abs(z%w-x)+abs(z/w-y) == 1 {
					seen[z] = true
					q = append(q, z)
				}
			}
		}
		out = append(out, image.Rect(x0, y0, x1+1, y1+1))
	}
	return out
}
func expand(r image.Rectangle, n, w, h int) image.Rectangle {
	return image.Rect(max(0, r.Min.X-n), max(0, r.Min.Y-n), min(w, r.Max.X+n), min(h, r.Max.Y+n))
}
func split(r image.Rectangle) []image.Rectangle {
	var o []image.Rectangle
	if r.Dx() > 3*r.Dy() {
		for x := r.Min.X; x < r.Max.X; {
			z := min(r.Max.X, x+max(1, r.Dy()))
			o = append(o, image.Rect(x, r.Min.Y, z, r.Max.Y))
			x = z
		}
	} else {
		for y := r.Min.Y; y < r.Max.Y; {
			z := min(r.Max.Y, y+max(1, r.Dx()))
			o = append(o, image.Rect(r.Min.X, y, r.Max.Y, z))
			y = z
		}
	}
	return o
}
// solidBackground reports whether the unmasked pixels in the window are close
// to uniform. Manga bubbles and captions are usually flat white or black, so a
// small variance lets us skip the model entirely. The threshold is generous:
// screen-tone dots produce local variance, but they also defeat LaMa, so those
// windows should go to the model anyway. Variance is computed on 0..255 gray.
func solidBackground(im *image.RGBA, mask []uint8, w int, r image.Rectangle) bool {
	var s, s2 float64
	n := 0
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if mask[y*w+x] > 0 {
				continue
			}
			c := im.RGBAAt(x, y)
			v := float64(c.R+c.G+c.B) / 3
			s += v
			s2 += v * v
			n++
		}
	}
	if n == 0 {
		return false
	}
	return s2/float64(n)-math.Pow(s/float64(n), 2) < 40
}
func colorAt(im *image.RGBA, r image.Rectangle) color.RGBA {
	return im.RGBAAt((r.Min.X+r.Max.X)/2, (r.Min.Y+r.Max.Y)/2)
}
func fillMasked(im *image.RGBA, m []uint8, w int, r image.Rectangle, c color.RGBA) {
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			i := y*w + x
			if m[i] > 0 {
				im.SetRGBA(x, y, c)
			}
		}
	}
}
// edgeDistance approximates how deep a masked pixel sits inside the hole, in
// the 5x5 neighborhood. Pixels at the rim (distance 0) blend with the original
// image so the patch boundary does not show a hard seam.
func edgeDistance(m []uint8, w, x, y int) int {
	d := 3
	for dy := -2; dy <= 2; dy++ {
		for dx := -2; dx <= 2; dx++ {
			xx, yy := x+dx, y+dy
			if xx < 0 || yy < 0 || yy*w+xx >= len(m) || m[yy*w+xx] == 0 {
				cd := max(abs(dx), abs(dy))
				if cd < d {
					d = cd
				}
			}
		}
	}
	return d
}
func clamp(v float64) float64 { return math.Max(0, math.Min(255, v)) }

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
