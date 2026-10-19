package inpaint

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"math"
	"runtime/debug"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"

	"mangatrans/internal/infer"
	"mangatrans/internal/perf"
)

const side = 512

type Model struct {
	sess           *ort.AdvancedSession
	img            *ort.Tensor[float32]
	mask           *ort.Tensor[float32]
	out            *ort.Tensor[float32]
	ContextPadding int
	TileSize       int
	// MaskRadius controls glyph-edge growth, not rectangle expansion.
	MaskRadius *int
	MaskLimit  *image.Gray
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

func (m *Model) Inpaint(ctx context.Context, src image.Image, textMask *image.Gray) (out image.Image, err error) {
	// A malformed mask must never terminate the whole book. Convert any
	// unexpected panic into a page error so the caller can keep the original
	// page and continue with the next one.
	defer func() {
		if r := recover(); r != nil {
			out = nil
			err = fmt.Errorf("inpaint panic: %v\n%s", r, debug.Stack())
		}
	}()

	if src == nil {
		return nil, fmt.Errorf("inpaint: nil source image")
	}
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 {
		return nil, fmt.Errorf("inpaint: empty source image %v", b)
	}
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), src, b.Min, draw.Src)
	if textMask == nil {
		return rgba, nil
	}
	mask := make([]uint8, w*h)
	mb := textMask.Bounds()
	for y := 0; y < h; y++ {
		sy := mb.Min.Y + y
		if sy >= mb.Max.Y {
			continue
		}
		for x := 0; x < w; x++ {
			sx := mb.Min.X + x
			if sx >= mb.Max.X {
				continue
			}
			if textMask.GrayAt(sx, sy).Y > 0 {
				mask[y*w+x] = 255
			}
		}
	}
	if !any(mask) {
		return rgba, nil
	}
	radius := 6
	if m.MaskRadius != nil {
		radius = *m.MaskRadius
	}
	dilated := dilate(mask, w, h, radius)
	if m.MaskLimit != nil {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if m.MaskLimit.GrayAt(x, y).Y == 0 {
					dilated[y*w+x] = 0
				}
			}
		}
	}
	initialComponents := maskComponents(dilated, w, h)
	perf.Count(ctx, "inpaint_components", len(initialComponents))
	for _, component := range initialComponents {
		local := expand(component.bounds, 8, w, h)
		if bg, ok := localSolidBackground(rgba, dilated, w, local); ok {
			perf.Count(ctx, "inpaint_windows", 1)
			perf.Count(ctx, "inpaint_fills", 1)
			perf.Count(ctx, "inpaint_local_fills", 1)
			fillComponent(rgba, dilated, w, component, bg)
		}
	}
	edges := edgeDistances(dilated, w, h)
	components := maskComponents(dilated, w, h)
	bounds := image.Rect(0, 0, w, h)
	padding := m.ContextPadding
	if padding <= 0 {
		padding = 64
	}
	var windows []image.Rectangle
	for _, component := range components {
		r := component.bounds
		r = expand(r, padding, w, h).Intersect(bounds)
		if r.Empty() {
			continue
		}
		parts := []image.Rectangle{r}
		if r.Dx() > 3*r.Dy() || r.Dy() > 3*r.Dx() {
			parts = split(r)
		}
		windows = append(windows, parts...)
	}
	windows = mergeOverlapping(windows)
	type repairWindow struct{ read, write image.Rectangle }
	jobs := make([]repairWindow, 0, len(windows))
	for _, win := range windows {
		jobs = append(jobs, repairWindow{win, win})
	}
	if m.TileSize > 0 {
		jobs = nil
		// TileSize is the source-pixel core. Padding supplies overlapping
		// context; only the core is written, preventing repeated seam blending.
		for _, component := range components {
			win := component.bounds
			for y := win.Min.Y; y < win.Max.Y; y += m.TileSize {
				for x := win.Min.X; x < win.Max.X; x += m.TileSize {
					core := image.Rect(x, y, min(x+m.TileSize, win.Max.X), min(y+m.TileSize, win.Max.Y))
					jobs = append(jobs, repairWindow{expand(core, padding, w, h), core})
				}
			}
		}
	}
	for _, job := range jobs {
		win := job.read
		win = win.Intersect(bounds)
		if win.Empty() {
			continue
		}
		perf.Count(ctx, "inpaint_windows", 1)
		if bg, ok := solidBackground(rgba, dilated, w, win); ok {
			perf.Count(ctx, "inpaint_fills", 1)
			fillMasked(rgba, dilated, w, job.write, bg)
			continue
		}
		perf.Count(ctx, "inpaint_model_calls", 1)
		end := perf.ContextSpan(ctx, "nested", "inpaint_window")
		e := m.inpaintWindow(ctx, rgba, dilated, edges, w, win, job.write)
		end(e)
		if err := e; err != nil {
			return nil, err
		}
	}
	return rgba, nil
}

func (m *Model) inpaintWindow(ctx context.Context, src *image.RGBA, mask, edges []uint8, w int, r image.Rectangle, writeBounds ...image.Rectangle) error {
	if src == nil || w <= 0 || len(mask) < w {
		return fmt.Errorf("inpaint: invalid window %v", r)
	}
	h := len(mask) / w
	r = r.Intersect(src.Bounds()).Intersect(image.Rect(0, 0, w, h))
	if r.Empty() {
		return nil
	}
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
	endModel := perf.ContextSpan(ctx, "nested", "inpaint_model")
	err := m.sess.Run()
	endModel(err)
	if err != nil {
		return fmt.Errorf("run inpaint model for window %v: %w", r, err)
	}
	od := m.out.GetData()
	write := r
	if len(writeBounds) > 0 {
		write = write.Intersect(writeBounds[0])
	}
	for y := write.Min.Y; y < write.Max.Y; y++ {
		py := min(side-1, (y-r.Min.Y)*side/r.Dy())
		for x := write.Min.X; x < write.Max.X; x++ {
			px := min(side-1, (x-r.Min.X)*side/r.Dx())
			i := y*w + x
			if mask[i] == 0 {
				continue
			}
			edge := edges[i]
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
	regions := maskComponents(a, w, h)
	out := make([]image.Rectangle, len(regions))
	for i, c := range regions {
		out[i] = c.bounds
	}
	return out
}

type maskComponent struct {
	bounds image.Rectangle
	pixels []int
}

func maskComponents(a []uint8, w, h int) []maskComponent {
	seen := make([]bool, len(a))
	var out []maskComponent
	for i, v := range a {
		if v == 0 || seen[i] {
			continue
		}
		q := []int{i}
		seen[i] = true
		x0, x1, y0, y1 := i%w, i%w, i/w, i/w
		var pixels []int
		for len(q) > 0 {
			n := q[len(q)-1]
			q = q[:len(q)-1]
			pixels = append(pixels, n)
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
		out = append(out, maskComponent{bounds: image.Rect(x0, y0, x1+1, y1+1), pixels: pixels})
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
			o = append(o, image.Rect(r.Min.X, y, r.Max.X, z))
			y = z
		}
	}
	return o
}

// mergeOverlapping joins only intersecting windows, and only when their union
// still fits a single model patch. Adjacent split strips remain independent.
func mergeOverlapping(in []image.Rectangle) []image.Rectangle {
	out := append([]image.Rectangle(nil), in...)
	for changed := true; changed; {
		changed = false
		for i := 0; i < len(out) && !changed; i++ {
			for j := i + 1; j < len(out); j++ {
				if !out[i].Overlaps(out[j]) {
					continue
				}
				u := out[i].Union(out[j])
				if u.Dx() > side || u.Dy() > side {
					continue
				}
				out[i] = u
				out = append(out[:j], out[j+1:]...)
				changed = true
				break
			}
		}
	}
	return out
}

// solidBackground reports whether the unmasked pixels in the window are close
// to uniform. Manga bubbles and captions are usually flat white or black, so a
// small variance lets us skip the model entirely. The threshold is generous:
// screen-tone dots produce local variance, but they also defeat LaMa, so those
// windows should go to the model anyway. Variance is computed on 0..255 gray.
func solidBackground(im *image.RGBA, mask []uint8, w int, r image.Rectangle) (color.RGBA, bool) {
	bg, _, ok := backgroundColor(im, mask, w, r, 16, 8)
	return bg, ok
}

func localSolidBackground(im *image.RGBA, mask []uint8, w int, r image.Rectangle) (color.RGBA, bool) {
	bg, samples, ok := backgroundColor(im, mask, w, r, 4, 5)
	if !ok || samples < 64 || samples*4 < r.Dx()*r.Dy() {
		return color.RGBA{}, false
	}
	nearWhite := bg.R >= 223 && bg.G >= 223 && bg.B >= 223
	nearBlack := bg.R <= 32 && bg.G <= 32 && bg.B <= 32
	if !nearWhite && !nearBlack {
		return color.RGBA{}, false
	}
	return bg, true
}

func backgroundColor(im *image.RGBA, mask []uint8, w int, r image.Rectangle, maxVariance, maxChannelSpread float64) (color.RGBA, int, bool) {
	if im == nil || w <= 0 || len(mask) < w {
		return color.RGBA{}, 0, false
	}
	h := len(mask) / w
	r = r.Intersect(im.Bounds()).Intersect(image.Rect(0, 0, w, h))
	if r.Empty() {
		return color.RGBA{}, 0, false
	}
	var sums, squares [3]uint64
	n := 0
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			i := y*w + x
			if i < 0 || i >= len(mask) || mask[i] > 0 {
				continue
			}
			c := im.RGBAAt(x, y)
			for k, v := range []uint8{c.R, c.G, c.B} {
				sums[k] += uint64(v)
				squares[k] += uint64(v) * uint64(v)
			}
			n++
		}
	}
	if n == 0 {
		return color.RGBA{}, 0, false
	}
	means := [3]float64{}
	var out [3]uint8
	for k := range means {
		means[k] = float64(sums[k]) / float64(n)
		variance := float64(squares[k])/float64(n) - means[k]*means[k]
		// Require genuinely flat, near-neutral surroundings; this errs toward
		// sending illustrations and colored art to the inpainting model.
		if variance > maxVariance {
			return color.RGBA{}, n, false
		}
		out[k] = uint8(means[k] + 0.5)
	}
	if math.Max(math.Max(means[0], means[1]), means[2])-math.Min(math.Min(means[0], means[1]), means[2]) > maxChannelSpread {
		return color.RGBA{}, n, false
	}
	return color.RGBA{R: out[0], G: out[1], B: out[2], A: 255}, n, true
}
func fillMasked(im *image.RGBA, m []uint8, w int, r image.Rectangle, c color.RGBA) {
	if im == nil || w <= 0 || len(m) < w {
		return
	}
	h := len(m) / w
	r = r.Intersect(im.Bounds()).Intersect(image.Rect(0, 0, w, h))
	if r.Empty() {
		return
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			i := y*w + x
			if i >= 0 && i < len(m) && m[i] > 0 {
				im.SetRGBA(x, y, c)
			}
		}
	}
}

func fillComponent(im *image.RGBA, componentMask []uint8, w int, component maskComponent, c color.RGBA) {
	for _, i := range component.pixels {
		im.SetRGBA(i%w, i/w, c)
		// Clear only pixels belonging to this connected component.
		componentMask[i] = 0
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
			if xx < 0 || yy < 0 || xx >= w || yy*w+xx >= len(m) || m[yy*w+xx] == 0 {
				cd := max(abs(dx), abs(dy))
				if cd < d {
					d = cd
				}
			}
		}
	}
	return d
}
func edgeDistances(mask []uint8, w, h int) []uint8 {
	out := make([]uint8, len(mask))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask[y*w+x] > 0 {
				out[y*w+x] = uint8(edgeDistance(mask, w, x, y))
			}
		}
	}
	return out
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
