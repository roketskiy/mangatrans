// Package dbtext runs the PP-OCRv6_manga DB text-line detector.
//
// The model is fully convolutional with dynamic input, exported by PaddleOCR.
// We letterbox each page into a fixed square so one session and one pair of
// tensors serve every page shape. Outputs are one probability map; text lines
// are connected components of that map, expanded by the DB unclip rule.
package dbtext

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"

	"mangatrans/internal/infer"
)

// Side is the fixed square input side. The reference implementation resizes
// the long edge to 960, so a 960 square keeps the same glyph scale.
const Side = 960

// padValue is the letterbox fill, a neutral gray. Zero padding shifts the
// activation near the border; gray matches the paper tone better.
const padValue = 114.0 / 255.0

var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// Params mirrors the reference postprocessing constants. BoxThresh is the
// user-facing detection confidence; the rest are model-internal.
type Params struct {
	Thresh     float32 // probability binarization
	BoxThresh  float32 // minimum mean probability inside a line
	Unclip     float32 // DB kernel expansion ratio
	MaskThresh float32 // probability floor for the erasure mask
	MinSide    float32 // minimum line side in input pixels
}

// DefaultParams matches the PP-OCRv6_manga reference (thresh 0.15,
// box_thresh 0.25, unclip 1.4) plus the mask floor and size guard the engine
// needs.
func DefaultParams() Params {
	return Params{Thresh: 0.15, BoxThresh: 0.25, Unclip: 1.4, MaskThresh: 0.25, MinSide: 3}
}

// Model owns one ONNX session and its fixed tensors.
type Model struct {
	sess *ort.AdvancedSession
	in   *ort.Tensor[float32]
	out  *ort.Tensor[float32]
}

// Result is one page's detection output.
type Result struct {
	Boxes  []infer.Rect
	Scores []float32
	// Mask is the page-sized text mask used for region ownership, font-size
	// estimation and erasure. 0 means background.
	Mask *image.Gray
}

// Load builds the session with fixed 960x960 tensors.
func Load(path string, cuda bool, vram int64) (*Model, error) {
	opts, err := infer.CUDAOptions(cuda, vram)
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, Side, Side))
	if err != nil {
		return nil, err
	}
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, Side, Side))
	if err != nil {
		in.Destroy()
		return nil, err
	}
	sess, err := ort.NewAdvancedSession(path,
		[]string{"x"}, []string{"fetch_name_0"},
		[]ort.Value{in}, []ort.Value{out}, opts)
	if err != nil {
		in.Destroy()
		out.Destroy()
		return nil, fmt.Errorf("create dbtext session: %w", err)
	}
	return &Model{sess: sess, in: in, out: out}, nil
}

// Close releases the tensors and the session.
func (m *Model) Close() error {
	m.in.Destroy()
	m.out.Destroy()
	return m.sess.Destroy()
}

// Run detects text lines on one page.
func (m *Model) Run(img image.Image, p Params) (*Result, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW <= 0 || origH <= 0 {
		return &Result{Mask: image.NewGray(b)}, nil
	}
	scale := math.Min(float64(Side)/float64(origW), float64(Side)/float64(origH))
	sw := max(1, int(math.Round(float64(origW)*scale)))
	sh := max(1, int(math.Round(float64(origH)*scale)))
	dx := (Side - sw) / 2
	dy := (Side - sh) / 2

	fillNormalized(m.in.GetData())
	rgba := image.NewRGBA(image.Rect(0, 0, sw, sh))
	draw.CatmullRom.Scale(rgba, rgba.Bounds(), img, b, draw.Src, nil)
	writePlanes(m.in.GetData(), rgba, dx, dy)

	if err := m.sess.Run(); err != nil {
		return nil, fmt.Errorf("dbtext run: %w", err)
	}

	prob := m.out.GetData()
	res := &Result{
		Boxes: postprocess(prob, p, dx, dy, sw, sh, origW, origH),
	}
	// Erase by filled line boxes, not by the probability map: the map is a
	// strip-shaped core that misses stroke edges, and a half-erased glyph is
	// worse than extra white the inpainter refills anyway.
	mask := image.NewGray(b)
	for _, bx := range res.Boxes {
		r := image.Rect(int(bx.X)-2, int(bx.Y)-2, int(bx.X+bx.W)+2, int(bx.Y+bx.H)+2).Intersect(b)
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				mask.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	dilate(mask, 2)
	res.Mask = mask
	res.Scores = make([]float32, len(res.Boxes))
	for i := range res.Scores {
		res.Scores[i] = 1
	}
	sortBoxes(res.Boxes)
	return res, nil
}

func fillNormalized(buf []float32) {
	plane := Side * Side
	for i := 0; i < plane; i++ {
		buf[i] = (padValue - imagenetMean[0]) / imagenetStd[0]
		buf[plane+i] = (padValue - imagenetMean[1]) / imagenetStd[1]
		buf[2*plane+i] = (padValue - imagenetMean[2]) / imagenetStd[2]
	}
}

func writePlanes(buf []float32, src *image.RGBA, dx, dy int) {
	plane := Side * Side
	for y := 0; y < src.Bounds().Dy(); y++ {
		row := (dy + y) * Side
		for x := 0; x < src.Bounds().Dx(); x++ {
			c := src.RGBAAt(x, y)
			i := row + dx + x
			buf[i] = (float32(c.R)/255 - imagenetMean[0]) / imagenetStd[0]
			buf[plane+i] = (float32(c.G)/255 - imagenetMean[1]) / imagenetStd[1]
			buf[2*plane+i] = (float32(c.B)/255 - imagenetMean[2]) / imagenetStd[2]
		}
	}
}

type component struct {
	minX, minY, maxX, maxY int
	area                   int
	sum                    float64
}

// postprocess labels the probability map, filters weak lines, expands each by
// the DB unclip offset and maps it to page coordinates.
func postprocess(prob []float32, p Params, dx, dy, sw, sh, origW, origH int) []infer.Rect {
	n := Side * Side
	bin := make([]bool, n)
	for i := 0; i < n; i++ {
		bin[i] = prob[i] >= p.Thresh
	}
	visited := make([]bool, n)
	queue := make([]int32, 0, 4096)
	var out []infer.Rect

	for start := 0; start < n; start++ {
		if !bin[start] || visited[start] {
			continue
		}
		queue = queue[:0]
		queue = append(queue, int32(start))
		visited[start] = true
		comp := component{minX: Side, minY: Side, maxX: -1, maxY: -1}
		for head := 0; head < len(queue); head++ {
			i := int(queue[head])
			x, y := i%Side, i/Side
			comp.area++
			comp.sum += float64(prob[i])
			if x < comp.minX {
				comp.minX = x
			}
			if x > comp.maxX {
				comp.maxX = x
			}
			if y < comp.minY {
				comp.minY = y
			}
			if y > comp.maxY {
				comp.maxY = y
			}
			for oy := -1; oy <= 1; oy++ {
				ny := y + oy
				if ny < 0 || ny >= Side {
					continue
				}
				for ox := -1; ox <= 1; ox++ {
					nx := x + ox
					if nx < 0 || nx >= Side {
						continue
					}
					j := ny*Side + nx
					if bin[j] && !visited[j] {
						visited[j] = true
						queue = append(queue, int32(j))
					}
				}
			}
		}
		score := float32(comp.sum / float64(comp.area))
		if score < p.BoxThresh {
			continue
		}
		w := float32(comp.maxX - comp.minX + 1)
		h := float32(comp.maxY - comp.minY + 1)
		if w < p.MinSide || h < p.MinSide {
			continue
		}
		// DB unclip: constant offset derived from area and perimeter.
		offset := 0.0
		if perimeter := 2 * (w + h); perimeter > 0 {
			offset = float64(float32(comp.area)*p.Unclip) / float64(perimeter)
		}
		x0 := (float32(comp.minX) - float32(offset) - float32(dx)) * float32(origW) / float32(sw)
		y0 := (float32(comp.minY) - float32(offset) - float32(dy)) * float32(origH) / float32(sh)
		x1 := (float32(comp.maxX+1) + float32(offset) - float32(dx)) * float32(origW) / float32(sw)
		y1 := (float32(comp.maxY+1) + float32(offset) - float32(dy)) * float32(origH) / float32(sh)
		r := infer.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}.Clamp(origW, origH)
		if r.Area() <= 0 {
			continue
		}
		out = append(out, r)
	}
	return out
}

// buildMask samples the probability map back to page pixels and dilates the
// surviving strokes. The detector's map is thinner than a real glyph; the
// dilation restores glyph coverage for erasing and gap-based splitting.
func buildMask(prob []float32, thresh float32, dx, dy, sw, sh, origW, origH int, b image.Rectangle) *image.Gray {
	mask := image.NewGray(b)
	invX := float32(sw) / float32(origW)
	invY := float32(sh) / float32(origH)
	for y := 0; y < origH; y++ {
		iy := dy + int(float32(y)*invY)
		if iy < 0 || iy >= Side {
			continue
		}
		row := iy * Side
		for x := 0; x < origW; x++ {
			ix := dx + int(float32(x)*invX)
			if ix < 0 || ix >= Side {
				continue
			}
			if prob[row+ix] >= thresh {
				mask.SetGray(b.Min.X+x, b.Min.Y+y, color.Gray{Y: 255})
			}
		}
	}
	dilate(mask, 4)
	return mask
}

// dilate applies a separable square max filter of the given radius.
func dilate(m *image.Gray, radius int) {
	if radius <= 0 {
		return
	}
	b := m.Bounds()
	w, h := b.Dx(), b.Dy()
	step := m.Stride
	row := make([]uint8, w*h)
	copy(row, m.Pix[:w*h])
	// horizontal pass
	for y := 0; y < h; y++ {
		base := y * step
		for x := 0; x < w; x++ {
			var v uint8
			x0, x1 := max(0, x-radius), min(w-1, x+radius)
			for k := x0; k <= x1; k++ {
				if row[base+k] > v {
					v = row[base+k]
				}
			}
			m.Pix[base+x] = v
		}
	}
	// vertical pass
	col := make([]uint8, h)
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			col[y] = m.Pix[y*step+x]
		}
		for y := 0; y < h; y++ {
			var v uint8
			y0, y1 := max(0, y-radius), min(h-1, y+radius)
			for k := y0; k <= y1; k++ {
				if col[k] > v {
					v = col[k]
				}
			}
			m.Pix[y*step+x] = v
		}
	}
}

// sortBoxes orders lines for reading: top to bottom, right to left inside a
// row, matching Japanese column order.
func sortBoxes(boxes []infer.Rect) {
	sort.SliceStable(boxes, func(i, j int) bool {
		a, b := boxes[i], boxes[j]
		if math.Abs(float64(a.Y-b.Y)) > 20 {
			return a.Y < b.Y
		}
		return a.X > b.X
	})
}
