package detect

import (
	"fmt"
	"image"
	"math"
	"sort"

	ort "github.com/yalue/onnxruntime_go"

	"mangatrans/internal/infer"
)

// Model runs ComicTextDetector, which emits YOLO boxes, a UNet text mask and a
// DBNet shrink/threshold map from a single forward pass.
type Model struct {
	sess   *ort.AdvancedSession
	in     *ort.Tensor[float32]
	blk    *ort.Tensor[float32]
	seg    *ort.Tensor[float32]
	det    *ort.Tensor[float32]
	side   int
	layout infer.TensorLayout
}

// Params mirror the reference implementation's postprocessing constants.
type Params struct {
	ConfThreshold  float32
	NMSThreshold   float32
	BoxDilation    float32
	MaskThreshold  uint8
	DilateRadius   int
	CloseRadius    int
	FinalDilation  int
	MinBoxSide     float32
	MinBoxArea     float32
	Side           int
}

// DefaultParams matches koharu-slim's comic_text_detector settings.
func DefaultParams() Params {
	return Params{
		ConfThreshold: 0.4,
		NMSThreshold:  0.35,
		BoxDilation:   1.0,
		MaskThreshold: 60,
		DilateRadius:  3,
		CloseRadius:   10,
		FinalDilation: 2,
		MinBoxSide:    6,
		MinBoxArea:    48,
		Side:          1024,
	}
}

// Load builds the session. Input is a fixed 1x3x1024x1024 tensor.
func Load(path string, cuda bool, vram int64) (*Model, error) {
	opts, err := infer.CUDAOptions(cuda, vram)
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	const side = 1024
	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, side, side))
	if err != nil {
		return nil, err
	}
	blk, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 64512, 7))
	if err != nil {
		in.Destroy()
		return nil, err
	}
	seg, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, side, side))
	if err != nil {
		in.Destroy()
		blk.Destroy()
		return nil, err
	}
	det, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 2, side, side))
	if err != nil {
		in.Destroy()
		blk.Destroy()
		seg.Destroy()
		return nil, err
	}

	sess, err := ort.NewAdvancedSession(path,
		[]string{"images"}, []string{"blk", "seg", "det"},
		[]ort.Value{in}, []ort.Value{blk, seg, det}, opts)
	if err != nil {
		in.Destroy()
		blk.Destroy()
		seg.Destroy()
		det.Destroy()
		return nil, fmt.Errorf("create detector session: %w", err)
	}

	return &Model{sess: sess, in: in, blk: blk, seg: seg, det: det, side: side}, nil
}

// Close releases all tensors and the session.
func (m *Model) Close() error {
	m.in.Destroy()
	m.blk.Destroy()
	m.seg.Destroy()
	m.det.Destroy()
	return m.sess.Destroy()
}

// Result is one page's detection output.
type Result struct {
	Boxes  []infer.Rect
	Scores []float32
	// Mask is a full-resolution grayscale text mask, 0 means keep.
	Mask *image.Gray
	// DetMap is the DBNet probability map at input resolution.
	DetMap []float32
	// Layout maps padded-square coordinates back to the page.
	Layout infer.TensorLayout
}

// Run detects text regions on one page.
func (m *Model) Run(img image.Image, p Params) (*Result, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	buf, layout := infer.Letterbox(img, m.side, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
	copy(m.in.GetData(), buf)
	m.layout = layout

	if err := m.sess.Run(); err != nil {
		return nil, fmt.Errorf("detector run: %w", err)
	}

	boxes, scores := m.postprocessYOLO(origW, origH, layout, p)
	mask := m.postprocessMask(origW, origH, layout, p)

	detMap := make([]float32, m.side*m.side)
	detData := m.det.GetData()
	// channel 0 is the shrink map
	copy(detMap, detData[:m.side*m.side])

	return &Result{
		Boxes:  boxes,
		Scores: scores,
		Mask:   mask,
		DetMap: detMap,
		Layout: layout,
	}, nil
}

func (m *Model) postprocessYOLO(origW, origH int, layout infer.TensorLayout, p Params) ([]infer.Rect, []float32) {
	data := m.blk.GetData()
	// shape is [1, 64512, 7]: 4 box + 1 objectness + 2 class scores
	const n, stride = 64512, 7

	rx := float32(origW) / float32(layout.ScaledW)
	ry := float32(origH) / float32(layout.ScaledH)

	var boxes []infer.Rect
	var scores []float32
	for i := 0; i < n; i++ {
		base := i * stride
		obj := data[base+4]
		if obj < p.ConfThreshold*0.5 {
			continue
		}
		// pick the best class score
		best := data[base+5]
		if data[base+6] > best {
			best = data[base+6]
		}
		conf := obj * best
		if conf < p.ConfThreshold {
			continue
		}

		cx, cy := data[base], data[base+1]
		w, h := data[base+2], data[base+3]
		x0 := (cx-w/2)*rx - p.BoxDilation
		x1 := (cx+w/2)*rx + p.BoxDilation
		y0 := (cy-h/2)*ry - p.BoxDilation
		y1 := (cy+h/2)*ry + p.BoxDilation

		r := infer.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}.Clamp(origW, origH)
		if r.W < p.MinBoxSide || r.H < p.MinBoxSide || r.Area() < p.MinBoxArea {
			continue
		}
		boxes = append(boxes, r)
		scores = append(scores, conf)
	}

	keep := infer.NMS(boxes, scores, p.NMSThreshold)
	outBoxes := make([]infer.Rect, 0, len(keep))
	outScores := make([]float32, 0, len(keep))
	for _, i := range keep {
		outBoxes = append(outBoxes, boxes[i])
		outScores = append(outScores, scores[i])
	}

	// Reading order: top to bottom, then right to left for manga.
	sort.SliceStable(outBoxes, func(i, j int) bool {
		dy := outBoxes[i].Y - outBoxes[j].Y
		if math.Abs(float64(dy)) > 20 {
			return dy < 0
		}
		return outBoxes[i].X > outBoxes[j].X
	})
	return outBoxes, outScores
}

func (m *Model) postprocessMask(origW, origH int, layout infer.TensorLayout, p Params) *image.Gray {
	seg := m.seg.GetData()
	const side = 1024

	// Binarize at the input resolution, then scale to the page.
	bin := make([]uint8, side*side)
	for i := 0; i < side*side; i++ {
		v := seg[i] * 255
		if v > float32(p.MaskThreshold) {
			bin[i] = 255
		}
	}

	bin = dilate(bin, side, side, p.DilateRadius)
	bin = erode(bin, side, side, p.CloseRadius)
	bin = dilate(bin, side, side, p.CloseRadius)
	bin = dilate(bin, side, side, p.FinalDilation)

	// Nearest-neighbour downscale of the padded square to the page size.
	out := image.NewGray(image.Rect(0, 0, origW, origH))
	sx := float32(layout.ScaledW) / float32(origW)
	sy := float32(layout.ScaledH) / float32(origH)
	for y := 0; y < origH; y++ {
		iy := int(float32(y) * sy)
		if iy >= side {
			iy = side - 1
		}
		for x := 0; x < origW; x++ {
			ix := int(float32(x) * sx)
			if ix >= side {
				ix = side - 1
			}
			out.Pix[y*out.Stride+x] = bin[iy*side+ix]
		}
	}
	return out
}

// dilate/erode use separable passes. A square structuring element factors into
// a row pass and a column pass, which turns O(r²) per pixel into O(2r).
func dilate(src []uint8, w, h, radius int) []uint8 {
	if radius <= 0 {
		return src
	}
	tmp := dilateRows(src, w, h, radius)
	return dilateRowsT(tmp, w, h, radius)
}

func erode(src []uint8, w, h, radius int) []uint8 {
	if radius <= 0 {
		return src
	}
	tmp := erodeRows(src, w, h, radius)
	return erodeRowsT(tmp, w, h, radius)
}

func dilateRows(src []uint8, w, h, r int) []uint8 {
	dst := make([]uint8, len(src))
	for y := 0; y < h; y++ {
		row := src[y*w : (y+1)*w]
		out := dst[y*w : (y+1)*w]
		for x := 0; x < w; x++ {
			lo := x - r
			if lo < 0 {
				lo = 0
			}
			hi := x + r
			if hi >= w {
				hi = w - 1
			}
			var v uint8
			for i := lo; i <= hi; i++ {
				if row[i] != 0 {
					v = 255
					break
				}
			}
			out[x] = v
		}
	}
	return dst
}

// dilateRowsT dilates along columns by scanning the transposed image.
func dilateRowsT(src []uint8, w, h, r int) []uint8 {
	dst := make([]uint8, len(src))
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			lo := y - r
			if lo < 0 {
				lo = 0
			}
			hi := y + r
			if hi >= h {
				hi = h - 1
			}
			var v uint8
			for i := lo; i <= hi; i++ {
				if src[i*w+x] != 0 {
					v = 255
					break
				}
			}
			dst[y*w+x] = v
		}
	}
	return dst
}

func erodeRows(src []uint8, w, h, r int) []uint8 {
	dst := make([]uint8, len(src))
	for y := 0; y < h; y++ {
		row := src[y*w : (y+1)*w]
		out := dst[y*w : (y+1)*w]
		for x := 0; x < w; x++ {
			lo := x - r
			if lo < 0 {
				lo = 0
			}
			hi := x + r
			if hi >= w {
				hi = w - 1
			}
			v := uint8(255)
			for i := lo; i <= hi; i++ {
				if row[i] == 0 {
					v = 0
					break
				}
			}
			out[x] = v
		}
	}
	return dst
}

func erodeRowsT(src []uint8, w, h, r int) []uint8 {
	dst := make([]uint8, len(src))
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			lo := y - r
			if lo < 0 {
				lo = 0
			}
			hi := y + r
			if hi >= h {
				hi = h - 1
			}
			v := uint8(255)
			for i := lo; i <= hi; i++ {
				if src[i*w+x] == 0 {
					v = 0
					break
				}
			}
			dst[y*w+x] = v
		}
	}
	return dst
}
