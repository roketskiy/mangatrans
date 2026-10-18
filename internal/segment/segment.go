package segment

import (
	"fmt"
	"image"
	"image/color"
	"math"

	ort "github.com/yalue/onnxruntime_go"

	"mangatrans/internal/infer"
)

// Params controls bubble-segmentation inference and postprocessing.
type Params struct {
	// ConfThreshold is the instance confidence floor.
	ConfThreshold float32
	// IOUThreshold feeds the box NMS.
	IOUThreshold float32
	// MaskThreshold is the sigmoid cutoff on the prototype activation.
	MaskThreshold float32
	// Side is the square input side the model was exported with.
	Side int
}

// DefaultParams matches MangaLens' ultralytics defaults at its 1600 training size.
func DefaultParams() Params {
	return Params{ConfThreshold: 0.25, IOUThreshold: 0.7, MaskThreshold: 0.5, Side: 1600}
}

// Bubble is one segmented speech balloon in absolute page coordinates.
type Bubble struct {
	Box   image.Rectangle
	Score float32
	// Mask carries 255 inside the balloon and 0 elsewhere, sized to the page.
	Mask *image.Alpha
}

// Model runs a YOLO11-seg bubble segmenter exported to ONNX. The graph emits
// one detection row per anchor ([1, 4+nc+protos, N], channel-major) plus the
// shared prototype masks ([1, protos, side/4, side/4]).
type Model struct {
	sess  *ort.AdvancedSession
	in    *ort.Tensor[float32]
	out   *ort.Tensor[float32]
	proto *ort.Tensor[float32]
	side  int
}

// Load builds the session for the given fixed input side.
func Load(path string, cuda bool, vram int64, side int) (*Model, error) {
	opts, err := infer.CUDAOptions(cuda, vram)
	if err != nil {
		return nil, err
	}
	defer opts.Destroy()

	in, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 3, int64(side), int64(side)))
	if err != nil {
		return nil, err
	}
	// 4 box + 1 class + 32 mask coefficients; 52500 anchors over the three
	// strides at side 1600 (200^2 + 100^2 + 50^2, verified by a CPU run).
	// Do not simplify this export: onnxslim folds the anchor reshape wrong.
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 37, 52500))
	if err != nil {
		in.Destroy()
		return nil, err
	}
	proto, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 32, int64(side)/4, int64(side)/4))
	if err != nil {
		in.Destroy()
		out.Destroy()
		return nil, err
	}
	sess, err := ort.NewAdvancedSession(path,
		[]string{"images"}, []string{"output0", "output1"},
		[]ort.Value{in}, []ort.Value{out, proto}, opts)
	if err != nil {
		in.Destroy()
		out.Destroy()
		proto.Destroy()
		return nil, fmt.Errorf("create bubble segmenter session: %w", err)
	}
	return &Model{sess: sess, in: in, out: out, proto: proto, side: side}, nil
}

// Close releases all tensors and the session.
func (m *Model) Close() error {
	m.in.Destroy()
	m.out.Destroy()
	m.proto.Destroy()
	return m.sess.Destroy()
}

// Run segments speech balloons on one page.
func (m *Model) Run(img image.Image, p Params) ([]Bubble, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	buf, layout := infer.Letterbox(img, m.side, [3]float32{0, 0, 0}, [3]float32{1, 1, 1})
	copy(m.in.GetData(), buf)
	if err := m.sess.Run(); err != nil {
		return nil, fmt.Errorf("bubble segmenter run: %w", err)
	}
	return m.postprocess(origW, origH, layout, p), nil
}

type candidate struct {
	rect   infer.Rect
	score  float32
	coeffs [32]float32
}

func (m *Model) postprocess(origW, origH int, layout infer.TensorLayout, p Params) []Bubble {
	data := m.out.GetData()
	n := len(data) / 37

	var cands []candidate
	for i := 0; i < n; i++ {
		conf := data[4*n+i]
		if conf < p.ConfThreshold {
			continue
		}
		cx, cy := data[i], data[n+i]
		w, h := data[2*n+i], data[3*n+i]
		x0, y0 := layout.ToOriginal(cx-w/2, cy-h/2, origW, origH)
		x1, y1 := layout.ToOriginal(cx+w/2, cy+h/2, origW, origH)
		r := infer.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}.Clamp(origW, origH)
		if r.W < 8 || r.H < 8 {
			continue
		}
		c := candidate{rect: r, score: conf}
		for j := 0; j < 32; j++ {
			c.coeffs[j] = data[(5+j)*n+i]
		}
		cands = append(cands, c)
	}
	if len(cands) == 0 {
		return nil
	}

	boxes := make([]infer.Rect, len(cands))
	scores := make([]float32, len(cands))
	for i, c := range cands {
		boxes[i], scores[i] = c.rect, c.score
	}
	keep := infer.NMS(boxes, scores, p.IOUThreshold)

	protoData := m.proto.GetData()
	pSide := m.side / 4
	plane := pSide * pSide
	sx := float32(layout.ScaledW) / float32(origW)
	sy := float32(layout.ScaledH) / float32(origH)

	bubbles := make([]Bubble, 0, len(keep))
	for _, k := range keep {
		c := cands[k]
		act := activation(protoData, 32, plane, c.coeffs)
		mask, box := renderMask(act, pSide, m.side, sx, sy, c.rect, p.MaskThreshold, origW, origH)
		if mask == nil {
			continue
		}
		bubbles = append(bubbles, Bubble{Box: box, Score: c.score, Mask: mask})
	}
	return bubbles
}

// activation folds the 32 prototypes into one side x side score map.
func activation(protoData []float32, channels, plane int, coeffs [32]float32) []float32 {
	act := make([]float32, plane)
	for c := 0; c < channels; c++ {
		w := coeffs[c]
		if w == 0 {
			continue
		}
		base := protoData[c*plane : c*plane+plane]
		for i, v := range base {
			act[i] += w * v
		}
	}
	for i, v := range act {
		act[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	return act
}

// renderMask samples the low-resolution activation over the box (inflated by
// a margin so edge antialiasing survives) and writes 255 where the score
// passes the threshold, in absolute page coordinates.
func renderMask(act []float32, pSide, side int, sx, sy float32, r infer.Rect, thresh float32, origW, origH int) (*image.Alpha, image.Rectangle) {
	const margin = 8
	x0 := max(0, int(r.X)-margin)
	y0 := max(0, int(r.Y)-margin)
	x1 := min(origW, int(r.X+r.W)+margin)
	y1 := min(origH, int(r.Y+r.H)+margin)
	if x0 >= x1 || y0 >= y1 {
		return nil, image.Rectangle{}
	}
	mask := image.NewAlpha(image.Rect(x0, y0, x1, y1))
	for y := y0; y < y1; y++ {
		fy := (float32(y)+0.5)*sy/float32(side)*float32(pSide) - 0.5
		py := int(math.Floor(float64(fy)))
		fy -= float32(py)
		for x := x0; x < x1; x++ {
			fx := (float32(x)+0.5)*sx/float32(side)*float32(pSide) - 0.5
			px := int(math.Floor(float64(fx)))
			fx -= float32(px)
			v := bilinear(act, pSide, px, py, fx, fy)
			if v >= thresh {
				mask.SetAlpha(x, y, color.Alpha{A: 255})
			}
		}
	}
	return mask, image.Rect(x0, y0, x1, y1)
}

func bilinear(act []float32, w, px, py int, fx, fy float32) float32 {
	at := func(x, y int) float32 {
		if x < 0 {
			x = 0
		}
		if y < 0 {
			y = 0
		}
		if x >= w {
			x = w - 1
		}
		if y >= w {
			y = w - 1
		}
		return act[y*w+x]
	}
	top := at(px, py)*(1-fx) + at(px+1, py)*fx
	bottom := at(px, py+1)*(1-fx) + at(px+1, py+1)*fx
	return top*(1-fy) + bottom*fy
}
