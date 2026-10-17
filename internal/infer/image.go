package infer

import (
	"fmt"
	"image"
	"image/color"
	"math"

	"golang.org/x/image/draw"
)

// TensorLayout describes how an image maps onto a model input.
type TensorLayout struct {
	// Side is the square side length the image is padded to.
	Side int
	// ScaledW and ScaledH are the resized image dimensions before padding.
	ScaledW, ScaledH int
	// Mean and Std are applied per channel after scaling to [0,1].
	Mean, Std [3]float32
}

// Letterbox resizes an image so its long side is side, keeps the aspect ratio,
// and pads the rest with zeros at the top-left corner. It returns the NCHW
// float32 buffer plus the layout needed to map coordinates back.
func Letterbox(src image.Image, side int, mean, std [3]float32) ([]float32, TensorLayout) {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()

	sw, sh := side, side
	if w >= h {
		sh = side * h / w
	} else {
		sw = side * w / h
	}
	if sw < 1 {
		sw = 1
	}
	if sh < 1 {
		sh = 1
	}

	// Resize with a decent filter; models are trained on smooth resizes.
	rgba := image.NewRGBA(image.Rect(0, 0, sw, sh))
	draw.CatmullRom.Scale(rgba, rgba.Bounds(), src, b, draw.Src, nil)

	buf := make([]float32, 3*side*side)
	plane := side * side
	for y := 0; y < sh; y++ {
		for x := 0; x < sw; x++ {
			c := rgba.RGBAAt(x, y)
			i := y*side + x
			buf[i] = (float32(c.R)/255 - mean[0]) / std[0]
			buf[plane+i] = (float32(c.G)/255 - mean[1]) / std[1]
			buf[2*plane+i] = (float32(c.B)/255 - mean[2]) / std[2]
		}
	}

	return buf, TensorLayout{Side: side, ScaledW: sw, ScaledH: sh, Mean: mean, Std: std}
}

// ToOriginal maps a point in the padded square back to original image pixels.
func (l TensorLayout) ToOriginal(x, y float32, origW, origH int) (float32, float32) {
	rx := float32(origW) / float32(l.ScaledW)
	ry := float32(origH) / float32(l.ScaledH)
	return x * rx, y * ry
}

// Rect is an axis-aligned box in image pixels.
type Rect struct {
	X, Y, W, H float32
}

func (r Rect) Area() float32 {
	if r.W <= 0 || r.H <= 0 {
		return 0
	}
	return r.W * r.H
}

func (r Rect) Clamp(w, h int) Rect {
	x0 := clampf(r.X, 0, float32(w))
	y0 := clampf(r.Y, 0, float32(h))
	x1 := clampf(r.X+r.W, 0, float32(w))
	y1 := clampf(r.Y+r.H, 0, float32(h))
	return Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}
}

// IOU returns the intersection over union of two rectangles.
func IOU(a, b Rect) float32 {
	x0 := maxf(a.X, b.X)
	y0 := maxf(a.Y, b.Y)
	x1 := minf(a.X+a.W, b.X+b.W)
	y1 := minf(a.Y+a.H, b.Y+b.H)
	if x1 <= x0 || y1 <= y0 {
		return 0
	}
	inter := (x1 - x0) * (y1 - y0)
	union := a.Area() + b.Area() - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// NMS keeps the highest-scoring box of each overlapping group.
func NMS(boxes []Rect, scores []float32, thresh float32) []int {
	n := len(boxes)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	// insertion sort by score descending; n is small
	for i := 1; i < n; i++ {
		for j := i; j > 0 && scores[order[j]] > scores[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}

	keep := make([]int, 0, n)
	suppressed := make([]bool, n)
	for _, i := range order {
		if suppressed[i] {
			continue
		}
		keep = append(keep, i)
		for _, j := range order {
			if j == i || suppressed[j] {
				continue
			}
			if IOU(boxes[i], boxes[j]) > thresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

// GrayToRGB converts a single-channel float mask to an 8-bit gray image.
func GrayToRGB(mask []float32, w, h int, scale float32) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i := 0; i < w*h && i < len(mask); i++ {
		v := mask[i] * scale
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		img.Pix[i] = uint8(v)
	}
	return img
}

// RGBAToFloats converts an image to a 0..1 NCHW float buffer.
func RGBAToFloats(src image.Image, w, h int) []float32 {
	plane := w * h
	buf := make([]float32, 3*plane)
	b := src.Bounds()
	for y := 0; y < h; y++ {
		sy := b.Min.Y + y*b.Dy()/h
		for x := 0; x < w; x++ {
			sx := b.Min.X + x*b.Dx()/w
			c := color.RGBAModel.Convert(src.At(sx, sy)).(color.RGBA)
			i := y*w + x
			buf[i] = float32(c.R) / 255
			buf[plane+i] = float32(c.G) / 255
			buf[2*plane+i] = float32(c.B) / 255
		}
	}
	return buf
}

func clampf(v, lo, hi float32) float32 {
	return float32(math.Min(float64(hi), math.Max(float64(lo), float64(v))))
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func dimsString(d []int64) string {
	return fmt.Sprint(d)
}
