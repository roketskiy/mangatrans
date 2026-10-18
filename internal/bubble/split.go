package bubble

import (
	"image"
	"image/color"
	"sort"
)

// TextRegion is one independently recognized dialogue. Area constrains final
// placement; CropMask owns source pixels for OCR and erasure, including glyphs
// near the outline. Distinct children never share source pixels.
type TextRegion struct {
	Box      image.Rectangle
	Area     *image.Alpha
	CropMask *image.Alpha
	Parent   int
}

// Crop is the exact image sent to OCR. A rectangular crop may intersect a
// sibling's box; source ownership masks that sibling to white.
func (region TextRegion) Crop(source image.Image, padding int) image.Image {
	b := region.Box.Inset(-padding).Intersect(source.Bounds())
	if b.Empty() {
		return nil
	}
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			var c color.Color = color.White
			if region.CropMask == nil || region.CropMask.AlphaAt(x, y).A == 255 {
				c = source.At(x, y)
			}
			out.Set(x-b.Min.X, y-b.Min.Y, c)
		}
	}
	return out
}

func SplitTextRegions(source image.Image, textMask *image.Gray, boxes []image.Rectangle) []TextRegion {
	areas := Find(source, 210).FitAreas(boxes)
	var out []TextRegion
	for parent, box := range boxes {
		fallback := TextRegion{Box: box, Parent: parent}
		area := areas[parent]
		if area == nil || textMask == nil {
			out = append(out, fallback)
			continue
		}
		parts := SplitArea(area, area.Bounds())
		if len(parts) != 2 {
			out = append(out, fallback)
			continue
		}
		// Grow both interiors through the parent crop, retaining one owner for
		// every source pixel even where glyph strokes interrupted the white area.
		r := box.Inset(-8).Intersect(source.Bounds())
		w := r.Dx()
		owners := make([]uint8, w*r.Dy())
		queue := make([]int, 0, len(owners))
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				for k, part := range parts {
					if part.AlphaAt(x, y).A == 255 {
						i := (y-r.Min.Y)*w + x - r.Min.X
						owners[i] = uint8(k + 1)
						queue = append(queue, i)
						break
					}
				}
			}
		}
		for head := 0; head < len(queue); head++ {
			i := queue[head]
			x, y := i%w, i/w
			for _, p := range []image.Point{{X: x - 1, Y: y}, {X: x + 1, Y: y}, {X: x, Y: y - 1}, {X: x, Y: y + 1}} {
				if p.X < 0 || p.X >= w || p.Y < 0 || p.Y >= r.Dy() {
					continue
				}
				j := p.Y*w + p.X
				if owners[j] == 0 {
					owners[j] = owners[i]
					queue = append(queue, j)
				}
			}
		}
		children := []TextRegion{{Area: parts[0], CropMask: image.NewAlpha(r), Parent: parent}, {Area: parts[1], CropMask: image.NewAlpha(r), Parent: parent}}
		counts := [2]int{}
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				owner := owners[(y-r.Min.Y)*w+x-r.Min.X]
				if owner == 0 {
					continue
				}
				k := int(owner - 1)
				children[k].CropMask.SetAlpha(x, y, color.Alpha{A: 255})
				if image.Pt(x, y).In(box) && textMask.GrayAt(x, y).Y > 0 {
					children[k].Box = children[k].Box.Union(image.Rect(x, y, x+1, y+1))
					counts[k]++
				}
			}
		}
		if counts[0] < 24 || counts[1] < 24 || min(counts[0], counts[1])*10 < counts[0]+counts[1] || children[0].Box.Dx() < 6 || children[1].Box.Dx() < 6 || children[0].Box.Dy() < 6 || children[1].Box.Dy() < 6 {
			out = append(out, fallback)
			continue
		}
		// Japanese dialogue reads right-to-left across columns, or top-to-bottom
		// for vertically stacked lobes. Keep children adjacent in parent order.
		sort.SliceStable(children, func(i, j int) bool {
			a, b := children[i].Box, children[j].Box
			if min(a.Max.Y, b.Max.Y)-max(a.Min.Y, b.Min.Y) > min(a.Dy(), b.Dy())/3 {
				return a.Min.X > b.Min.X
			}
			return a.Min.Y < b.Min.Y
		})
		out = append(out, children...)
	}
	return out
}

// SplitArea removes progressively thicker boundary layers to find two broad
// interiors, then grows them back through the original writable pixels. A
// narrow neck separates the cores before either lobe disappears. Ordinary
// convex bubbles retain one core regardless of orientation.
func SplitArea(mask *image.Alpha, bounds image.Rectangle) []*image.Alpha {
	w, h := bounds.Dx()+2, bounds.Dy()+2
	distance := make([]int, w*h)
	total, peak := 0, 0
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			if mask.AlphaAt(bounds.Min.X+x-1, bounds.Min.Y+y-1).A == 255 {
				distance[i] = w + h
				total++
			}
		}
	}
	if total == 0 {
		return []*image.Alpha{mask}
	}
	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			if distance[i] > 0 {
				distance[i] = min(distance[i], min(distance[i-1], distance[i-w])+1)
			}
		}
	}
	for y := h - 2; y > 0; y-- {
		for x := w - 2; x > 0; x-- {
			i := y*w + x
			if distance[i] > 0 {
				distance[i] = min(distance[i], min(distance[i+1], distance[i+w])+1)
				peak = max(peak, distance[i])
			}
		}
	}
	neighbors := []int{-1, 1, -w, w}
	for percent := 45; percent <= 80; percent += 5 {
		radius := max(2, peak*percent/100)
		seen := make([]bool, len(distance))
		var cores [][]int
		for i, d := range distance {
			if d <= radius || seen[i] {
				continue
			}
			q := []int{i}
			seen[i] = true
			corePeak := d
			for head := 0; head < len(q); head++ {
				for _, step := range neighbors {
					j := q[head] + step
					if !seen[j] && distance[j] > radius {
						seen[j] = true
						q = append(q, j)
						corePeak = max(corePeak, distance[j])
					}
				}
			}
			// Ignore small fragments and cuts so deep that a minor mask defect
			// could have separated the surviving pixels.
			if len(q) >= max(9, total/100) && radius*10 <= corePeak*9 {
				cores = append(cores, q)
			}
		}
		if len(cores) != 2 {
			continue
		}
		owner := make([]uint8, len(distance))
		queue := make([]int, 0, total)
		for k, core := range cores {
			for _, i := range core {
				owner[i] = uint8(k + 1)
				queue = append(queue, i)
			}
		}
		counts := [2]int{len(cores[0]), len(cores[1])}
		for head := 0; head < len(queue); head++ {
			i := queue[head]
			for _, step := range neighbors {
				j := i + step
				if distance[j] > 0 && owner[j] == 0 {
					owner[j] = owner[i]
					counts[owner[i]-1]++
					queue = append(queue, j)
				}
			}
		}
		if counts[0]*100 < total*23 || counts[1]*100 < total*23 || counts[0]+counts[1] != total {
			continue
		}
		parts := []*image.Alpha{image.NewAlpha(bounds), image.NewAlpha(bounds)}
		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				if k := owner[y*w+x]; k > 0 {
					parts[k-1].SetAlpha(bounds.Min.X+x-1, bounds.Min.Y+y-1, color.Alpha{A: 255})
				}
			}
		}
		return parts
	}
	return []*image.Alpha{mask}
}
