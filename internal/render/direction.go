package render

import "fmt"

type Direction string

const (
	Vertical   Direction = "vertical"
	Horizontal Direction = "horizontal"
	Auto       Direction = "auto"
)

func ParseDirection(s string) (Direction, error) {
	d := Direction(s)
	if d != Vertical && d != Horizontal && d != Auto {
		return "", fmt.Errorf("text-direction must be vertical, horizontal or auto, got %q", s)
	}
	return d, nil
}

func (d Direction) IsVertical(sourceVertical bool) bool {
	return d == Vertical || d == Auto && sourceVertical
}
