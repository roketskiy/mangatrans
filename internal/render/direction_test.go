package render

import "testing"

func TestUniformDirectionIgnoresSourceBoxOrientation(t *testing.T) {
	for _, sourceVertical := range []bool{false, true} {
		for _, tc := range []struct {
			value string
			want  bool
		}{
			{"vertical", true}, {"horizontal", false}, {"auto", sourceVertical},
		} {
			d, err := ParseDirection(tc.value)
			if err != nil || d.IsVertical(sourceVertical) != tc.want {
				t.Fatalf("direction %q, sourceVertical=%v: got %v, err=%v", tc.value, sourceVertical, d.IsVertical(sourceVertical), err)
			}
		}
	}
	if _, err := ParseDirection("sideways"); err == nil {
		t.Fatal("invalid direction accepted")
	}
}
