package cbz

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNaturalLess(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"page2.png", "page10.png", true},
		{"page10.png", "page2.png", false},
		{"001.png", "002.png", true},
		{"img_9.jpg", "img_10.jpg", true},
		{"a.png", "a.png", false},
	}
	for _, c := range cases {
		if got := naturalLess(c.a, c.b); got != c.want {
			t.Errorf("naturalLess(%q,%q)=%v want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestReadWriteRoundTrip(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "in.cbz")
	pages := []Page{
		{Name: "p2.png", Bytes: []byte("second")},
		{Name: "p10.png", Bytes: []byte("tenth")},
		{Name: "p1.png", Bytes: []byte("first")},
	}
	if err := WriteCBZ(src, pages); err != nil {
		t.Fatal(err)
	}

	got, err := ReadCBZ(src)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d pages", len(got))
	}
	want := []string{"p1.png", "p2.png", "p10.png"}
	for i, w := range want {
		if got[i].Name != w {
			t.Errorf("page %d = %q want %q", i, got[i].Name, w)
		}
	}

	dst := filepath.Join(dir, "out.cbz")
	if err := WriteCBZ(dst, got); err != nil {
		t.Fatal(err)
	}
	if st, err := os.Stat(dst); err != nil || st.Size() == 0 {
		t.Fatalf("output not written: %v", err)
	}
}

func TestReadCBZMissing(t *testing.T) {
	if _, err := ReadCBZ(filepath.Join(t.TempDir(), "nope.cbz")); err == nil {
		t.Fatal("expected error for missing file")
	}
}
