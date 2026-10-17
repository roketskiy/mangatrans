package translate

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"mangatrans/internal/llm"
)

func TestParseLoose(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    Result
	}{
		{
			name:    "plain json",
			content: `{"blocks":[{"id":1,"text":"甲"},{"id":2,"text":"乙"}]}`,
			want:    Result{1: "甲", 2: "乙"},
		},
		{
			name:    "code fence",
			content: "```json\n{\"blocks\":[{\"id\":1,\"text\":\"甲\"},{\"id\":2,\"text\":\"乙\"}]}\n```",
			want:    Result{1: "甲", 2: "乙"},
		},
		{
			name:    "prose around json",
			content: `好的：{"blocks":[{"id":1,"text":"甲"},{"id":2,"text":""}]}`,
			want:    Result{1: "甲", 2: ""},
		},
		{
			name:    "missing id",
			content: `{"blocks":[{"id":1,"text":"甲"}]}`,
			want:    Result{1: "甲"},
		},
		{
			name:    "no json",
			content: `抱歉我不会`,
			want:    Result{},
		},
		{
			name:    "extra unknown id kept",
			content: `{"blocks":[{"id":1,"text":"甲"},{"id":2,"text":"乙"},{"id":9,"text":"多"}]}`,
			want:    Result{1: "甲", 2: "乙", 9: "多"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := parseLoose(c.content)
			if len(got) != len(c.want) {
				t.Fatalf("got %v want %v", got, c.want)
			}
			for id, want := range c.want {
				if got[id] != want {
					t.Errorf("id %d = %q want %q", id, got[id], want)
				}
			}
		})
	}
}

// TestTranslateBatchLive exercises a real llama-server when the model is
// present. Skipped otherwise so CI stays green.
func TestTranslateBatchLive(t *testing.T) {
	if os.Getenv("MANGATRANS_LIVE_TEST") == "" {
		t.Skip("set MANGATRANS_LIVE_TEST=1 to run the live model test")
	}
	exe := `C:\Users\Public\Temp\opencode\llamacpp\llama\llama-server.exe`
	model := `E:\work\transl\models\Hy-MT2-1.8B-UD-Q4_K_XL.gguf`
	if _, err := os.Stat(exe); err != nil {
		t.Skip("llama-server not found")
	}
	if _, err := os.Stat(model); err != nil {
		t.Skip("translation model not found")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	log := filepath.Join(t.TempDir(), "llama.log")
	srv, err := llm.Start(ctx, llm.Config{
		Exe:     exe,
		Model:   model,
		LogPath: log,
		Args:    []string{"-ngl", "99", "-c", "4096", "--jinja"},
	})
	if err != nil {
		t.Skipf("llama-server failed to start: %v", err)
	}
	defer srv.Kill()

	blocks := []Block{
		{ID: 1, Text: "宇宙船で来たの？"},
		{ID: 2, Text: "うん"},
		{ID: 3, Text: "……"},
	}
	got, err := New(srv, nil).TranslateBatch(ctx, blocks)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("translations: %v", got)
	if got[1] == "" || got[1] == blocks[0].Text {
		t.Errorf("block 1 not translated: %q", got[1])
	}
}
