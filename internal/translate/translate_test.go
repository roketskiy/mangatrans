package translate

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"mangatrans/internal/llm"
)

func TestParseBlocks(t *testing.T) {
	blocks := []Block{{ID: 1, Text: "a"}, {ID: 2, Text: "b"}}

	cases := []struct {
		name    string
		content string
		want    Result
		wantErr bool
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
			wantErr: true,
		},
		{
			name:    "no json",
			content: `抱歉我不会`,
			wantErr: true,
		},
		{
			name:    "extra unknown id ignored",
			content: `{"blocks":[{"id":1,"text":"甲"},{"id":2,"text":"乙"},{"id":9,"text":"多"}]}`,
			want:    Result{1: "甲", 2: "乙"},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseBlocks(c.content, blocks)
			if c.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
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
	exe := `C:\Users\Public\Temp\opencode\llamacpp\llama\llama-server.exe`
	model := `E:\work\transl\models\Hy-MT2-1.8B-UD-Q3_K_XL.gguf`
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
	got, err := New(srv).TranslateBatch(ctx, blocks)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("translations: %v", got)
	if got[1] == "" || got[1] == blocks[0].Text {
		t.Errorf("block 1 not translated: %q", got[1])
	}
}
