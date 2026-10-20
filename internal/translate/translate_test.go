package translate

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"mangatrans/internal/llm"
)

func TestFlatResponseAcceptsOnlyUniqueRequestedTargets(t *testing.T) {
	blocks := []Block{
		{Key: TargetKey{Page: 4, Region: 0}, Text: "甲"},
		{Key: TargetKey{Page: 4, Region: 1}, Text: "乙"},
		{Key: TargetKey{Page: 5, Region: 2}, Text: "丙"},
	}
	targets, err := collectTargets(blocks)
	if err != nil {
		t.Fatal(err)
	}
	content := "prefix {\"blocks\":[{\"id\":\"p000004-r0000\",\"text\":\"甲译\"},{\"id\":\"p000004-r0008\",\"text\":\"未知\"},{\"id\":\"p000005-r0002\",\"text\":\"丙译\"},{\"id\":\"p000005-r0002\",\"text\":\"重复\"},{\"id\":\"p000009-r0001\",\"text\":\"别页\"}]} suffix"
	reply, err := parseBatchResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	got, issues := validateResponse(targets, reply)
	if got[TargetKey{Page: 4, Region: 0}] != "甲译" {
		t.Fatalf("page 4 region 0 = %q, want %q", got[TargetKey{Page: 4, Region: 0}], "甲译")
	}
	if _, ok := got[TargetKey{Page: 4, Region: 1}]; ok {
		t.Fatal("missing target was accepted")
	}
	if _, ok := got[TargetKey{Page: 5, Region: 2}]; ok {
		t.Fatal("duplicate target was accepted")
	}
	if len(issues) != 4 {
		t.Fatalf("got validation issues %v, want unknown, duplicate, and two missing ids", issues)
	}
}

func TestCollectTargetsRejectsAmbiguousIdentity(t *testing.T) {
	_, err := collectTargets([]Block{
		{Key: TargetKey{Page: 3, Region: 0}, Text: "甲"},
		{Key: TargetKey{Page: 3, Region: 0}, Text: "乙"},
	})
	if err == nil {
		t.Fatal("duplicate target keys must be rejected")
	}
	_, err = collectTargets([]Block{{Key: TargetKey{Page: -1, Region: 0}, Text: "甲"}})
	if err == nil {
		t.Fatal("negative target coordinates must be rejected")
	}
}

func TestTargetIDIncludesPageAndRegion(t *testing.T) {
	first := targetID(TargetKey{Page: 4, Region: 1})
	second := targetID(TargetKey{Page: 5, Region: 1})
	third := targetID(TargetKey{Page: 4, Region: 2})
	if first == second || first == third || second == third {
		t.Fatal("distinct page/region keys produced the same ID")
	}
}

func TestDiagnosticWritesExactMessagesAndRawResponse(t *testing.T) {
	dir := t.TempDir()
	record := translationDiagnostic{
		BatchIndex: 7,
		Attempt:    2,
		Kind:       "single-target",
		Page:       4,
		Region:     1,
		Messages: []llm.ChatMessage{
			{Role: "system", Content: "system prompt"},
			{Role: "user", Content: "payload"},
		},
		RawResponse: "{\"blocks\":[]}",
	}
	if err := writeDiagnostic(dir, record); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "batch-000007-p000004-r0001-attempt-02.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got translationDiagnostic
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	if got.Messages[1].Content != "payload" || got.RawResponse != record.RawResponse || got.Kind != record.Kind {
		t.Fatalf("diagnostic did not preserve request/response: %+v", got)
	}
	if got.Page != 4 || got.Region != 1 {
		t.Fatalf("diagnostic did not preserve the target key: %+v", got)
	}
}

func TestParseReplyIgnoresTrailingProse(t *testing.T) {
	content := "{\"blocks\":[{\"id\":\"p000003-r0000\",\"text\":\"你好\"}]} 机内提示：已翻译完成，不要输出解释。"
	reply, err := parseBatchResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(reply) != 1 || reply[0].Text == nil || *reply[0].Text != "你好" {
		t.Fatalf("reply not recovered: %+v", reply)
	}
}

func TestParseReplyRepairsTypographicDelimiter(t *testing.T) {
	content := "{\"blocks\":[{\"id\":\"p000008-r0006\",\"text\":\"叔叔，你没醉吗？”}]}机内声音：叔父さん，你没喝醉吗\"}] }"
	reply, err := parseBatchResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(reply) != 1 || reply[0].Text == nil {
		t.Fatalf("reply not recovered: %+v", reply)
	}
	if got := cleanOne(*reply[0].Text); got != "叔叔，你没醉吗？" {
		t.Fatalf("recovered text = %q, want %q", got, "叔叔，你没醉吗？")
	}
}

func TestParseReplyKeepsValidStringWithQuotationMark(t *testing.T) {
	content := "{\"blocks\":[{\"id\":\"p000003-r0000\",\"text\":\"他说“好”]然后走了\"}]}"
	reply, err := parseBatchResponse(content)
	if err != nil {
		t.Fatal(err)
	}
	if len(reply) != 1 || reply[0].Text == nil || *reply[0].Text != "他说“好”]然后走了" {
		t.Fatalf("valid reply was altered: %+v", reply)
	}
}

func TestUntranslatedKeepsChineseButRejectsJapaneseRemainder(t *testing.T) {
	cases := []struct {
		name string
		src  string
		dst  string
		want bool
	}{
		{"empty reply", "宇宙船で来たの？", "", true},
		{"chinese name kept in katakana", "トージョー・リタの兄だな", "原来是トージョー・リタ的哥哥啊", false},
		{"chinese with middle dot prefix", "・社会から疎外されるのは当然だった", "・被社会排斥是理所当然的", false},
		{"date needs no translation", "3月19日", "3月19日", false},
		{"latin brand needs no translation", "♡niceríaHermanos", "♡niceríaHermanos", false},
		{"hiragana particle left", "アステカの廻天", "阿斯特卡の回天", true},
		{"verb ending left", "身代金期限の", "赎金期限の", true},
		{"katakana echo", "ヒソヒソ", "ヒソヒソ", true},
		{"translated", "国境を越えてトロカが走る", "跨越国界，托洛卡在奔跑", false},
	}
	for _, c := range cases {
		if got := Untranslated(c.src, c.dst); got != c.want {
			t.Errorf("%s: Untranslated(%q, %q) = %v, want %v", c.name, c.src, c.dst, got, c.want)
		}
	}
}

func TestTranslateBatchLive(t *testing.T) {
	if os.Getenv("MANGATRANS_LIVE_TEST") == "" {
		t.Skip("set MANGATRANS_LIVE_TEST=1 to run the live model test")
	}
	exe := "C:\\Users\\Public\\Temp\\opencode\\llamacpp\\llama\\llama-server.exe"
	model := "E:\\work\\transl\\models\\Hy-MT2-1.8B-UD-Q4_K_XL.gguf"
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

	blocks := []Block{{Key: TargetKey{Page: 3, Region: 0}, Text: "宇宙船で来たの？"}}
	got, err := New(srv, nil).TranslateBatch(ctx, 0, blocks)
	if err != nil {
		t.Fatal(err)
	}
	translated := got[blocks[0].Key]
	t.Logf("translations: %v", got)
	if translated == "" || translated == blocks[0].Text {
		t.Errorf("block not translated: %q", translated)
	}
}
