package translate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"mangatrans/internal/llm"
)

// Block is one OCR'd text region to translate.
type Block struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

// Result maps block ID to its translation. Empty string means the model
// judged the block untranslatable (pure symbols, OCR noise, etc.).
type Result map[int]string

const systemPrompt = `你是一个专业的日文漫画翻译，把日文对白翻译成简体中文。

【规则】
1. 译成自然、口语化的中文，保留人物语气，不要书面腔。
2. 片假名外来语必须译成中文，例如：ガイネン→概念，ギタイ→拟态，ケーキ→蛋糕。
3. text 里的换行只是排版分行，要先理解整段再翻译，不要逐行直译。
4. 译文尽量简短，不添加原文没有的信息，不加解释，不加引号或括号。译文写成一行，不要输出换行符。
5. 人名用常见汉字写法；拟声词、感叹词照常翻译。
6. 纯符号、页码或 OCR 乱码，text 输出空字符串。

【输入】
{"blocks":[{"id":1,"text":"..."},...]}

【输出】
只输出一个 JSON 对象，不要任何其他文字：
{"blocks":[{"id":1,"text":"译文"},...]}
每个输入 id 必须对应一个输出 id，不得遗漏、合并或调换顺序。

【示例】
输入：{"blocks":[{"id":1,"text":"宇宙船で来たの？"},{"id":2,"text":"うん"},{"id":3,"text":"ガイネン…"},{"id":4,"text":"どうしよ…"}]}
输出：{"blocks":[{"id":1,"text":"你是坐宇宙船来的？"},{"id":2,"text":"嗯"},{"id":3,"text":"概念…"},{"id":4,"text":"怎么办…"}]}`

// Client translates batches of blocks through a llama-server.
type Client struct {
	srv           *llm.Server
	glossary      map[string]string
	BatchAttempts int
}

// New wraps a running llama-server. glossary maps source terms to the exact
// Chinese wording the translation must use; it may be nil.
func New(srv *llm.Server, glossary map[string]string) *Client {
	return &Client{srv: srv, glossary: glossary, BatchAttempts: 2}
}

// TranslateBatch sends one batch and parses the JSON reply. Blocks the model
// fails to return are retried once on their own; anything still missing is
// left out of the result so the caller keeps the original text. A partial
// result is far better than failing a whole page because of one bad object.
func (c *Client) TranslateBatch(ctx context.Context, blocks []Block) (Result, error) {
	if len(blocks) == 0 {
		return Result{}, nil
	}

	result := make(Result, len(blocks))
	pending := append([]Block(nil), blocks...)
	var lastErr error

	for attempt := 0; attempt < max(1, min(c.BatchAttempts, 5)) && len(pending) > 0; attempt++ {
		req := struct {
			Blocks []Block `json:"blocks"`
		}{Blocks: pending}
		payload, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		msgs := []llm.ChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: withGlossary(string(payload), pending, c.glossary)},
		}
		content, err := c.srv.ChatJSON(ctx, msgs, batchSchema, 0.0, 4096)
		if err != nil {
			lastErr = fmt.Errorf("translate request: %w", err)
			if ctx.Err() != nil {
				break
			}
			continue
		}
		got := parseLoose(content)
		next := pending[:0]
		for _, b := range pending {
			if txt, ok := got[b.ID]; ok {
				txt = cleanOne(txt)
				if !Untranslated(b.Text, txt) {
					result[b.ID] = txt
					continue
				}
			}
			next = append(next, b)
		}
		pending = next
		lastErr = fmt.Errorf("reply missing %d of %d ids", len(pending), len(blocks))
	}

	if len(result) == 0 && len(blocks) > 0 {
		return nil, fmt.Errorf("translate parse: %w", lastErr)
	}
	return result, nil
}

// TranslateOne translates a single line with the model's plain translation
// template. It is the fallback for blocks that came back empty from the JSON
// batch, which happens with name lists and lines that start with ellipses.
func (c *Client) TranslateOne(ctx context.Context, text string) (string, error) {
	text = strings.TrimSpace(text)
	if text == "" {
		return "", nil
	}
	msgs := []llm.ChatMessage{
		{Role: "system", Content: "你是日文漫画翻译。把用户给的一行日文翻译成简体中文，只输出译文本身，不要解释，不要引号。"},
		{Role: "user", Content: text},
	}
	out, err := c.srv.Chat(ctx, msgs, 0.0, 256)
	if err != nil {
		return "", err
	}
	out = cleanOne(out)
	if Untranslated(text, out) {
		return "", nil
	}
	return out, nil
}

func cleanOne(s string) string {
	s = strings.TrimSpace(s)
	for _, p := range []string{"译文：", "翻译：", "中文："} {
		s = strings.TrimPrefix(s, p)
	}
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "\"“”「」『』")
	return strings.TrimSpace(s)
}

// Untranslated rejects replies that are empty, unchanged, or still contain
// kana. A small model occasionally copies Japanese into the "translation";
// drawing that over the erased source is worse than keeping the original
// page untouched, so the caller treats these blocks as missing.
func Untranslated(src, dst string) bool {
	dst = strings.TrimSpace(dst)
	if dst == "" || dst == strings.TrimSpace(src) {
		return true
	}
	for _, r := range dst {
		if (r >= 0x3040 && r <= 0x30ff) || (r >= 0x31f0 && r <= 0x31ff) ||
			(r >= 0xff66 && r <= 0xff9d) {
			return true
		}
	}
	return false
}

var trailingComma = regexp.MustCompile(`,\s*([}\]])`)

// LoadGlossary reads a JSON object {"source":"target"} or a text file with one
// "source=target" pair per line. Blank lines and # comments are ignored.
func LoadGlossary(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]string
	if err := json.Unmarshal(data, &m); err == nil {
		return m, nil
	}
	m = make(map[string]string)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(line, "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		src, dst, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		src, dst = strings.TrimSpace(src), strings.TrimSpace(dst)
		if src != "" && dst != "" {
			m[src] = dst
		}
	}
	return m, nil
}

// withGlossary prepends the entries that occur in this batch. Only relevant
// terms are sent, so the prompt stays short and the model is not distracted by
// names from another scene.
func withGlossary(payload string, blocks []Block, g map[string]string) string {
	if len(g) == 0 {
		return payload
	}
	var joined strings.Builder
	for _, b := range blocks {
		joined.WriteString(b.Text)
	}
	src := joined.String()
	var pairs []string
	for from, to := range g {
		if strings.Contains(src, from) {
			pairs = append(pairs, from+" => "+to)
		}
	}
	if len(pairs) == 0 {
		return payload
	}
	sort.Strings(pairs)
	return "术语表（以下译名必须原样使用）：\n" + strings.Join(pairs, "\n") + "\n\n" + payload
}

// batchSchema lets llama-server constrain generation to the reply shape. It
// cannot force every id to appear, so parseLoose still validates the result.
var batchSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"blocks": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id":   map[string]any{"type": "integer"},
					"text": map[string]any{"type": "string"},
				},
				"required":             []string{"id", "text"},
				"additionalProperties": false,
			},
		},
	},
	"required":             []string{"blocks"},
	"additionalProperties": false,
}

// parseLoose extracts {id, text} pairs from a reply that may contain code
// fences, prose, concatenated objects or a top-level array. It never fails
// outright: whatever valid pairs it finds are returned.
func parseLoose(content string) Result {
	out := make(Result)
	content = strings.TrimSpace(content)

	// Fast path: a well-formed {"blocks":[...]} object.
	if start := strings.IndexByte(content, '{'); start >= 0 {
		if end := strings.LastIndexByte(content, '}'); end > start {
			var strict struct {
				Blocks []struct {
					ID   json.RawMessage `json:"id"`
					Text *string         `json:"text"`
				} `json:"blocks"`
			}
			if err := json.Unmarshal([]byte(content[start:end+1]), &strict); err == nil {
				for _, b := range strict.Blocks {
					if id, ok := parseID(b.ID); ok && b.Text != nil {
						if _, dup := out[id]; !dup {
							out[id] = *b.Text
						}
					}
				}
				if len(out) > 0 {
					return out
				}
			}
		}
	}

	// Slow path: walk every balanced object and keep those that carry both an
	// "id" and a "text" field. This survives stray commas, markdown and
	// objects glued together without an array wrapper.
	for _, obj := range balancedObjects(content) {
		var one struct {
			ID   json.RawMessage `json:"id"`
			Text *string         `json:"text"`
		}
		if err := json.Unmarshal([]byte(obj), &one); err != nil {
			fixed := trailingComma.ReplaceAllString(obj, "$1")
			if err2 := json.Unmarshal([]byte(fixed), &one); err2 != nil {
				continue
			}
		}
		if one.Text == nil {
			continue
		}
		id, ok := parseID(one.ID)
		if !ok {
			continue
		}
		if _, dup := out[id]; !dup {
			out[id] = *one.Text
		}
	}
	return out
}

func parseID(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return 0, false
		}
		n, err := strconv.Atoi(strings.TrimSpace(s))
		return n, err == nil
	}
	var n int
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0, false
	}
	return n, true
}

// balancedObjects returns every balanced {...} substring, innermost first, so
// nested block objects are visited before their wrapper.
func balancedObjects(s string) []string {
	var out []string
	var stack []int
	inString := false
	escaped := false
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if inString {
			if escaped {
				escaped = false
			} else if ch == '\\' {
				escaped = true
			} else if ch == '"' {
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			stack = append(stack, i)
		case '}':
			if n := len(stack); n > 0 {
				start := stack[n-1]
				stack = stack[:n-1]
				out = append(out, s[start:i+1])
			}
		}
	}
	return out
}
