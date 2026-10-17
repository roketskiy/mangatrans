package translate

import (
	"context"
	"encoding/json"
	"fmt"
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

const systemPrompt = `你是一个专业的漫画翻译器。任务是把日文漫画对白翻译成简体中文。

【翻译要求】
1. 准确保留原意、语气和人物性格，译文要自然口语化，符合中文漫画表达习惯。
2. 拟声词和感叹词也要翻译。
3. 人名保留原文读音对应的常见汉字写法，没有把握时保留原文。
4. 遇到歧义立即选最可能的译法，不解释。
5. 不添加原文没有的信息。

【输入格式】
输入是 JSON：{"blocks":[{"id":1,"text":"..."},...]}

【输出格式】
只输出一个 JSON 对象，不要任何其他文字：
{"blocks":[{"id":1,"text":"译文"},...]}

每个输入 id 必须对应一个输出 id，不得遗漏、合并或调换顺序。
无法翻译的块（纯符号、OCR 乱码、页码）text 输出空字符串。

【示例】
输入：{"blocks":[{"id":1,"text":"宇宙船で来たの？"},{"id":2,"text":"うん"}]}
输出：{"blocks":[{"id":1,"text":"你是坐宇宙船来的？"},{"id":2,"text":"嗯"}]}`

// Client translates batches of blocks through a llama-server.
type Client struct {
	srv *llm.Server
}

// New wraps a running llama-server.
func New(srv *llm.Server) *Client { return &Client{srv: srv} }

// TranslateBatch sends one batch and parses the JSON reply.
// Retries once on parse failure before returning the error.
func (c *Client) TranslateBatch(ctx context.Context, blocks []Block) (Result, error) {
	if len(blocks) == 0 {
		return Result{}, nil
	}

	req := struct {
		Blocks []Block `json:"blocks"`
	}{Blocks: blocks}
	payload, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	msgs := []llm.ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: string(payload)},
	}

	var lastErr error
	for attempt := 0; attempt < 2; attempt++ {
		content, err := c.srv.Chat(ctx, msgs, 0.1, 4096)
		if err != nil {
			return nil, fmt.Errorf("translate request: %w", err)
		}
		result, perr := parseBlocks(content, blocks)
		if perr == nil {
			return result, nil
		}
		lastErr = perr
	}
	return nil, fmt.Errorf("translate parse after retry: %w", lastErr)
}

func parseBlocks(content string, blocks []Block) (Result, error) {
	// The model sometimes wraps JSON in a code fence or prepends prose.
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(content, "```")
	content = strings.TrimSpace(content)

	// Find the outermost JSON object.
	start := strings.IndexByte(content, '{')
	end := strings.LastIndexByte(content, '}')
	if start < 0 || end <= start {
		return nil, fmt.Errorf("no JSON object in reply: %.200q", content)
	}
	content = content[start : end+1]

	var parsed struct {
		Blocks []struct {
			ID   int    `json:"id"`
			Text string `json:"text"`
		} `json:"blocks"`
	}
	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		return nil, fmt.Errorf("decode %d chars: %w", len(content), err)
	}

	want := make(map[int]bool, len(blocks))
	for _, b := range blocks {
		want[b.ID] = true
	}
	result := make(Result, len(blocks))
	seen := make(map[int]bool, len(blocks))
	for _, b := range parsed.Blocks {
		if !want[b.ID] || seen[b.ID] {
			continue
		}
		seen[b.ID] = true
		result[b.ID] = b.Text
	}
	for _, b := range blocks {
		if !seen[b.ID] {
			return nil, fmt.Errorf("reply missing id %d", b.ID)
		}
	}
	return result, nil
}
