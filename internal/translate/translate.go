package translate

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"mangatrans/internal/llm"
)

type TargetKey struct {
	Page   int `json:"page"`
	Region int `json:"region"`
}

type Block struct {
	Key  TargetKey
	Text string
}

type Result map[TargetKey]string

type Client struct {
	srv            *llm.Server
	glossary       map[string]string
	BatchAttempts  int
	DiagnosticsDir string
}

type translationTarget struct {
	key  TargetKey
	id   string
	text string
}

type translationRequest struct {
	Blocks []requestBlock `json:"blocks"`
}

type requestBlock struct {
	ID     string `json:"id"`
	Source string `json:"source"`
}

type responseEnvelope struct {
	Blocks *[]responseBlock `json:"blocks"`
}

type responseBlock struct {
	ID   json.RawMessage `json:"id"`
	Text *string         `json:"text"`
}

type diagnosticTarget struct {
	ID     string `json:"id"`
	Page   int    `json:"page"`
	Region int    `json:"region"`
	Source string `json:"source"`
}

type translationDiagnostic struct {
	BatchIndex   int                `json:"batch_index"`
	Attempt      int                `json:"attempt"`
	Kind         string             `json:"kind"`
	Page         int                `json:"page"`
	Region       int                `json:"region"`
	Messages     []llm.ChatMessage  `json:"messages"`
	Schema       map[string]any     `json:"schema"`
	Targets      []diagnosticTarget `json:"targets"`
	RawResponse  string             `json:"raw_response"`
	ReturnedIDs  []string           `json:"returned_ids,omitempty"`
	Issues       []string           `json:"validation_issues,omitempty"`
	RequestError string             `json:"request_error,omitempty"`
	ParseError   string             `json:"parse_error,omitempty"`
}

// A single dialogue line is short. A low token ceiling bounds the model's
// known degenerate repetition loops instead of letting them run to the context
// limit.
const maxReplyTokens = 512

const systemPrompt = "你是一个专业的日文漫画翻译，把日文对白翻译成简体中文。\n\n" +
	"【规则】\n" +
	"1. 只翻译 source 这一条，只输出一条译文。不合并、不拆分、不省略、不补充。\n" +
	"2. 准确保留肯定或否定、疑问或命令、时态、数量、方向、人物关系和语气。遇到惯用语先按语境理解，不要按字面误译；不得反转原意或猜测 OCR 没有提供的事实。\n" +
	"3. 日语的否定和反事实必须译成中文的否定和反事实：ない、ねー、ぬ、ず 是否定；「〜のに」「〜ねーのに」是“明明没有…却…”的结构，例如「金も持ってねーのに」是“明明没钱”，不得译成“明明有钱”。不得把否定译成肯定。\n" +
	"4. 数量、年龄、顺序按原文：手前＝快要到（９０手前＝快90岁，不是90年前）；倍数、单位、金额不要换算或改写。\n" +
	"5. 译成自然、口语化的简体中文，保留人物语气，不要书面腔。片假名外来语译成中文；人名、地名使用常见汉字写法；拟声词和感叹词照常翻译。生物、食物、物品要用准确的中文常用名，不要换成别的物种或近义词。\n" +
	"6. 译文里不得出现平假名（の、は、が、を、に、り 等），助词和送假名必须译成中文。句子被截断时按中文习惯补全或直接省略助词，例如「国は…」译成「国家…」，「身代金期限の」译成「赎金期限」，不要把假名留在译文里。人名、地名确实无法确定中文写法时，才可以保留片假名。\n" +
	"7. source 中的换行只是排版分行，应先理解整段再翻译。译文尽量简短，不添加原文没有的信息，不加解释、引号或括号；原文的「」引号写成中文引号；译文只占一行。\n" +
	"8. 纯符号、页码或明显 OCR 乱码可输出空字符串，不要把看不懂的内容编造成对白。\n\n" +
	"【输入格式】\n" +
	"{\"blocks\":[{\"id\":\"p000003-r0000\",\"source\":\"...\"}]}\n\n" +
	"【输出格式】\n" +
	"只输出一个 JSON 对象，不要任何其他文字。返回完全相同的 id，只返回这一条：\n" +
	"{\"blocks\":[{\"id\":\"p000003-r0000\",\"text\":\"译文\"}]}\n\n" +
	"【示例】\n" +
	"输入：{\"blocks\":[{\"id\":\"p000003-r0000\",\"source\":\"宇宙船で来たの？\"}]}\n" +
	"输出：{\"blocks\":[{\"id\":\"p000003-r0000\",\"text\":\"你是坐宇宙船来的？\"}]}"

func New(srv *llm.Server, glossary map[string]string) *Client {
	return &Client{srv: srv, glossary: glossary, BatchAttempts: 2}
}

// TranslateBatch translates one region per request.
//
// The model renumbers composite IDs when a request carries several targets, so
// a batched reply cannot be matched to its source region at all. One target per
// request keeps the requested ID exact, prevents neighbouring blocks from
// bleeding into each other, and makes every reply verifiable without relying on
// array position.
func (c *Client) TranslateBatch(ctx context.Context, batchIndex int, blocks []Block) (Result, error) {
	targets, err := collectTargets(blocks)
	if err != nil {
		return nil, err
	}
	if len(targets) == 0 {
		return Result{}, nil
	}

	result := make(Result, len(targets))
	var failures []string
	var diagnosticErrors []string
	attempts := max(1, min(c.BatchAttempts, 5))
	for _, target := range targets {
		resolved := false
		var lastErr error
		var lastText string
		for attempt := 1; attempt <= attempts; attempt++ {
			if ctx.Err() != nil {
				lastErr = ctx.Err()
				break
			}
			hint := ""
			if attempt > 1 {
				hint = retryHint(target, lastText)
			}
			got, issues, requestErr, diagnosticErr := c.translateRequest(ctx, batchIndex, attempt, target, hint)
			if diagnosticErr != nil {
				diagnosticErrors = append(diagnosticErrors, diagnosticErr.Error())
			}
			if requestErr != nil {
				lastErr = requestErr
				continue
			}
			if text, ok := got[target.key]; ok {
				if !Untranslated(target.text, text) {
					result[target.key] = text
					resolved = true
					break
				}
				lastText = text
				lastErr = fmt.Errorf("page %d region %d kept Japanese text %q", target.key.Page, target.key.Region, text)
				continue
			}
			lastText = ""
			lastErr = fmt.Errorf("page %d region %d produced no usable reply: %s", target.key.Page, target.key.Region, strings.Join(issues, "; "))
		}
		if !resolved {
			if lastErr == nil {
				lastErr = fmt.Errorf("page %d region %d unresolved", target.key.Page, target.key.Region)
			}
			failures = append(failures, lastErr.Error())
		}
	}

	var errors []string
	if len(failures) > 0 {
		errors = append(errors, fmt.Sprintf("%d of %d targets unresolved: %s", len(failures), len(targets), strings.Join(failures, "; ")))
	}
	if len(diagnosticErrors) > 0 {
		errors = append(errors, "diagnostics write failed: "+strings.Join(diagnosticErrors, "; "))
	}
	if len(errors) > 0 {
		return result, fmt.Errorf("translate batch %d: %s", batchIndex, strings.Join(errors, "; "))
	}
	return result, nil
}

func (c *Client) translateRequest(ctx context.Context, batchIndex, attempt int, target translationTarget, hint string) (Result, []string, error, error) {
	request := translationRequest{Blocks: []requestBlock{{ID: target.id, Source: target.text}}}
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, nil, err, nil
	}
	userPayload := withGlossary(string(payload), target, c.glossary)
	if hint != "" {
		userPayload = hint + "\n\n" + userPayload
	}
	messages := []llm.ChatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPayload},
	}
	content, requestErr := c.srv.ChatJSON(ctx, messages, batchSchema, 0.0, maxReplyTokens)
	var responseBlocks []responseBlock
	var parseErr error
	result := make(Result)
	var issues []string
	var returnedIDs []string
	if requestErr == nil {
		responseBlocks, parseErr = parseBatchResponse(content)
		if parseErr == nil {
			for _, block := range responseBlocks {
				if id, ok := parseResponseID(block.ID); ok {
					returnedIDs = append(returnedIDs, id)
				}
			}
			result, issues = validateResponse([]translationTarget{target}, responseBlocks)
		}
	}
	var diagnosticErr error
	if c.DiagnosticsDir != "" {
		record := translationDiagnostic{
			BatchIndex:  batchIndex,
			Attempt:     attempt,
			Kind:        "single-target",
			Page:        target.key.Page,
			Region:      target.key.Region,
			Messages:    messages,
			Schema:      batchSchema,
			Targets:     []diagnosticTarget{{ID: target.id, Page: target.key.Page, Region: target.key.Region, Source: target.text}},
			RawResponse: content,
			ReturnedIDs: returnedIDs,
			Issues:      issues,
		}
		if requestErr != nil {
			record.RequestError = requestErr.Error()
		}
		if parseErr != nil {
			record.ParseError = parseErr.Error()
		}
		diagnosticErr = writeDiagnostic(c.DiagnosticsDir, record)
	}
	if requestErr != nil {
		return result, issues, fmt.Errorf("translate request: %w", requestErr), diagnosticErr
	}
	if parseErr != nil {
		return result, issues, fmt.Errorf("translate parse: %w", parseErr), diagnosticErr
	}
	return result, issues, nil, diagnosticErr
}

// retryHint tells the model what failed in the previous reply. Repeating the
// identical prompt returned the identical wrong text, so a retry only helps
// when it names the problem.
func retryHint(target translationTarget, previous string) string {
	previous = strings.TrimSpace(previous)
	if previous == "" {
		return fmt.Sprintf("上一版没有给出可用的 JSON 译文。请只输出 {\"blocks\":[{\"id\":%q,\"text\":\"译文\"}]}，不要输出其他文字。", target.id)
	}
	return fmt.Sprintf("上一版译文 %q 里还留着日文假名 %s，未通过检查。请把 %s 和所有助词、送假名、动词词尾都译成简体中文，把残句补成完整的中文句子，只输出 JSON 对象。",
		previous, hiraganaList(previous), hiraganaList(previous))
}

func hiraganaList(text string) string {
	seen := make(map[rune]bool)
	var out []string
	for _, r := range text {
		if !unicode.Is(unicode.Hiragana, r) || seen[r] {
			continue
		}
		seen[r] = true
		out = append(out, string(r))
	}
	return "「" + strings.Join(out, "」「") + "」"
}

func collectTargets(blocks []Block) ([]translationTarget, error) {
	targets := make([]translationTarget, 0, len(blocks))
	seen := make(map[TargetKey]struct{}, len(blocks))
	for _, block := range blocks {
		text := strings.TrimSpace(block.Text)
		if block.Key.Page < 0 || block.Key.Region < 0 || text == "" {
			return nil, fmt.Errorf("invalid target page %d region %d", block.Key.Page, block.Key.Region)
		}
		if _, exists := seen[block.Key]; exists {
			return nil, fmt.Errorf("duplicate target page %d region %d", block.Key.Page, block.Key.Region)
		}
		seen[block.Key] = struct{}{}
		targets = append(targets, translationTarget{key: block.Key, id: targetID(block.Key), text: text})
	}
	return targets, nil
}

func targetID(key TargetKey) string {
	return fmt.Sprintf("p%06d-r%04d", key.Page, key.Region)
}

func validateResponse(targets []translationTarget, reply []responseBlock) (Result, []string) {
	expected := make(map[string]TargetKey, len(targets))
	for _, target := range targets {
		expected[target.id] = target.key
	}
	result := make(Result, len(targets))
	seen := make(map[string]bool, len(targets))
	invalid := make(map[string]bool)
	var issues []string
	for _, block := range reply {
		id, ok := parseResponseID(block.ID)
		if !ok {
			issues = append(issues, "response block has invalid id")
			continue
		}
		key, exists := expected[id]
		if !exists {
			issues = append(issues, "unknown id "+id)
			continue
		}
		if seen[id] {
			delete(result, key)
			invalid[id] = true
			issues = append(issues, "duplicate id "+id)
			continue
		}
		seen[id] = true
		if block.Text == nil {
			invalid[id] = true
			issues = append(issues, "missing text for "+id)
			continue
		}
		result[key] = cleanOne(*block.Text)
	}
	for id, key := range expected {
		if !seen[id] {
			issues = append(issues, "missing id "+id)
			continue
		}
		if invalid[id] {
			delete(result, key)
		}
	}
	sort.Strings(issues)
	return result, issues
}

func parseBatchResponse(content string) ([]responseBlock, error) {
	content = strings.TrimSpace(content)
	// The model sometimes appends prose after a complete object, or closes a JSON
	// string with a typographic quote and then swallows that prose into the last
	// value. A repaired candidate is tried first because "valid JSON" can still
	// contain the swallowed tail; the raw text remains the fallback.
	for _, candidate := range []string{repairQuotedDelimiter(content), content} {
		if start := strings.IndexByte(candidate, '{'); start >= 0 {
			var response responseEnvelope
			if err := json.NewDecoder(strings.NewReader(candidate[start:])).Decode(&response); err == nil && response.Blocks != nil {
				return *response.Blocks, nil
			}
		}
	}
	var recovered []responseBlock
	for _, object := range balancedObjects(content) {
		var block responseBlock
		if err := json.Unmarshal([]byte(object), &block); err != nil {
			fixed := trailingComma.ReplaceAllString(object, "$1")
			if err = json.Unmarshal([]byte(fixed), &block); err != nil {
				continue
			}
		}
		if len(block.ID) > 0 && block.Text != nil {
			recovered = append(recovered, block)
		}
	}
	if len(recovered) > 0 {
		return recovered, nil
	}
	return nil, fmt.Errorf("response contains no usable blocks")
}

func parseResponseID(raw json.RawMessage) (string, bool) {
	if len(raw) == 0 {
		return "", false
	}
	var id string
	if err := json.Unmarshal(raw, &id); err != nil || id == "" {
		return "", false
	}
	return id, true
}

var trailingComma = regexp.MustCompile(`,\s*([}\]])`)

// typographicDelimiter matches a typographic closing quote used where JSON needs
// an ASCII string terminator. The model produced replies like
// {"text":"叔叔，你没醉吗？”}]} where the ASCII quote is missing; inserting it
// recovers the intended single block instead of an empty target.
var typographicDelimiter = regexp.MustCompile(`”\s*([\]}])`)

func repairQuotedDelimiter(s string) string {
	if !typographicDelimiter.MatchString(s) {
		return s
	}
	return typographicDelimiter.ReplaceAllString(s, `"$1`)
}

func balancedObjects(s string) []string {
	var out []string
	var stack []int
	inString, escaped := false, false
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

func cleanOne(s string) string {
	s = strings.TrimSpace(s)
	for _, prefix := range []string{"译文：", "翻译：", "中文："} {
		s = strings.TrimPrefix(s, prefix)
	}
	s = strings.TrimSpace(s)
	s = strings.Trim(s, "\"“”「」『』")
	return strings.TrimSpace(s)
}

// Untranslated reports whether dst still needs a translation for src.
//
// An empty reply is unresolved. A source with no kana needs no translation, so
// echoing it (a date, a symbol line, a Latin brand) is the correct result.
// Chinese output may keep katakana proper nouns, but hiragana marks an
// untranslated Japanese clause and a mostly katakana reply is an echo, so both
// stay unresolved. A reply that is otherwise Chinese may quote a Japanese
// term; a few kana beside many Han characters are accepted, because rejecting
// them threw away a finished translation.
func Untranslated(src, dst string) bool {
	src = strings.TrimSpace(src)
	dst = strings.TrimSpace(dst)
	if dst == "" {
		return true
	}
	if !hasKana(src) {
		return false
	}
	hiragana, katakana, letters, han := 0, 0, 0, 0
	for _, r := range dst {
		switch {
		case unicode.Is(unicode.Hiragana, r):
			hiragana++
			letters++
		case unicode.Is(unicode.Katakana, r):
			katakana++
			letters++
		case unicode.Is(unicode.Han, r):
			han++
			letters++
		case unicode.IsLetter(r):
			letters++
		}
	}
	// A quoted Japanese term can leave one or two kana inside an otherwise
	// Chinese reply ("“牛の脳”啊，…"). Rejecting those discarded a finished
	// translation. Japanese that was never translated is kana-heavy, so a
	// reply is only suspect when kana are not a small minority of the Han
	// characters.
	if hiragana > 0 && (han == 0 || hiragana*4 > han) {
		return true
	}
	return letters > 0 && katakana*10 > letters*6
}

func hasKana(s string) bool {
	for _, r := range s {
		if unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r) {
			return true
		}
	}
	return false
}

func LoadGlossary(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var glossary map[string]string
	if err := json.Unmarshal(data, &glossary); err == nil {
		return glossary, nil
	}
	glossary = make(map[string]string)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(line, "\ufeff"))
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		source, target, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		source, target = strings.TrimSpace(source), strings.TrimSpace(target)
		if source != "" && target != "" {
			glossary[source] = target
		}
	}
	return glossary, nil
}

func withGlossary(payload string, target translationTarget, glossary map[string]string) string {
	if len(glossary) == 0 {
		return payload
	}
	var pairs []string
	for from, to := range glossary {
		if strings.Contains(target.text, from) {
			pairs = append(pairs, from+" => "+to)
		}
	}
	if len(pairs) == 0 {
		return payload
	}
	sort.Strings(pairs)
	return "术语表（以下译名必须原样使用）：\n" + strings.Join(pairs, "\n") + "\n\n" + payload
}

func writeDiagnostic(dir string, record translationDiagnostic) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(dir, fmt.Sprintf("batch-%06d-p%06d-r%04d-attempt-%02d.json",
		record.BatchIndex, record.Page, record.Region, record.Attempt))
	file, err := os.CreateTemp(dir, ".translation-*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(file.Name())
	encoder := json.NewEncoder(file)
	encoder.SetEscapeHTML(false)
	err = encoder.Encode(record)
	if err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(file.Name(), path)
}

var batchSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"blocks": map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id":   map[string]any{"type": "string"},
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
