package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"mangatrans/internal/perf"
)

type contextContext = context.Context

func jsonMarshal(v interface{}) ([]byte, error)   { return json.Marshal(v) }
func jsonUnmarshal(b []byte, v interface{}) error { return json.Unmarshal(b, v) }
func newRequest(ctx contextContext, method, url string, body []byte) (*http.Request, error) {
	return http.NewRequestWithContext(ctx, method, url, bytes.NewReader(body))
}
func readAll(r io.Reader) ([]byte, error) { return io.ReadAll(r) }

var client = &http.Client{Timeout: 300 * time.Second}

// ImageContentPart is one image in a multimodal message.
type ImageContentPart struct {
	Type     string   `json:"type"`
	ImageURL imageURL `json:"image_url"`
}

type imageURL struct {
	URL string `json:"url"`
}

// TextContentPart is one text chunk in a multimodal message.
type TextContentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ImageURLStruct builds the nested image_url object.
func ImageURLStruct(url string) imageURL {
	return imageURL{URL: url}
}

// MultimodalMessage is a message whose content mixes text and images.
type MultimodalMessage struct {
	Role    string        `json:"role"`
	Content []interface{} `json:"content"`
}

type mmChatRequest struct {
	Messages    []MultimodalMessage `json:"messages"`
	Temperature float64             `json:"temperature"`
	MaxTokens   int                 `json:"max_tokens"`
	Stream      bool                `json:"stream"`
}

// ChatMultimodal sends a chat completion with image parts.
// PNG/JPEG bytes are inlined as base64 data URLs.
func (s *Server) ChatMultimodal(ctx contextContext, messages []MultimodalMessage, temperature float64, maxTokens int) (content string, err error) {
	end := perf.Request(ctx, "ocr_http")
	defer func() { end(err) }()
	body, err := jsonMarshal(mmChatRequest{
		Messages:    messages,
		Temperature: temperature,
		MaxTokens:   maxTokens,
	})
	if err != nil {
		return "", err
	}
	return s.post(ctx, body)
}

// DataURL wraps image bytes for an image_url content part.
func DataURL(mime string, data []byte) string {
	return "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
}

func (s *Server) post(ctx contextContext, body []byte) (string, error) {
	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	req, err := newRequest(ctx, "POST", url, body)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	data, err := readAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("chat http %d: %s", resp.StatusCode, truncate(string(data), 400))
	}

	var cr chatResponse
	if err := jsonUnmarshal(data, &cr); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if cr.Error != nil {
		return "", fmt.Errorf("server error: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("empty choices")
	}
	return cr.Choices[0].Message.Content, nil
}
