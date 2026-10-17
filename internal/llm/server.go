package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

// Server manages one llama-server subprocess.
type Server struct {
	cmd      *exec.Cmd
	port     int
	exe      string
	logFile  string
	logBuf   *bytes.Buffer
	stopOnce sync.Once
	Shutdown func()
}

// Config for starting a llama-server.
type Config struct {
	Exe     string   // path to llama-server.exe
	Model   string   // path to the gguf
	MMProj  string   // path to mmproj gguf, empty for text-only
	Args    []string // extra CLI args
	LogPath string   // where to write server output
	Timeout time.Duration
}

// Start launches llama-server and waits until /health reports ready.
func Start(ctx context.Context, c Config) (*Server, error) {
	if c.Timeout == 0 {
		c.Timeout = 180 * time.Second
	}

	port, err := freePort()
	if err != nil {
		return nil, fmt.Errorf("find free port: %w", err)
	}

	args := []string{
		"--model", c.Model,
		"--host", "127.0.0.1",
		"--port", fmt.Sprint(port),
		"--no-webui",
	}
	if c.MMProj != "" {
		args = append(args, "--mmproj", c.MMProj)
	}
	args = append(args, c.Args...)

	logFile, err := os.Create(c.LogPath)
	if err != nil {
		return nil, fmt.Errorf("create log: %w", err)
	}

	cmd := exec.CommandContext(ctx, c.Exe, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: 0x08000000, // CREATE_NO_WINDOW
	}
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("start %s: %w", c.Exe, err)
	}

	s := &Server{cmd: cmd, port: port, exe: c.Exe, logFile: c.LogPath}
	s.Shutdown = func() {
		s.stopOnce.Do(func() {
			if cmd.Process != nil {
				cmd.Process.Kill()
				_, _ = cmd.Process.Wait()
			}
			logFile.Close()
		})
	}

	if err := s.waitReady(ctx, c.Timeout); err != nil {
		s.Shutdown()
		return nil, err
	}
	return s, nil
}

func (s *Server) waitReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/health", s.port)
	client := &http.Client{Timeout: 2 * time.Second}

	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		resp, err := client.Get(url)
		if err == nil {
			if resp.StatusCode == 200 {
				resp.Body.Close()
				return nil
			}
			resp.Body.Close()
		}
		// The process may have died.
		if s.cmd.ProcessState != nil && s.cmd.ProcessState.Exited() {
			return fmt.Errorf("server exited early, log at %s", s.logFile)
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("server not ready within %s, log at %s", timeout, s.logFile)
}

// Port returns the port the server listens on.
func (s *Server) Port() int { return s.port }

// ChatCompletion is one OpenAI-compatible chat request.
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type chatRequest struct {
	Messages       []ChatMessage   `json:"messages"`
	Temperature    float64         `json:"temperature"`
	MaxTokens      int             `json:"max_tokens"`
	Stream         bool            `json:"stream"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type responseFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Chat sends one chat completion and returns the assistant message.
func (s *Server) Chat(ctx context.Context, messages []ChatMessage, temperature float64, maxTokens int) (string, error) {
	return s.chat(ctx, messages, temperature, maxTokens, nil)
}

// ChatJSON asks llama-server to constrain the reply to a JSON schema. If the
// server build does not support response_format, it retries without it.
func (s *Server) ChatJSON(ctx context.Context, messages []ChatMessage, schema map[string]any, temperature float64, maxTokens int) (string, error) {
	rf := &responseFormat{Type: "json_object", Schema: schema}
	content, status, err := s.chatStatus(ctx, messages, temperature, maxTokens, rf)
	if err != nil && status == 400 {
		return s.Chat(ctx, messages, temperature, maxTokens)
	}
	return content, err
}

func (s *Server) chat(ctx context.Context, messages []ChatMessage, temperature float64, maxTokens int, rf *responseFormat) (string, error) {
	content, _, err := s.chatStatus(ctx, messages, temperature, maxTokens, rf)
	return content, err
}

func (s *Server) chatStatus(ctx context.Context, messages []ChatMessage, temperature float64, maxTokens int, rf *responseFormat) (string, int, error) {
	body, err := json.Marshal(chatRequest{
		Messages:       messages,
		Temperature:    temperature,
		MaxTokens:      maxTokens,
		ResponseFormat: rf,
	})
	if err != nil {
		return "", 0, err
	}

	url := fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", s.port)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", resp.StatusCode, err
	}
	if resp.StatusCode != 200 {
		return "", resp.StatusCode, fmt.Errorf("chat http %d: %s", resp.StatusCode, truncate(string(data), 400))
	}

	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return "", resp.StatusCode, fmt.Errorf("decode response: %w", err)
	}
	if cr.Error != nil {
		return "", resp.StatusCode, fmt.Errorf("server error: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return "", resp.StatusCode, fmt.Errorf("empty choices")
	}
	return cr.Choices[0].Message.Content, resp.StatusCode, nil
}

// Kill stops the server process.
func (s *Server) Kill() {
	if s.Shutdown != nil {
		s.Shutdown()
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
