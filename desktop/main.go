// Command desktop is the MyGo native-UI shell for mangatrans. It runs the
// mangatrans.exe engine as a child process, streams its log, and reports
// progress, so the UI binary stays free of cgo and ONNX Runtime.
package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
)

type app struct {
	mu       sync.Mutex
	win      *mygo.Window
	input    string
	output   string
	busy     bool
	status   string
	progress float64
	log      []string
	cancel   context.CancelFunc
	total    int
}

func main() {
	a := &app{status: "选择一本漫画，然后点开始翻译。"}
	mygo.App.WhenReady(func() {
		a.win = mygo.NewWindow(mygo.WindowOptions{
			Title:   "漫画翻译器",
			Width:   820,
			Height:  640,
			Content: ui.View(a.view),
		})
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}

func (a *app) view(c *ui.Context) {
	a.mu.Lock()
	busy := a.busy
	status := a.status
	progress := a.progress
	lines := append([]string(nil), a.log...)
	a.mu.Unlock()

	ui.Column(c).Fill().Padding(20).Gap(12).Font("Microsoft YaHei UI, Microsoft YaHei, system-ui").Children(func() {
		ui.Text(c, "漫画翻译器").FontSize(24).Bold()
		ui.Text(c, "日文 CBZ / ZIP → 简体中文 CBZ。全部在本机 GPU 上完成。").FontSize(13).TextColor(ui.RGB(110, 110, 110))

		ui.Row(c).Gap(8).Children(func() {
			ui.Text(c, "输入").Width(56)
			ui.TextInput(c, &a.input).Grow(1)
			if ui.Button(c, "选择 CBZ").Clicked() {
				paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
					Title:   "选择漫画文件",
					Filters: []mygo.FileFilter{{Name: "Comic book", Extensions: []string{"cbz", "zip"}}},
				})
				if err == nil && len(paths) > 0 {
					a.input = paths[0]
					if a.output == "" {
						a.output = defaultOutput(a.input)
					}
				}
			}
		})

		ui.Row(c).Gap(8).Children(func() {
			ui.Text(c, "输出").Width(56)
			ui.TextInput(c, &a.output).Grow(1)
			if ui.Button(c, "另存为").Clicked() {
				out, err := mygo.Dialog.Save(mygo.SaveDialogOptions{
					Title:       "保存译文",
					DefaultPath: a.output,
					Filters:     []mygo.FileFilter{{Name: "Comic book", Extensions: []string{"cbz"}}},
				})
				if err == nil && out != "" {
					a.output = out
				}
			}
		})

		ui.Row(c).Gap(8).Children(func() {
			if ui.PrimaryButton(c, "开始翻译").Disabled(busy).Clicked() {
				a.start()
			}
			if ui.Button(c, "取消").Disabled(!busy).Clicked() {
				a.stop()
			}
			if ui.Button(c, "打开输出目录").Disabled(busy || a.output == "").Clicked() {
				openFolder(a.output)
			}
		})

		ui.Row(c).Gap(12).Children(func() {
			ui.Text(c, status).FontSize(13)
			if busy {
				ui.Spinner(c)
			}
		})
		ui.Progress(c, progress)

		ui.Text(c, "运行日志").FontSize(13).Bold()
		ui.Scroll(c).Grow(1).Children(func() {
			ui.Column(c).Gap(2).Children(func() {
				if len(lines) == 0 {
					ui.Text(c, "（尚无输出）").FontSize(12).TextColor(ui.RGB(140, 140, 140))
				}
				for _, line := range lines {
					ui.Text(c, line).FontSize(12)
				}
			})
		})
	})
}

func (a *app) start() {
	a.mu.Lock()
	if a.busy {
		a.mu.Unlock()
		return
	}
	in, out := strings.TrimSpace(a.input), strings.TrimSpace(a.output)
	if in == "" {
		a.status = "先选择输入文件。"
		a.mu.Unlock()
		a.invalidate()
		return
	}
	if out == "" {
		out = defaultOutput(in)
		a.output = out
	}
	engine := findEngine()
	if engine == "" {
		a.status = "找不到 mangatrans.exe，请把它放在本程序旁边。"
		a.mu.Unlock()
		a.invalidate()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.cancel = cancel
	a.busy = true
	a.status = "启动引擎…"
	a.progress = 0
	a.total = 0
	a.log = nil
	a.mu.Unlock()
	a.invalidate()

	go a.run(ctx, engine, in, out)
}

func (a *app) stop() {
	a.mu.Lock()
	cancel := a.cancel
	a.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (a *app) run(ctx context.Context, engine, in, out string) {
	cmd := exec.CommandContext(ctx, engine, "-in", in, "-out", out)
	cmd.Dir = filepath.Dir(engine)
	cmd.Stdout = io.Discard
	pipe, err := cmd.StderrPipe()
	if err != nil {
		a.finish(err)
		return
	}
	if err := cmd.Start(); err != nil {
		a.finish(err)
		return
	}
	scanner := bufio.NewScanner(pipe)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		a.handleLine(scanner.Text())
	}
	err = cmd.Wait()
	if ctx.Err() != nil {
		a.finish(context.Canceled)
		return
	}
	a.finish(err)
}

var (
	stampPrefix = regexp.MustCompile(`^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} `)
	pagesLine   = regexp.MustCompile(`^pages: (\d+)$`)
	pageLine    = regexp.MustCompile(`^page (\d+):`)
)

func (a *app) handleLine(raw string) {
	line := stampPrefix.ReplaceAllString(raw, "")
	if line == "" {
		return
	}
	a.mu.Lock()
	switch {
	case pagesLine.MatchString(line):
		fmt.Sscanf(line, "pages: %d", &a.total)
	case pageLine.MatchString(line):
		var idx int
		if _, err := fmt.Sscanf(line, "page %d:", &idx); err == nil && a.total > 0 {
			a.progress = float64(idx+1) / float64(a.total) * 0.9
		}
	case strings.Contains(line, "starting OCR server"):
		a.status = "加载 OCR 模型…"
	case strings.Contains(line, "OCR done"):
		a.status = "OCR 完成，加载翻译模型…"
	case strings.Contains(line, "starting translation server"):
		a.status = "加载翻译模型…"
	case strings.Contains(line, "translation done"):
		a.status = "翻译完成，擦除原文…"
		a.progress = 0.92
	case strings.Contains(line, "starting inpainting"):
		a.status = "擦除原文并排版…"
	case strings.Contains(line, "finish done"):
		a.status = "排版完成，写入 CBZ…"
		a.progress = 0.98
	case strings.HasPrefix(line, "wrote "):
		a.status = "完成"
		a.progress = 1
	}
	a.log = append(a.log, truncateLine(line, 220))
	if len(a.log) > 300 {
		a.log = a.log[len(a.log)-300:]
	}
	a.mu.Unlock()
	a.invalidate()
}

func truncateLine(s string, n int) string {	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func (a *app) finish(err error) {
	a.mu.Lock()
	a.busy = false
	a.cancel = nil
	switch {
	case err == nil:
		a.status = "完成"
		a.progress = 1
	case err == context.Canceled:
		a.status = "已取消"
	default:
		a.status = "失败：" + err.Error()
	}
	a.log = append(a.log, a.status)
	a.mu.Unlock()
	a.invalidate()
}

func (a *app) invalidate() {
	a.mu.Lock()
	win := a.win
	a.mu.Unlock()
	if win != nil {
		win.Invalidate()
	}
}

func defaultOutput(in string) string {
	ext := filepath.Ext(in)
	base := strings.TrimSuffix(in, ext)
	if ext == "" {
		ext = ".cbz"
	}
	return base + ".zh" + ext
}

func findEngine() string {
	if p := os.Getenv("MANGATRANS_EXE"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	exe, err := os.Executable()
	if err == nil {
		dir := filepath.Dir(exe)
		for _, p := range []string{
			filepath.Join(dir, "mangatrans.exe"),
			filepath.Join(dir, "..", "bin", "mangatrans.exe"),
		} {
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	if p, err := exec.LookPath("mangatrans.exe"); err == nil {
		return p
	}
	return ""
}

func openFolder(path string) {
	if path == "" {
		return
	}
	cmd := exec.Command("explorer.exe", "/select,", path)
	_ = cmd.Start()
}
