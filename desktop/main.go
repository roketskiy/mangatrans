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
	"strconv"
	"strings"
	"sync"
	"syscall"

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
	done     int
	log      []string
	cancel   context.CancelFunc
	cmd      *exec.Cmd
	total    int
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "-job-selftest" {
		jobSelfTest()
		return
	}
	if err := initKillJob(); err != nil {
		log.Printf("job object: %v", err)
	}

	a := &app{status: "选择文件后开始。"}
	mygo.App.WhenReady(func() {
		a.win = mygo.NewWindow(mygo.WindowOptions{
			Title:           "漫画翻译器",
			Width:           1000,
			Height:          720,
			MinWidth:        900,
			MinHeight:       640,
			TitleBarStyle:   mygo.TitleBarHidden,
			TitleBarHeight:  52,
			BackgroundColor: "#F4F4F5",
			Content:         ui.View(a.view),
		})
		a.win.OnFileDrop(func(e *mygo.FileDropEvent) {
			a.mu.Lock()
			busy := a.busy
			a.mu.Unlock()
			if busy {
				return
			}
			a.acceptFiles(e.Paths)
			a.invalidate()
		})
		a.win.OnClose(func(e *mygo.CloseEvent) {
			a.shutdownBackend()
		})
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}

func (a *app) view(c *ui.Context) {
	c.SetTheme(codexTheme())

	root := ui.Column(c).Fill()
	over := root.FileDragOver()
	dropped := root.DroppedFiles()

	a.mu.Lock()
	busy := a.busy
	status := a.status
	progress := a.progress
	done := a.done
	total := a.total
	lines := append([]string(nil), a.log...)
	a.mu.Unlock()

	if busy {
		over = false
	} else if len(dropped) > 0 {
		a.acceptFiles(dropped)
		a.mu.Lock()
		status = a.status
		a.mu.Unlock()
	}

	bar := c.TitleBar()
	headerH := float32(52)
	if bar.Height > 0 {
		headerH = bar.Height
	}

	root.Background(appBg).Children(func() {
		a.topBar(c, headerH, bar.Right)
		ui.Row(c).Fill().Background(appBg).Padding(20).Gap(20).
			AlignItems(ui.Stretch).Children(func() {
			a.settingsCard(c, over, busy, status, progress, done, total)
			a.logCard(c, lines, busy)
		})
	})
}

func (a *app) topBar(c *ui.Context, height, right float32) {
	ui.Row(c).Height(height).Background(surface).Padding(0, 18).
		PaddingEnd(right + 12).Gap(9).AlignItems(ui.Center).
		DragWindow().Children(func() {
		ui.Column(c).Width(22).Height(22).Background(ink).Radius(7).
			AlignItems(ui.Center).Justify(ui.Center).Children(func() {
			ui.Text(c, "译").FontSize(11).Bold().TextColor(surface)
		})
		ui.Text(c, "漫画翻译器").FontSize(14).Bold().TextColor(ink)
	})
	ui.Row(c).Height(1).Background(line)
}

func (a *app) settingsCard(c *ui.Context, over, busy bool, status string, progress float64, done, total int) {
	ui.Column(c).Grow(1).Background(surface).Border(1, line).Radius(16).
		Shadow(0, 8, 24, 0, ui.RGBA(20, 20, 24, 0.06)).
		Padding(24).Gap(18).Children(func() {
		ui.Column(c).Gap(5).Children(func() {
			ui.Text(c, "漫画翻译").FontSize(22).Bold().TextColor(ink)
			ui.Text(c, "选择漫画和保存位置，然后开始翻译。").FontSize(13).
				TextColor(muted).SingleLine()
		})

		a.dropZone(c, over)
		a.outputField(c)

		ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "翻译语言").Width(56).FontSize(11).TextColor(muted)
			ui.Row(c).Background(peach).Border(1, peachLn).Radius(99).
				Padding(4, 10).Children(func() {
				ui.Text(c, "日文 → 简体中文").FontSize(11).TextColor(peachText)
			})
		})

		ui.Spacer(c).Grow(1)
		a.actionArea(c, busy, status, progress, done, total)
	})
}

func (a *app) dropZone(c *ui.Context, over bool) {
	bg, border := ui.RGB(250, 250, 251), line
	tagBg, tagBorder, tagColor := surface, line, muted
	title, sub := "拖入 CBZ 或 ZIP", "或点击选择文件"
	meta := "支持 CBZ / ZIP"
	metaBg, metaColor := ui.RGB(242, 242, 244), muted
	tag := "+"
	tagSize := float32(18)
	shadow := ui.RGBA(20, 20, 24, 0.04)

	if a.input != "" {
		bg, border = peach, peachLn
		tagBg, tagBorder, tagColor = surface, peachLn, accent
		title = clip(filepath.Base(a.input), 44)
		sub = clip(filepath.Dir(a.input), 60)
		tag = strings.ToUpper(strings.TrimPrefix(filepath.Ext(a.input), "."))
		if tag == "" {
			tag = "CBZ"
		}
		tagSize = 9
		meta = tag + " · " + fileSize(a.input)
		metaBg = surface
		metaColor = muted
	}
	if over {
		bg, border = ui.RGB(255, 247, 242), accent
		tagBg, tagBorder, tagColor = surface, accent, accent
		title, sub, tag = "松开即可导入", "支持 CBZ 和 ZIP", "+"
		tagSize = 18
		meta = "松开即可导入"
		metaBg = surface
		metaColor = accent
		shadow = ui.RGBA(232, 106, 51, 0.12)
	}

	b := ui.Button(c, "").Height(250).Padding(0).Radius(12).
		Background(bg).Border(1, border).Shadow(0, 1, 3, 0, shadow)
	b.Justify(ui.Center).AlignItems(ui.Center).Children(func() {
		ui.Column(c).AlignItems(ui.Center).Gap(7).Children(func() {
			ui.Column(c).Width(40).Height(40).Background(tagBg).Border(1, tagBorder).
				Radius(11).AlignItems(ui.Center).Justify(ui.Center).Children(func() {
				ui.Text(c, tag).FontSize(tagSize).Bold().TextColor(tagColor)
			})
			ui.Text(c, title).FontSize(16).Bold().TextColor(ink).SingleLine()
			ui.Text(c, sub).FontSize(11).TextColor(muted).SingleLine()
			ui.Row(c).Background(metaBg).Border(1, border).Radius(99).
				Padding(3, 9).Children(func() {
				ui.Text(c, meta).FontSize(10).TextColor(metaColor).SingleLine()
			})
		})
	})
	if b.Clicked() {
		a.chooseFile()
	}
}

func (a *app) outputField(c *ui.Context) {
	ui.Column(c).Gap(7).Children(func() {
		ui.Text(c, "保存位置").FontSize(11).TextColor(muted)
		ui.Row(c).Background(surface).Border(1, line).Radius(10).
			Shadow(0, 1, 2, 0, ui.RGBA(20, 20, 24, 0.03)).
			Padding(5, 12).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.TextInput(c, &a.output).Grow(1).Background(surface).
				Border(0, ui.Color{}).Placeholder("自动生成文件名")
			if strings.TrimSpace(a.output) != "" {
				if ui.Button(c, "打开").Background(surface).Border(0, ui.Color{}).
					Padding(2, 8).Clicked() {
					openFolder(a.output)
				}
			}
			if ui.Button(c, "选择").Background(surface).Border(0, ui.Color{}).
				Padding(2, 8).Clicked() {
				a.chooseSave()
			}
		})
	})
}

func (a *app) actionArea(c *ui.Context, busy bool, status string, progress float64, done, total int) {
	canStart := strings.TrimSpace(a.input) != ""
	ui.Column(c).Gap(12).Children(func() {
		ui.Row(c).Height(4).Background(ui.RGB(238, 238, 241)).Radius(2).
			Children(func() {
				if progress > 0 {
					ui.Row(c).WidthPercent(float32(progress * 100)).Height(4).
						Background(accent).Radius(2)
				}
			})
		ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "●").FontSize(9).TextColor(statusDot(status, busy))
			ui.Text(c, status).FontSize(11).TextColor(muted).SingleLine()
			ui.Spacer(c).Grow(1)
			if total > 0 {
				ui.Text(c, fmt.Sprintf("%d / %d 页", done, total)).FontSize(11).TextColor(muted)
			}
			if busy || progress > 0 {
				ui.Text(c, fmt.Sprintf("%d%%", int(progress*100))).FontSize(11).TextColor(muted)
			}
			if busy {
				if ui.Button(c, "停止").Height(36).Padding(0, 18).Clicked() {
					a.stop()
				}
				return
			}
			label := "开始翻译"
			if !canStart {
				label = "选择漫画"
			}
			if ui.PrimaryButton(c, label).Height(36).Padding(0, 18).
				Shadow(0, 3, 10, 0, ui.RGBA(232, 106, 51, 0.24)).
				Clicked() {
				if canStart {
					a.start()
				} else {
					a.chooseFile()
				}
			}
		})
	})
}

func (a *app) logCard(c *ui.Context, lines []string, busy bool) {
	ui.Column(c).Width(360).Background(surface).Border(1, line).Radius(16).
		Shadow(0, 8, 24, 0, ui.RGBA(20, 20, 24, 0.06)).Children(func() {
		ui.Row(c).Padding(20, 20, 14, 20).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "运行日志").FontSize(12).Bold().TextColor(ink)
			ui.Spacer(c).Grow(1)
			meta := "就绪"
			if len(lines) > 0 {
				meta = fmt.Sprintf("%d 行", len(lines))
			}
			ui.Text(c, meta).FontSize(10).TextColor(muted)
		})
		ui.Row(c).Height(1).Background(innerLine)
		ui.Column(c).Fill().Padding(12).Children(func() {
			inner := ui.Column(c).Fill().Background(ui.RGB(250, 250, 251)).
				Border(1, innerLine).Radius(10).Padding(14)
			if len(lines) == 0 {
				text := "暂无日志"
				sub := "开始翻译后会显示在这里。"
				if busy {
					text = "正在启动…"
					sub = "请稍候。"
				}
				inner.AlignItems(ui.Center).Justify(ui.Center).Gap(5).Children(func() {
					ui.Text(c, text).FontSize(12).TextColor(muted)
					ui.Text(c, sub).FontSize(10).TextColor(ui.RGB(178, 178, 186))
				})
				return
			}
			inner.Children(func() {
				ui.Scroll(c).Grow(1).Children(func() {
					ui.Column(c).Gap(3).Children(func() {
						for _, line := range lines {
							ui.Text(c, clip(line, 140)).FontSize(11).
								Font("Cascadia Mono, Consolas, Microsoft YaHei, monospace").
								TextColor(logText).SingleLine()
						}
					})
				})
			})
		})
	})
}

var (
	appBg     = ui.RGB(244, 244, 245)
	surface   = ui.RGB(255, 255, 255)
	ink       = ui.RGB(32, 33, 35)
	muted     = ui.RGB(142, 142, 160)
	line      = ui.RGB(229, 229, 232)
	innerLine = ui.RGB(239, 239, 242)
	accent    = ui.RGB(232, 106, 51)
	peach     = ui.RGB(253, 244, 238)
	peachLn   = ui.RGB(240, 216, 202)
	peachText = ui.RGB(166, 76, 40)
	ok        = ui.RGB(16, 163, 127)
	danger    = ui.RGB(217, 45, 32)
	warn      = ui.RGB(181, 71, 8)
	logText   = ui.RGB(58, 59, 67)
)

func codexTheme() *ui.Theme {
	t := ui.LightTheme()
	t.Background = appBg
	t.Surface = surface
	t.SurfaceHover = ui.RGB(246, 246, 247)
	t.SurfacePressed = ui.RGB(236, 236, 238)
	t.Border = line
	t.Text = ink
	t.TextMuted = muted
	t.Accent = accent
	t.AccentHover = ui.RGB(217, 95, 43)
	t.AccentPressed = ui.RGB(197, 82, 36)
	t.AccentText = surface
	t.Danger = danger
	t.Success = ok
	t.Selection = ui.RGBA(232, 106, 51, 0.16)
	t.Focus = accent
	t.Scrollbar = ui.RGB(199, 199, 204)
	t.Radius = 8
	t.Spacing = 4
	t.Font = "Segoe UI, Microsoft YaHei UI, Microsoft YaHei, system-ui"
	t.FontSize = 13
	return t
}

func statusDot(status string, busy bool) ui.Color {
	switch {
	case busy:
		return accent
	case strings.HasPrefix(status, "翻译完成"):
		return ok
	case strings.HasPrefix(status, "翻译失败") || strings.HasPrefix(status, "失败"):
		return danger
	case strings.HasPrefix(status, "已取消"):
		return warn
	case strings.HasPrefix(status, "准备就绪"):
		return ok
	default:
		return ui.RGB(199, 199, 204)
	}
}

func clip(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func fileSize(path string) string {
	info, err := os.Stat(path)
	if err != nil {
		return ""
	}
	b := info.Size()
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(b)/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.0f KB", float64(b)/(1<<10))
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func (a *app) chooseFile() {
	paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
		Title:   "选择漫画文件",
		Filters: []mygo.FileFilter{{Name: "漫画文件", Extensions: []string{"cbz", "zip"}}},
	})
	if err != nil || len(paths) == 0 {
		return
	}
	a.setInput(paths[0])
}

func (a *app) chooseSave() {
	out, err := mygo.Dialog.Save(mygo.SaveDialogOptions{
		Title:       "保存译文",
		DefaultPath: a.output,
		Filters:     []mygo.FileFilter{{Name: "漫画文件", Extensions: []string{"cbz"}}},
	})
	if err != nil || out == "" {
		return
	}
	a.mu.Lock()
	a.output = out
	a.mu.Unlock()
	a.invalidate()
}

func (a *app) acceptFiles(paths []string) bool {
	for _, p := range paths {
		ext := strings.ToLower(filepath.Ext(p))
		if ext == ".cbz" || ext == ".zip" {
			a.setInput(p)
			return true
		}
	}
	if len(paths) > 0 {
		a.mu.Lock()
		if !a.busy {
			a.status = "只支持 CBZ 和 ZIP 文件。"
		}
		a.mu.Unlock()
	}
	return false
}

func (a *app) setInput(p string) {
	a.mu.Lock()
	oldDefault := ""
	if a.input != "" {
		oldDefault = defaultOutput(a.input)
	}
	a.input = p
	if a.output == "" || a.output == oldDefault {
		a.output = defaultOutput(p)
	}
	if !a.busy {
		a.status = "准备就绪。"
	}
	a.mu.Unlock()
	a.invalidate()
}

func (a *app) start() {
	a.mu.Lock()
	if a.busy {
		a.mu.Unlock()
		return
	}
	in, out := strings.TrimSpace(a.input), strings.TrimSpace(a.output)
	if in == "" {
		a.status = "先选择漫画文件。"
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
		a.status = "找不到翻译引擎，请重新安装应用。"
		a.mu.Unlock()
		a.invalidate()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	a.cancel = cancel
	a.busy = true
	a.cmd = nil
	a.status = "正在启动…"
	a.progress = 0
	a.done = 0
	a.total = 0
	a.log = nil
	a.mu.Unlock()
	a.invalidate()

	go a.run(ctx, engine, in, out)
}

func (a *app) stop() {
	a.shutdownBackend()
}

func (a *app) shutdownBackend() {
	a.mu.Lock()
	cmd := a.cmd
	cancel := a.cancel
	a.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	_ = terminateKillJob()
	if cmd != nil && cmd.Process != nil {
		killTree(cmd.Process.Pid)
		_ = cmd.Process.Kill()
	}
}

func killTree(pid int) {
	cmd := exec.Command("taskkill", "/PID", strconv.Itoa(pid), "/T", "/F")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: 0x08000000, // CREATE_NO_WINDOW
	}
	_ = cmd.Run()
}

func (a *app) run(ctx context.Context, engine, in, out string) {
	cmd := exec.Command(engine, "-in", in, "-out", out)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		HideWindow:    true,
		CreationFlags: 0x08000000, // CREATE_NO_WINDOW
	}
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
	if err := assignKillJob(cmd.Process.Pid); err != nil {
		log.Printf("assign job: %v", err)
	}
	if ctx.Err() != nil {
		_ = cmd.Process.Kill()
		a.finish(context.Canceled)
		return
	}
	a.mu.Lock()
	a.cmd = cmd
	a.mu.Unlock()
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
			a.done = idx + 1
			a.progress = float64(idx+1) / float64(a.total) * 0.9
		}
	case strings.Contains(line, "starting OCR server"):
		a.status = "正在识别文字…"
	case strings.Contains(line, "OCR done"):
		a.status = "正在翻译…"
	case strings.Contains(line, "starting translation server"):
		a.status = "正在翻译…"
	case strings.Contains(line, "translation done"):
		a.status = "正在处理画面…"
		a.progress = 0.92
	case strings.Contains(line, "starting inpainting"):
		a.status = "正在排版…"
	case strings.Contains(line, "finish done"):
		a.status = "正在保存…"
		a.progress = 0.98
	case strings.HasPrefix(line, "wrote "):
		a.status = "翻译完成。"
		a.progress = 1
	}
	a.log = append(a.log, truncateLine(line, 220))
	if len(a.log) > 300 {
		a.log = a.log[len(a.log)-300:]
	}
	a.mu.Unlock()
	a.invalidate()
}

func truncateLine(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func (a *app) finish(err error) {
	_ = terminateKillJob()
	a.mu.Lock()
	a.busy = false
	a.cancel = nil
	a.cmd = nil
	switch {
	case err == nil:
		a.status = "翻译完成。"
		a.progress = 1
	case err == context.Canceled:
		a.status = "已取消。"
	default:
		a.status = "翻译失败，请查看日志。"
		a.log = append(a.log, "错误："+err.Error())
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
	if _, err := os.Stat(path); err == nil {
		_ = exec.Command("explorer.exe", "/select,", path).Start()
		return
	}
	_ = exec.Command("explorer.exe", filepath.Dir(path)).Start()
}
