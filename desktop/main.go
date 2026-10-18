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
	"math"
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
	mu        sync.Mutex
	win       *mygo.Window
	input     string
	output    string
	busy      bool
	status    string
	progress  float64
	done      int
	total     int
	stage     int
	failed    bool
	lastErr   string
	log       []string
	logDirty  bool
	logFollow bool
	logView   ui.ScrollState
	cancel    context.CancelFunc
	cmd       *exec.Cmd
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
	if len(os.Args) > 2 && os.Args[1] == "-logtest" {
		if n, err := strconv.Atoi(os.Args[2]); err == nil {
			seedLog(a, n)
		}
	}
	mygo.App.WhenReady(func() {
		a.win = mygo.NewWindow(mygo.WindowOptions{
			Title:           "漫画翻译器",
			Width:           1040,
			Height:          740,
			MinWidth:        920,
			MinHeight:       660,
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

// seedLog pre-fills the log with n realistic lines for layout debugging:
// run with -logtest N to open the window in a full-log state without
// starting a translation.
func seedLog(a *app, n int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := 0; i < n; i++ {
		a.log = append(a.log, fmt.Sprintf("page %d: translated %d/%d blocks", i+1, 5+i%9, 9))
	}
	a.log = append(a.log, "warning: 13 text blocks kept the original text")
	a.logDirty = true
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
	stage := a.stage
	failed := a.failed
	lastErr := a.lastErr
	lines := append([]string(nil), a.log...)
	dirty := a.logDirty
	a.logDirty = false
	a.mu.Unlock()

	if a.logView.MaxY > 0 && a.logView.Y+32 < a.logView.MaxY {
		// The reader scrolled up to inspect an earlier line. Stop following
		// until they return to the bottom.
		a.logFollow = false
	} else if a.logView.Y >= a.logView.MaxY {
		// Back at the end: new lines follow again.
		a.logFollow = true
	}
	if dirty && (a.logFollow || a.logView.MaxY == 0) {
		a.logView.Y = math.MaxFloat32
		a.logView.X = 0
	}

	if busy {
		over = false
	} else if len(dropped) > 0 {
		a.acceptFiles(dropped)
		a.mu.Lock()
		status = a.status
		stage = a.stage
		failed = a.failed
		lastErr = a.lastErr
		a.mu.Unlock()
	}

	bar := c.TitleBar()
	headerH := float32(52)
	if bar.Height > 0 {
		headerH = bar.Height
	}

	root.Background(appBg).Children(func() {
		a.topBar(c, headerH, bar.Right)
		// Grow(1) lets this row shrink below its content's intrinsic height:
		// the log inside scrolls, and without it a full log pushes the cards
		// past the window edge, hiding the action row.
		ui.Row(c).Fill().Grow(1).Background(appBg).Padding(20).Gap(20).
			AlignItems(ui.Stretch).Children(func() {
			a.settingsCard(c, over, busy, status, progress, done, total, stage, failed, lastErr)
			a.logCard(c, lines, busy, failed)
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

func (a *app) settingsCard(c *ui.Context, over, busy bool, status string, progress float64, done, total, stage int, failed bool, lastErr string) {
	ui.Column(c).Basis(560).Grow(3).Shrink(1).MinWidth(0).Background(surface).
		Border(1, line).Radius(16).
		Shadow(0, 8, 24, 0, ui.RGBA(20, 20, 24, 0.06)).
		Padding(24).Gap(16).Children(func() {
		ui.Column(c).Gap(5).Children(func() {
			ui.Text(c, "漫画翻译").FontSize(22).Bold().TextColor(ink)
			ui.Text(c, "选择漫画和保存位置，然后开始翻译。").FontSize(13).
				TextColor(muted).SingleLine()
		})

		// The drop zone takes the free vertical space. The card never grows an
		// awkward empty gap, and the file target stays the visual focus.
		a.dropZone(c, over, busy)
		a.outputField(c, busy)

		ui.Row(c).Gap(10).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "翻译语言").Width(56).FontSize(11).TextColor(muted)
			ui.Row(c).Background(peach).Border(1, peachLn).Radius(99).
				Padding(4, 10).Children(func() {
				ui.Text(c, "日文 → 简体中文").FontSize(11).TextColor(peachText)
			})
		})

		a.stageStrip(c, busy, stage, failed)
		a.actionArea(c, busy, status, progress, done, total, failed, lastErr)
	})
}

func (a *app) dropZone(c *ui.Context, over, busy bool) {
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
		title = clip(filepath.Base(a.input), 30)
		sub = clip(filepath.Dir(a.input), 46)
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

	b := ui.Button(c, "").Grow(1).MinHeight(170).Padding(0).Radius(12).
		Background(bg).Border(1, border).Shadow(0, 1, 3, 0, shadow).Disabled(busy)
	if a.input != "" {
		b.Tooltip(a.input)
	}
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

func (a *app) outputField(c *ui.Context, busy bool) {
	ui.Column(c).Gap(7).MinWidth(0).Children(func() {
		ui.Text(c, "保存位置").FontSize(11).TextColor(muted)
		ui.Row(c).MinWidth(0).Background(surface).Border(1, line).Radius(10).
			Shadow(0, 1, 2, 0, ui.RGBA(20, 20, 24, 0.03)).
			Padding(5, 12).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.TextInput(c, &a.output).Grow(1).MinWidth(0).Background(surface).
				Border(0, ui.Color{}).Placeholder("自动生成文件名").ReadOnly(busy)
			if busy {
				return
			}
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

// stageStrip is a compact view of the four pipeline stages. It answers the
// question the bare progress bar cannot: what is happening now, and what is
// still ahead. A failed stage turns red instead of leaving a misleading
// nearly-full bar.
func (a *app) stageStrip(c *ui.Context, busy bool, stage int, failed bool) {
	labels := []string{"识别文字", "翻译", "擦除排版", "保存校验"}
	ui.Row(c).MinWidth(0).Gap(6).AlignItems(ui.Center).Children(func() {
		ui.Text(c, "流程").Width(32).FontSize(10).TextColor(muted)
		for i, label := range labels {
			done := stage >= 5 || stage > i+1
			active := (stage == i+1 || (stage == stageIdle && failed && i == 0)) && stage < 5
			bg := ui.RGB(247, 247, 249)
			border := innerLine
			fg := ui.RGB(166, 166, 176)
			switch {
			case failed && active:
				bg, border, fg = ui.RGB(253, 235, 232), ui.RGB(244, 199, 194), danger
			case done:
				bg, border, fg = ui.RGB(236, 249, 244), ui.RGB(201, 235, 223), ui.RGB(14, 122, 95)
			case active:
				bg, border, fg = peach, peachLn, peachText
			}
			ui.Row(c).Background(bg).Border(1, border).Radius(99).
				Padding(3, 8).Children(func() {
				ui.Text(c, label).FontSize(10).TextColor(fg)
			})
		}
		if busy {
			ui.Spacer(c).Grow(1)
			ui.Text(c, "运行中").FontSize(10).TextColor(accent)
		}
	})
}

func (a *app) actionArea(c *ui.Context, busy bool, status string, progress float64, done, total int, failed bool, lastErr string) {
	canStart := strings.TrimSpace(a.input) != ""
	barColor := accent
	textColor := muted
	switch {
	case failed:
		barColor, textColor = danger, danger
	case strings.HasPrefix(status, "翻译完成"):
		barColor, textColor = ok, ui.RGB(14, 122, 95)
	case strings.HasPrefix(status, "已取消"):
		barColor, textColor = warn, warn
	}

	ui.Column(c).Gap(10).Children(func() {
		if lastErr != "" {
			tone, bg, border, label := danger, ui.RGB(253, 242, 240), ui.RGB(246, 214, 209), "失败原因"
			if !failed {
				tone, bg, border, label = warn, ui.RGB(255, 249, 238), ui.RGB(244, 224, 188), "注意"
			}
			ui.Row(c).FillWidth().Background(bg).Border(1, border).Radius(8).
				Padding(7, 9).Children(func() {
				ui.Text(c, label+"："+clip(lastErr, 120)).FontSize(10).
					TextColor(tone)
			})
		}
		ui.Row(c).Height(4).Background(ui.RGB(238, 238, 241)).Radius(2).
			Children(func() {
				if progress > 0 {
					ui.Row(c).WidthPercent(float32(progress * 100)).Height(4).
						Background(barColor).Radius(2)
				}
			})
		ui.Row(c).MinWidth(0).Gap(10).AlignItems(ui.Center).Children(func() {
			dot := statusDot(status, busy)
			if failed {
				dot = danger
			}
			ui.Text(c, "●").FontSize(9).TextColor(dot)
			ui.Text(c, status).FontSize(11).TextColor(textColor).SingleLine()
			ui.Spacer(c).Grow(1)
			if total > 0 {
				ui.Text(c, fmt.Sprintf("%d / %d 页", done, total)).FontSize(11).TextColor(muted)
			}
			if busy || progress > 0 {
				ui.Text(c, fmt.Sprintf("%d%%", int(progress*100))).FontSize(11).TextColor(textColor)
			} else if failed {
				ui.Text(c, "失败").FontSize(11).TextColor(danger)
			}
			if busy {
				if ui.Button(c, "停止").Height(36).Padding(0, 18).Shrink(0).Clicked() {
					a.stop()
				}
				return
			}
			label := "开始翻译"
			if !canStart {
				label = "选择漫画"
			} else if failed {
				label = "重试"
			} else if strings.HasPrefix(status, "翻译完成") {
				label = "再次翻译"
			}
			if ui.PrimaryButton(c, label).Height(36).Padding(0, 18).Shrink(0).
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

func (a *app) logCard(c *ui.Context, lines []string, busy, failed bool) {
	ui.Column(c).Basis(300).Grow(1).Shrink(1).MaxWidth(430).MinWidth(0).
		Background(surface).Border(1, line).Radius(16).
		Shadow(0, 8, 24, 0, ui.RGBA(20, 20, 24, 0.06)).Children(func() {
		ui.Row(c).Padding(20, 20, 14, 20).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "运行日志").FontSize(12).Bold().TextColor(ink)
			if failed {
				ui.Row(c).Background(ui.RGB(253, 235, 232)).Border(1, ui.RGB(244, 199, 194)).
					Radius(99).Padding(2, 7).Children(func() {
					ui.Text(c, "失败").FontSize(9).TextColor(danger)
				})
			}
			ui.Spacer(c).Grow(1)
			meta := "就绪"
			if len(lines) > 0 {
				meta = fmt.Sprintf("%d 行", len(lines))
			} else if failed {
				meta = "失败"
			}
			ui.Text(c, meta).FontSize(10).TextColor(muted)
		})
		ui.Row(c).Height(1).Background(innerLine)
		ui.Column(c).Fill().Grow(1).MinHeight(0).Padding(12).Children(func() {
			inner := ui.Column(c).Fill().Grow(1).MinHeight(0).Background(ui.RGB(250, 250, 251)).
				Border(1, innerLine).Radius(10).Padding(14)
			if len(lines) == 0 {
				text := "暂无日志"
				sub := "开始翻译后会显示在这里。"
				if busy {
					text = "正在启动…"
					sub = "请稍候。"
				} else if failed {
					text = "没有日志输出"
					sub = "请查看左侧的失败原因。"
				}
				inner.AlignItems(ui.Center).Justify(ui.Center).Gap(5).Children(func() {
					ui.Text(c, text).FontSize(12).TextColor(muted)
					ui.Text(c, sub).FontSize(10).TextColor(ui.RGB(178, 178, 186))
				})
				return
			}
			inner.Children(func() {
				// ScrollBoth, not Scroll: the horizontal overflow belongs in
				// the scroll range, so long paths and stack traces stay
				// readable instead of being painted over the window edge.
				ui.ScrollBoth(c).Grow(1).FillWidth().TrackScroll(&a.logView).
					Children(func() {
						ui.Column(c).FillWidth().Gap(4).Children(func() {
							for _, line := range lines {
								// Break at any character, as terminals do: log
								// lines have no spaces to wrap at.
								ui.Text(c, line).NoWrap().FontSize(10.5).
									Font("Cascadia Mono, Consolas, Microsoft YaHei, monospace").
									TextColor(logColor(line))
							}
						})
					})
			})
		})
	})
}

func logColor(line string) ui.Color {
	switch {
	case strings.HasPrefix(line, "panic:"),
		strings.HasPrefix(line, "fatal:"),
		strings.HasPrefix(line, "错误："),
		strings.Contains(line, "failed"),
		strings.Contains(line, "失败"),
		strings.Contains(line, "do not decode"):
		return danger
	case strings.HasPrefix(line, "warning:"),
		strings.HasPrefix(line, "警告："),
		strings.Contains(line, "已取消"):
		return warn
	default:
		return logText
	}
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
	a.mu.Lock()
	busy := a.busy
	a.mu.Unlock()
	if busy {
		return
	}
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
	a.mu.Lock()
	busy := a.busy
	a.mu.Unlock()
	if busy {
		return
	}
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
			a.lastErr = a.status
			a.failed = true
		}
		a.mu.Unlock()
		a.invalidate()
	}
	return false
}

func (a *app) setInput(p string) {
	a.mu.Lock()
	if a.busy {
		a.mu.Unlock()
		return
	}
	oldDefault := ""
	if a.input != "" {
		oldDefault = defaultOutput(a.input)
	}
	a.input = p
	if a.output == "" || a.output == oldDefault {
		a.output = defaultOutput(p)
	}
	a.status = "准备就绪。"
	a.stage = stageIdle
	a.progress = 0
	a.done = 0
	a.total = 0
	a.failed = false
	a.lastErr = ""
	a.log = nil
	a.logDirty = true
	a.logFollow = true
	a.logView.X, a.logView.Y = 0, 0
	a.logView.MaxX, a.logView.MaxY = 0, 0
	a.mu.Unlock()
	a.invalidate()
}

func (a *app) start() {
	a.mu.Lock()
	if a.busy {
		a.mu.Unlock()
		return
	}
	fail := func(msg string) {
		a.status = msg
		a.lastErr = msg
		a.failed = true
		a.mu.Unlock()
		a.invalidate()
	}
	in, out := strings.TrimSpace(a.input), strings.TrimSpace(a.output)
	if in == "" {
		fail("先选择漫画文件。")
		return
	}
	if out == "" {
		out = defaultOutput(in)
	}
	inAbs, err := filepath.Abs(in)
	if err != nil {
		fail("无法读取输入路径。")
		return
	}
	outAbs, err := filepath.Abs(out)
	if err != nil {
		fail("无法读取保存路径。")
		return
	}
	info, err := os.Stat(inAbs)
	if err != nil || info.IsDir() {
		fail("输入文件不存在或无法读取。")
		return
	}
	ext := strings.ToLower(filepath.Ext(inAbs))
	if ext != ".cbz" && ext != ".zip" {
		fail("只支持 CBZ 和 ZIP 文件。")
		return
	}
	if strings.EqualFold(filepath.Clean(inAbs), filepath.Clean(outAbs)) {
		fail("保存路径不能和原文件相同。")
		return
	}
	outExt := strings.ToLower(filepath.Ext(outAbs))
	if outExt == "" {
		outAbs += ".cbz"
	} else if outExt != ".cbz" && outExt != ".zip" {
		fail("保存文件请使用 .cbz 后缀。")
		return
	}
	if err := os.MkdirAll(filepath.Dir(outAbs), 0o755); err != nil {
		fail("无法创建保存目录。")
		return
	}
	if st, err := os.Stat(outAbs); err == nil && st.IsDir() {
		fail("保存路径是一个文件夹，请选择文件。")
		return
	}
	// A forced kill can leave the atomic-write scratch file and the run log
	// directory behind. The app runs one job at a time, so clear both before
	// the next start. The engine repeats this defensively.
	_ = os.Remove(outAbs + ".part")
	_ = os.RemoveAll(filepath.Join(filepath.Dir(outAbs), "."+filepath.Base(outAbs)+".mangatrans"))
	engine := findEngine()
	if engine == "" {
		fail("找不到翻译引擎，请重新安装应用。")
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	a.cancel = cancel
	a.input, a.output = inAbs, outAbs
	a.busy = true
	a.cmd = nil
	a.status = "正在启动…"
	a.progress = 0
	a.done = 0
	a.total = 0
	a.stage = stageIdle
	a.failed = false
	a.lastErr = ""
	a.log = nil
	a.logDirty = true
	a.logFollow = true
	a.logView.X, a.logView.Y = 0, 0
	a.logView.MaxX, a.logView.MaxY = 0, 0
	a.mu.Unlock()
	a.invalidate()

	go a.run(ctx, engine, inAbs, outAbs)
}

func (a *app) stop() {
	a.mu.Lock()
	if !a.busy {
		a.mu.Unlock()
		return
	}
	parent := a.win
	a.mu.Unlock()

	// Do not hold the state lock while the native dialog runs: the UI still
	// rebuilds behind it. Re-check busy afterwards in case the job finished.
	res, err := mygo.Dialog.Message(mygo.MessageOptions{
		Parent:        parent,
		Type:          mygo.MessageWarning,
		Title:         "停止翻译？",
		Message:       "当前任务会立即结束，已经处理的内容不会保存。",
		Buttons:       []string{"继续翻译", "停止"},
		DefaultButton: 0,
		CancelButton:  0,
	})
	if err != nil || res.Button != 1 {
		return
	}

	a.mu.Lock()
	if !a.busy {
		a.mu.Unlock()
		return
	}
	a.status = "正在停止…"
	a.mu.Unlock()
	a.invalidate()
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
		_ = cmd.Wait()
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
	if scanErr := scanner.Err(); scanErr != nil {
		a.handleLine("warning: log read: " + scanErr.Error())
	}
	err = cmd.Wait()
	if ctx.Err() != nil {
		a.finish(context.Canceled)
		return
	}
	a.finish(err)
}

const (
	stageIdle      = 0
	stageOCR       = 1
	stageTranslate = 2
	stageInpaint   = 3
	stageSave      = 4
	stageDone      = 5
)

var (
	stampPrefix  = regexp.MustCompile(`^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} `)
	pagesLine    = regexp.MustCompile(`^pages: (\d+)$`)
	pageLine     = regexp.MustCompile(`^page (\d+):`)
	progressLine = regexp.MustCompile(`^progress (ocr|translate|inpaint|save|done) (\d+)/(\d+)$`)
)

func (a *app) handleLine(raw string) {
	line := stampPrefix.ReplaceAllString(raw, "")
	if line == "" {
		return
	}

	// Machine-readable events drive the bar. Human log text is only displayed;
	// the UI never guesses progress from English words in the log.
	if m := progressLine.FindStringSubmatch(line); m != nil {
		num, _ := strconv.Atoi(m[2])
		den, _ := strconv.Atoi(m[3])
		ratio := 0.0
		if den > 0 {
			ratio = float64(num) / float64(den)
		}
		a.mu.Lock()
		if den > 0 {
			a.total = den
		}
		a.done = num
		switch m[1] {
		case "ocr":
			a.stage = stageOCR
			a.progress = 0.03 + ratio*0.37
			a.status = "正在识别文字…"
		case "translate":
			a.stage = stageTranslate
			a.progress = 0.43 + ratio*0.25
			a.status = "正在翻译…"
		case "inpaint":
			a.stage = stageInpaint
			a.progress = 0.71 + ratio*0.25
			if num > 0 {
				a.status = "正在擦除与排版…"
			}
		case "save":
			a.stage = stageSave
			a.progress = 0.98
			a.status = "正在保存…"
		case "done":
			a.stage = stageDone
			a.progress = 1
			a.status = "翻译完成。"
		}
		a.mu.Unlock()
		a.invalidate()
		return
	}

	a.mu.Lock()
	switch {
	case pagesLine.MatchString(line):
		fmt.Sscanf(line, "pages: %d", &a.total)
		a.done, a.stage, a.progress = 0, stageIdle, 0
	case strings.HasPrefix(line, "panic:"),
		strings.HasPrefix(line, "fatal:"),
		strings.Contains(line, "failed"):
		// These must come before pageLine: a failed page also starts with
		// "page N:", and must not be mistaken for an OCR/translation update.
		a.lastErr = line
	case pageLine.MatchString(line):
		if strings.Contains(line, "translated") {
			a.stage = stageTranslate
			a.status = "正在翻译…"
		} else {
			a.stage = stageOCR
			a.status = "正在识别文字…"
		}
	case strings.Contains(line, "starting OCR server"):
		a.stage = stageOCR
		a.status = "正在加载识别模型…"
		if a.progress < 0.02 {
			a.progress = 0.02
		}
	case strings.Contains(line, "OCR done"):
		a.stage = stageTranslate
		a.status = "正在翻译…"
		if a.progress < 0.42 {
			a.progress = 0.42
		}
	case strings.Contains(line, "starting translation server"):
		a.stage = stageTranslate
		a.status = "正在加载翻译模型…"
		if a.progress < 0.43 {
			a.progress = 0.43
		}
	case strings.Contains(line, "translation done"):
		a.stage = stageInpaint
		a.status = "正在准备擦除模型…"
		if a.progress < 0.70 {
			a.progress = 0.70
		}
	case strings.Contains(line, "starting inpainting"):
		a.stage = stageInpaint
		a.status = "正在加载擦除模型…"
		if a.progress < 0.71 {
			a.progress = 0.71
		}
	case strings.Contains(line, "finish done"):
		a.stage = stageSave
		a.status = "正在保存…"
		if a.progress < 0.97 {
			a.progress = 0.97
		}
	case strings.HasPrefix(line, "wrote "):
		a.stage = stageSave
		a.status = "正在校验输出…"
		if a.progress < 0.98 {
			a.progress = 0.98
		}
	case strings.Contains(line, "validate output:"):
		if strings.Contains(line, "all decode") {
			a.stage = stageDone
			a.status = "翻译完成。"
			a.progress = 1
		} else {
			a.lastErr = line
		}
	}
	a.log = append(a.log, truncateLine(line, 600))
	if len(a.log) > 500 {
		a.log = a.log[len(a.log)-500:]
	}
	a.logDirty = true
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
		a.stage = stageDone
		a.progress = 1
		a.failed = false
		if a.lastErr != "" {
			a.status = "翻译完成，部分页面已保留原图。"
		} else {
			a.status = "翻译完成。"
		}
	case err == context.Canceled:
		a.failed = false
		a.lastErr = ""
		a.status = "已取消。"
	default:
		a.failed = true
		a.status = "翻译失败。"
		if a.lastErr == "" {
			a.lastErr = err.Error()
		}
		a.log = append(a.log, "错误："+err.Error())
	}
	a.log = append(a.log, a.status)
	if len(a.log) > 500 {
		a.log = a.log[len(a.log)-500:]
	}
	a.logDirty = true
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
