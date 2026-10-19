package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"mangatrans/settings"
)

type appPage uint8

const (
	pageHome appPage = iota
	pageSettings
)

type settingsCategory uint8

const (
	categoryProcessing settingsCategory = iota
	categoryTranslation
	categoryRepair
	categoryLayout
	categoryOutput
)

var categories = [...]struct{ title, description string }{
	{"处理与速度", "复用识别和翻译结果，减少重复处理。"},
	{"识别与翻译", "调整文字检测、识别和异常译文重试。"},
	{"擦字与修复", "控制残字清除范围和背景修复。"},
	{"字体与排版", "设置译文方向、字体和气泡内的排版。"},
	{"输出与存储", "设置漫画包内的图片格式及附加文件。"},
}

type settingOption[T comparable] struct {
	value T
	label string
}

func (a *app) loadSettings() {
	path, err := settings.DefaultPath()
	if err == nil {
		a.settingsPath = path
		a.settings, err = settings.Load(path)
	}
	if err != nil {
		a.settings = settings.Default()
		a.settingsLoadFailed = true
		a.settingsError = "无法读取设置，原文件未改动。" + err.Error()
		a.status = "设置读取失败，请进入设置检查。"
	}
	a.editor = a.settings
	a.lastSettingsAttempt = a.editor
}

func (a *app) openSettings() {
	a.mu.Lock()
	a.editor = a.settings
	a.integerInputs = make(map[*int]*float64)
	a.mu.Unlock()
	a.page = pageSettings
}

func (a *app) persistSettings() {
	if err := a.editor.Validate(); err != nil {
		a.settingsError = "设置尚未保存。" + err.Error()
		return
	}
	if a.settingsPath == "" {
		a.settingsError = "无法确定设置保存位置。"
		return
	}
	if a.settingsLoadFailed {
		result, err := mygo.Dialog.Message(mygo.MessageOptions{
			Parent: a.win, Type: mygo.MessageWarning, Title: "替换无法读取的设置？",
			Message: "原设置文件无法读取。保存将用当前设置替换它。",
			Buttons: []string{"取消", "替换并保存"}, DefaultButton: 0, CancelButton: 0,
		})
		if err != nil || result.Button != 1 {
			a.editor = a.settings
			return
		}
	}
	if err := settings.Save(a.settingsPath, a.editor); err != nil {
		a.settingsError = "保存失败。" + err.Error()
		return
	}
	a.mu.Lock()
	a.settings = a.editor
	a.mu.Unlock()
	a.settingsError = ""
	a.settingsLoadFailed = false
}

func (a *app) settingsView(c *ui.Context, busy bool) {
	ui.Row(c).Fill().Grow(1).MinHeight(0).Padding(20).Gap(18).Children(func() {
		ui.Column(c).Width(178).FillHeight().Shrink(0).Gap(3).Justify(ui.Start).Children(func() {
			ui.Text(c, "设置").FontSize(17).Bold().TextColor(ink).Padding(4, 12, 14, 12)
			for i, category := range categories {
				item := ui.Button(c, category.title).FillWidth().Height(38).
					Padding(0, 12).Justify(ui.Start).TextAlign(ui.Start).
					Border(0, appBg).Background(appBg).TextColor(logText).Radius(7)
				if a.category == settingsCategory(i) {
					item.Background(peach).TextColor(peachText).Bold()
				}
				if item.Clicked() {
					a.category = settingsCategory(i)
				}
			}
			ui.Spacer(c).Grow(1)
		})
		ui.Column(c).Fill().Grow(1).MinWidth(0).MinHeight(0).Background(surface).
			Border(1, line).Radius(16).Children(func() {
			ui.Column(c).Padding(24, 24, 16, 24).Gap(6).Children(func() {
				ui.Text(c, categories[a.category].title).FontSize(20).Bold().TextColor(ink)
				ui.Text(c, categories[a.category].description).FontSize(12).TextColor(muted)
				if busy {
					ui.Text(c, "任务正在运行，设置暂时锁定。").FontSize(12).TextColor(warn)
				}
				if a.settingsError != "" {
					ui.Text(c, a.settingsError).FontSize(11).TextColor(danger)
					if ui.Button(c, "重试保存").Height(30).Disabled(busy).Clicked() {
						a.persistSettings()
					}
				}
			})
			ui.Row(c).Height(1).Background(innerLine)
			ui.Scroll(c).FillWidth().Grow(1).MinHeight(0).TrackScroll(&a.settingsScroll[a.category]).Children(func() {
				ui.Column(c).FillWidth().Padding(8, 24, 20, 24).Children(func() {
					switch a.category {
					case categoryProcessing:
						a.processingSettings(c, busy)
					case categoryTranslation:
						a.translationSettings(c, busy)
					case categoryRepair:
						a.repairSettings(c, busy)
					case categoryLayout:
						a.layoutSettings(c, busy)
					case categoryOutput:
						a.outputSettings(c, busy)
					}
				})
			})
		})
	})
	if a.editor != a.settings && a.editor != a.lastSettingsAttempt && !busy {
		a.lastSettingsAttempt = a.editor
		a.persistSettings()
	}
}

func settingRow(c *ui.Context, title, description string, control func()) {
	ui.Row(c).FillWidth().Padding(16, 0).Gap(18).AlignItems(ui.Center).Children(func() {
		ui.Column(c).Grow(1).Shrink(1).MinWidth(0).Gap(5).Children(func() {
			ui.Text(c, title).FontSize(13).TextColor(ink)
			ui.Text(c, description).FontSize(11).TextColor(muted)
		})
		ui.Column(c).Width(180).Shrink(0).Children(control)
	})
	ui.Row(c).Height(1).Background(innerLine)
}

func settingSelect[T comparable](c *ui.Context, value *T, options []settingOption[T], disabled bool) {
	label := ""
	for _, option := range options {
		if option.value == *value {
			label = option.label
		}
	}
	if label == "" {
		label = fmt.Sprint(*value)
	}
	sel := ui.SelectBase(c, value)
	sel.Trigger.FillWidth().Height(34).Padding(6, 10).Background(surface).
		Border(1, line).Radius(7).Disabled(disabled).Children(func() {
		ui.Row(c).FillWidth().AlignItems(ui.Center).Children(func() {
			ui.Text(c, label).FontSize(12).TextColor(ink)
			ui.Spacer(c).Grow(1)
			ui.Text(c, "⌄").FontSize(12).TextColor(muted)
		})
	})
	sel.Popup(func(panel ui.Element) {
		panel.Padding(4).Background(surface).Border(1, line).Radius(7)
		for _, option := range options {
			item := sel.Item(option.value).Padding(8, 10)
			if item.Highlighted() {
				item.Background(peach)
			}
			item.Children(func() { ui.Text(c, option.label).FontSize(12).TextColor(ink) })
		}
	})
}

func settingNumber(c *ui.Context, value *float64, lo, hi, step, defaultValue float64, disabled bool) bool {
	changed := false
	ui.Row(c).FillWidth().Gap(6).AlignItems(ui.Center).Children(func() {
		changed = ui.NumberInput(c, value, lo, hi, step).Grow(1).MinWidth(0).Height(34).Disabled(disabled).Changed()
		if *value == defaultValue {
			ui.Text(c, "默认").FontSize(10).TextColor(muted).Shrink(0)
		}
	})
	return changed
}

func (a *app) settingInt(c *ui.Context, value *int, lo, hi float64, defaultValue int, disabled bool) {
	if a.integerInputs == nil {
		a.integerInputs = make(map[*int]*float64)
	}
	n := a.integerInputs[value]
	if n == nil {
		n = new(float64)
		*n = float64(*value)
		a.integerInputs[value] = n
	}
	settingNumber(c, n, lo, hi, 1, float64(defaultValue), disabled)
	*value = int(*n)
}

func (a *app) processingSettings(c *ui.Context, busy bool) {
	d := settings.Default()
	settingRow(c, "复用已有处理结果", "修改排版或输出时，复用匹配的识别和翻译结果。", func() {
		ui.Switch(c, &a.editor.Processing.ReuseResults).Disabled(busy)
	})
	ui.Text(c, "高级").FontSize(11).Bold().TextColor(muted).Padding(18, 0, 0, 0)
	settingRow(c, "翻译批次页数", "连续页共享翻译批次的上限，不代表完整上下文记忆。", func() {
		a.settingInt(c, &a.editor.Processing.BatchPages, 1, 4, d.Processing.BatchPages, busy)
	})
}

func (a *app) translationSettings(c *ui.Context, busy bool) {
	d := settings.Default()
	settingRow(c, "检测置信度阈值", "0.05～0.95。越低越容易检出文字，也越容易误检。", func() {
		settingNumber(c, &a.editor.Recognition.Confidence, 0.05, 0.95, 0.05, d.Recognition.Confidence, busy)
	})
	settingRow(c, "检测补充分块（px）", "0 为仅整页检测。256～4096 为原图补检块尺寸，可能增加耗时。", func() {
		a.settingInt(c, &a.editor.Recognition.TileSize, 0, 4096, d.Recognition.TileSize, busy)
	})
	settingRow(c, "批量翻译最大尝试次数", "1～5，包含首次请求。缺失译文仍会单独补译，失败时保留原文。", func() {
		a.settingInt(c, &a.editor.Translation.BatchAttempts, 1, 5, d.Translation.BatchAttempts, busy)
	})
	a.pathSetting(c, "翻译模型", "选择兼容的本地 GGUF 模型。留空时使用内置模型。", &a.editor.Translation.ModelPath, []string{"gguf"}, busy)
}

func (a *app) repairSettings(c *ui.Context, busy bool) {
	d := settings.Default()
	settingRow(c, "擦字扩展（px）", "0～24。扩大文字掩码可减少残字，也可能擦除邻近线稿。", func() {
		a.settingInt(c, &a.editor.Repair.MaskDilation, 0, 24, d.Repair.MaskDilation, busy)
	})
	settingRow(c, "修复上下文（px）", "16～256。修复区域周围用于参考背景的像素范围。", func() {
		a.settingInt(c, &a.editor.Repair.ContextPadding, 16, 256, d.Repair.ContextPadding, busy)
	})
	settingRow(c, "修复分块尺寸（px）", "0 为按连通区域处理。128～2048 为原图窗口尺寸，不是模型输入尺寸。", func() {
		a.settingInt(c, &a.editor.Repair.TileSize, 0, 2048, d.Repair.TileSize, busy)
	})
	ui.Text(c, "擦字或排版失败时保留原图，不在未清理的原图上叠字。").FontSize(11).TextColor(muted).Padding(18, 0, 0, 0)
}

func (a *app) layoutSettings(c *ui.Context, busy bool) {
	d := settings.Default()
	settingRow(c, "文字方向", "自动模式根据原文方向选择横排或竖排。", func() {
		settingSelect(c, &a.editor.Layout.Direction, []settingOption[string]{{"vertical", "竖排"}, {"horizontal", "横排"}, {"auto", "自动"}}, busy)
	})
	a.pathSetting(c, "译文字体", "支持 TTF、OTF 和 TTC 字体文件。", &a.editor.Layout.FontPath, []string{"ttf", "otf", "ttc"}, busy)
	settingRow(c, "字号调整", "在气泡边界允许的范围内调整，不放大文字框。", func() {
		settingNumber(c, &a.editor.Layout.FontScale, 0.7, 1.4, 0.05, d.Layout.FontScale, busy)
	})
	settingRow(c, "最小字号", "0 为自动。气泡放不下时仍优先遵守边界，单位为 pt。", func() {
		settingNumber(c, &a.editor.Layout.MinFontSize, 0, 72, 0.5, d.Layout.MinFontSize, busy)
	})
	settingRow(c, "行列间距", "1 为标准间距，较大间距会减少可容纳的文字。", func() {
		settingNumber(c, &a.editor.Layout.Spacing, 0.5, 2, 0.1, d.Layout.Spacing, busy)
	})
	settingRow(c, "文字颜色", "auto 为自动黑白，也可输入 #RRGGBB，例如 #202020。", func() {
		ui.TextInput(c, &a.editor.Layout.Color).FillWidth().Height(34).Disabled(busy)
	})
	settingRow(c, "描边宽度（px）", "-1 为自动，0 为关闭，正数为实际描边宽度，上限 4。", func() {
		settingNumber(c, &a.editor.Layout.StrokeWidth, -1, 4, 0.5, d.Layout.StrokeWidth, busy)
	})
}

func (a *app) outputSettings(c *ui.Context, busy bool) {
	d := settings.Default()
	settingRow(c, "图片格式", "设置 CBZ 内的页面编码。漫画包仍保存为 CBZ。", func() {
		settingSelect(c, &a.editor.Output.Format, []settingOption[string]{{"jpeg", "JPEG"}, {"png", "PNG"}, {"webp", "WebP"}}, busy)
	})
	if a.editor.Output.Format == "jpeg" {
		settingRow(c, "JPEG 质量", "80～100。提高质量会增加文件大小。", func() {
			a.settingInt(c, &a.editor.Output.JPEGQuality, 80, 100, d.Output.JPEGQuality, busy)
		})
	}
	if a.editor.Output.Format == "webp" {
		settingRow(c, "WebP 无损", "不增加有损编码失真，但文件可能更大。", func() {
			ui.Switch(c, &a.editor.Output.WebPLossless).Disabled(busy)
		})
		if !a.editor.Output.WebPLossless {
			settingRow(c, "WebP 质量", "1～100，与 JPEG 的同值质量不能直接比较。", func() {
				a.settingInt(c, &a.editor.Output.WebPQuality, 1, 100, d.Output.WebPQuality, busy)
			})
		}
	}
	settingRow(c, "输出长边上限（px）", "0 保持原尺寸，或输入 640～16383。只缩小，不放大；回退页保留原图。", func() {
		a.settingInt(c, &a.editor.Output.MaxLongEdge, 0, 16383, d.Output.MaxLongEdge, busy)
	})
	settingRow(c, "无文字页面保留原图", "保留原格式和尺寸。关闭后统一转换为输出格式。", func() {
		ui.Switch(c, &a.editor.Output.PreserveNoText).Disabled(busy)
	})
	settingRow(c, "保存可编辑结果", "在输出旁保存识别、译文和排版数据。", func() {
		ui.Switch(c, &a.editor.Output.SaveEditable).Disabled(busy)
	})
	settingRow(c, "保存诊断文件", "保留排查漏字、擦字与排版所需文件，占用更多空间。", func() {
		ui.Switch(c, &a.editor.Output.SaveDiagnostics).Disabled(busy)
	})
	settingRow(c, "已有输出文件", "替换前询问，或保留旧文件并使用新文件名。", func() {
		settingSelect(c, &a.editor.Output.Existing, []settingOption[string]{{"ask", "每次询问"}, {"rename", "自动改名"}}, busy)
	})
}

func (a *app) pathSetting(c *ui.Context, title, description string, value *string, extensions []string, busy bool) {
	ui.Column(c).FillWidth().Padding(16, 0).Gap(8).Children(func() {
		ui.Text(c, title).FontSize(13).TextColor(ink)
		ui.Text(c, description).FontSize(11).TextColor(muted)
		ui.Row(c).FillWidth().Gap(8).AlignItems(ui.Center).Children(func() {
			ui.TextInput(c, value).Placeholder("使用默认文件").ReadOnly(true).Grow(1).MinWidth(0).Height(34).Disabled(busy)
			if ui.Button(c, "选择").Height(34).Disabled(busy).Clicked() {
				paths, err := mygo.Dialog.Open(mygo.OpenDialogOptions{
					Title:   "选择" + title,
					Filters: []mygo.FileFilter{{Name: title, Extensions: extensions}},
				})
				if err != nil {
					a.settingsError = "无法选择文件。" + err.Error()
				} else if len(paths) > 0 {
					*value = paths[0]
				}
			}
			if ui.Button(c, "默认").Height(34).Disabled(busy || *value == "").Clicked() {
				*value = ""
			}
		})
	})
	ui.Row(c).Height(1).Background(innerLine)
}

func unusedOutput(path string) (string, error) {
	ext := filepath.Ext(path)
	stem := strings.TrimSuffix(path, ext)
	for i := 2; i < 10000; i++ {
		candidate := fmt.Sprintf("%s (%d)%s", stem, i, ext)
		if _, err := os.Stat(candidate); os.IsNotExist(err) {
			return candidate, nil
		} else if err != nil {
			return "", err
		}
	}
	return "", fmt.Errorf("too many existing outputs")
}
