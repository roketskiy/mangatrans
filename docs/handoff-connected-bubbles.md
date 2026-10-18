# 当前工作交接：连体气泡与性能基准

> **过时声明（2026-10-18 晚）：** 本文的几何与排版改动仍在代码里，但真实链路验证表明单靠它们解决不了连体气泡。最新的根因分析和方案见 `docs/fullchain-optimization.md`。本文仅作历史记录。

本文记录 2026 年 10 月 18 日在 `E:\work\transl` 完成的性能改造和连体气泡排版工作。文档只记录代码和实际产物已经证明的事实。

## 当前状态

- 性能测试基础设施已经完成。
- 前 20 张图片的正式对比已经完成。旧实现和优化实现各运行 5 轮，共 5 个交错配对。
- OCR 保持串行。OCR 并发试跑没有进入正式优化结果。
- 连体气泡的几何区域和形状排版已经写入源码。
- 连体气泡改动已经通过 `internal/bubble`、`internal/render` 和 `cmd/mangatrans` 的针对性测试。
- 6 张测试页的固定译文对比已经生成。新排版没有把文字像素写到允许区域之外。
- 已补上 OCR 前的对白分框，默认统一竖排，并替换 `bin/mangatrans.exe`。最新状态见下一节；本文后面的固定译文图属于此前的排版验证，不能证明对白已经独立识别。
- 工作区仍有大量未提交修改。不要用 `git reset`、`git checkout` 或其他破坏性命令清理工作区。

## 17:20 更新：修复对白处理单位

此前把连体气泡当作一个 OCR 框、一个翻译块，然后只在渲染阶段分配文字，方向不完整。现在 `internal/bubble/split.go` 在检测后、OCR 前分析气泡内部厚度，识别由细连接相连的两个主体区域，并生成独立 `TextRegion`。没有可信分区时保留原框。

- 每个子区域独立进入 OCR 和翻译。`Area` 限定最终排版，`CropMask` 确定源图像素归属；OCR 裁图会将邻居对白置白。
- `render.Block.SourceArea` 限定擦字，避免成功的子块擦掉失败子块的原文。
- 已删除本轮尝试的“在渲染末端拆分整段译文”逻辑；渲染只负责一个独立对白的换列。
- 默认 `-text-direction vertical`；可显式选择 `horizontal` 或 `auto`。原先默认仅按检测框高宽比决定横竖，宽连体框因此会变成横排。
- 性能 JSON 新增 `ocr_regions`，区分原始检测框和分框后的 OCR 工作量。之前的正式性能数字不适用于新工作量。

`cmd/probe-regions` 与正式 OCR 共用 `TextRegion.Crop`，输出原日文裁图和父子框。运行：

```powershell
.\.cache\bubble-layout\probe-regions.exe `
  -in 'TestResourse\Training set\彼方のアストラ\1話.cbz' `
  -blocks '.cache\perf-opt\profile-final-20\work' `
  -out '.cache\bubble-layout\regions-final'
```

六页缓存父框共 56 个，得到 61 个独立 OCR 区域。第 07 页右下父框已拆成 `ありがとうございます！` 和 `取り返してくださったんですね！` 两个日文输入；`crop-07-007.png`、`crop-07-008.png` 可直接查看。第 08、11、12 页也拆出了相连对白。这里复用了旧检测框和 mask，没有重新运行 OCR、翻译模型，不能当作新译文质量验收。

`internal/bubble`、`internal/render`、`cmd/mangatrans` 的检查通过；包含真实样例分框、普通椭圆不误拆、源像素唯一归属、OCR 输入互不混入、失败邻居原文不擦除、独立翻译块等检查。`cmd/probe-regions`、`cmd/probe-layout` 编译通过。

当前 `bin/mangatrans.exe` SHA256 为 `f0f994fa69fd188aad717b9fc80fb0aab4893c81b88750ae829f6a3b91d64ba9`，与 `.cache/bubble-layout/mangatrans-bubbles.exe` 一致；旧引擎保存在 `.cache/bubble-layout/before.exe`。`-help` 已确认默认竖排。尚未运行新引擎的完整翻译链路，交由用户实测。

尚未覆盖的情况包括宽连接、多瓣气泡，以及找不到可信封闭内部的气泡。当前拆分器一次只接受两个主体；第 03、10 页没有新增分框。不能把这批结果描述为“所有连体气泡已修复”。

## 性能改造结果

正式测试使用：

```text
输入：TestResourse/Training set/彼方のアストラ/1話.cbz
范围：自然排序后的前 20 张图片
旧版：.cache/perf-opt/before.exe
优化版：.cache/perf-opt/after.exe
测试目录：.cache/perf-opt/paired-20
```

中位数结果如下：

| 指标 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 总耗时 | 93.833 秒 | 79.527 秒 |
| 进程树 RSS 峰值 | 3.663 GiB | 3.091 GiB |
| Job 提交内存峰值 | 6.616 GiB | 4.938 GiB |
| Job 累计 CPU | 88.047 秒 | 79.078 秒 |
| 整卡显存采样峰值 | 5378 MiB | 3907 MiB |

配对总耗时变化中位数为 `-14.58%`。这只是当前机器和当前输入的描述性结果。

主要改动在以下位置：

- `cmd/mangatrans/main.go`
  - 翻译完成后尽早释放检测和翻译模型。
  - 擦字阶段按页重新解码，避免长期保留所有解码图。
  - 增加 `-ocr-workers` 和 `-mt-batch-pages` 参数。
  - 增加 `perf-json` 事件路径。
- `cmd/mangatrans/pipeline.go`
  - OCR 结果按原框索引回填。
  - 翻译批次最多跨相邻页面，并使用请求内局部 ID。
- `internal/inpaint/inpaint.go`
  - 合并可安全合并的擦字窗口。
  - 对满足条件的平坦背景使用局部填充。
  - 记录擦字组件、窗口、模型调用和局部填充计数。
- `internal/perf/`
  - 使用 Windows QPC 记录高精度事件。
  - 区分顶层操作、嵌套事件、HTTP 请求和计数器。
- `scripts/bench.py`
  - 记录进程树 RSS、Job CPU、Job I/O、Job 提交内存和整卡 GPU 采样。
  - 校验输出 ZIP 的页名、数量、CRC、完整解码、尺寸和应绘制页变化。
- `scripts/compare_bench.py`
  - 交错运行 before 和 after。
  - 要求逐页工作量一致。
  - 少于 5 个有效配对时不下结论。

完整性能报告在 `.cache/perf-opt/summary.md`，原始对比在 `.cache/perf-opt/paired-20/comparison.json` 和 `.cache/perf-opt/paired-20/comparison.md`。性能决策记录在 `.cache/perf-opt/decisions.tsv`。

## 连体气泡问题的根因

当前问题不是检测置信度或 NMS 参数造成的。

检测器参数扫描使用了 6 张测试页：`03.png`、`07.png`、`08.png`、`10.png`、`11.png`、`12.png`。扫描结果保存在 `.cache/bubble-layout/detect-sweep/boxes.json`。

扫描结论：

- `ConfThreshold` 从 `0.25` 到 `0.55`，大多数测试页的文本框完全不变。
- `NMSThreshold` 从 `0.20` 到 `0.70`，这 6 张测试页的文本框完全不变。
- `BoxDilation` 只改变框边界，不改变框数量。
- `CloseRadius` 只改变擦字掩码面积，不拆分文本框。
- `ConfThreshold=0.55` 在第 10 张测试页少了一个框，因此不能作为解决方案。

原来的排版问题来自两层几何损失：

1. `internal/bubble/bubble.go` 的旧 `Fit` 只返回白色连通区域的外接矩形。连体气泡的凹口和收窄部分被矩形填平。
2. `internal/render/render.go` 以前只在矩形中居中文字。同一个白色连通区域被多个文本块使用时，只有第一个块能拿到气泡位置，其他块回退到 OCR 框。

因此，继续调 `conf`、NMS 或 `close` 参数不能解决主要问题。参数扫描结论已经记录在本交接文档对应的产物中，不要因为视觉上“像是合框”就继续盲调检测阈值。

## 连体气泡实现

### 气泡几何

`internal/bubble/bubble.go` 新增：

```go
func (f *Finder) FitAreas(boxes []image.Rectangle) []*image.Alpha
```

行为：

- 返回值与输入框一一对应。
- 返回的 `image.Alpha` 使用原图绝对坐标。
- 通过边界 flood fill 保留真实凹口。
- 只在关联 OCR 框附近恢复封闭文字孔洞。
- 对同一连通气泡内的多个 OCR 框按距离划分互斥区域。
- 对不可信区域返回 `nil`，调用方回退到原 OCR 框。
- 在最后一轮排除所有回退框的区域，避免扩展区域覆盖仍保留的原文字。
- 返回前裁剪到实际非零区域。

相关测试在 `internal/bubble/areas_test.go`，覆盖：

- 开口凹口不被填回。
- OCR 字符内部的封闭孔洞可以使用。
- OCR 框外的封闭插画区域不被当作排版空间。
- 同一气泡内的两个文本块区域互不重叠。
- 回退框不会被其他扩展区域覆盖。
- 非零原图坐标仍返回绝对坐标。

### 形状排版

`internal/render/render.go` 增加了可选的 `render.Block.Area`。当 `Area != nil` 时，`internal/render/shape.go` 会：

- 在完全安全的行列带内排版。
- 保持译文 rune 顺序。
- 保持原 OCR 框排版得到的字号上限，避免短句因为气泡过大而被放大。
- 在相邻列之间重新平衡字符，避免最后一列只剩一个字符。
- 处理开头闭合标点和结尾开括号的边界。
- 如果最小字号也无法完整容纳文本，返回错误。调用方保留原页，不静默丢字。

相关测试在 `internal/render/shape_test.go`，覆盖：

- 凹形和双瓣区域的中英文排版。
- 横排和竖排。
- rune 顺序完整保留。
- 文字像素不越过安全区域。
- 分列平衡。
- 不可排版区域返回错误。

### 主流程接入

`cmd/mangatrans/main.go` 在擦字成功后使用擦字后的 `clean` 图重新识别气泡轮廓。原因是原文字笔画会把白色气泡内部切碎，直接对原图找白色连通区域会得到错误的碎片。

主流程仍使用原 OCR 框生成擦字 mask。气泡识别只负责排版区域，不改变检测结果，也不改变擦字范围。

## 固定译文对比产物

`cmd/probe-layout/main.go` 是可重复运行的固定译文对比工具。它复用已有的 `blocksNN.json` 和 `maskNN.png`，比较旧矩形排版与新形状排版，不重新调用 OCR 或翻译。

当前对比目录：

```text
.cache/bubble-layout/fixed-v2/
```

本次使用的页面：

```text
03, 07, 08, 10, 11, 12
```

主要文件：

- `before-NN.png`：旧矩形排版。
- `after-NN.png`：新形状排版。
- `compare-NN.png`：左右对照图。
- `areas-NN.png`：气泡区域、OCR 框和排版区域叠加图。
- `allowed-NN.png`：逐像素验证允许写入的区域。
- `clean-NN.png`：擦字后的共同输入图。
- `manifest.json`：输入哈希、固定译文、框、区域、回退数和像素验证结果。

当前 6 张页面都满足：

```text
outside_allowed_pixels = 0
```

这只能证明文字没有写到允许区域之外，不能证明译文语义、视觉美观或所有气泡都已经正确。

## 已验证命令

在 Windows PowerShell 中，使用仓库自带的 CGO 工具链：

```powershell
$env:CGO_ENABLED = '1'
$env:CC = 'E:\work\transl\runtime\toolchain\mingw64\mingw64\bin\gcc.exe'
$env:PATH = 'E:\work\transl\runtime\toolchain\mingw64\mingw64\bin;' + $env:PATH

gofmt -w cmd/mangatrans/main.go internal/bubble/bubble.go internal/render/render.go internal/render/shape.go cmd/probe-layout/main.go
go test ./internal/bubble ./internal/render ./cmd/mangatrans -count=1
```

最近一次结果：

```text
ok    mangatrans/internal/bubble
ok    mangatrans/internal/render
ok    mangatrans/cmd/mangatrans
```

构建固定译文对比工具：

```powershell
go build -o .cache\bubble-layout\probe-layout.exe .\cmd\probe-layout
```

复用已经生成的擦字页重新生成固定译文对比：

```powershell
.\.cache\bubble-layout\probe-layout.exe `
  -in 'TestResourse\Training set\彼方のアストラ\1話.cbz' `
  -blocks '.cache\perf-opt\profile-final-20\work' `
  -clean-dir '.cache\bubble-layout\fixed-v1' `
  -out '.cache\bubble-layout\fixed-v2' `
  -pages '3,7,8,10,11,12'
```

运行检测参数扫描：

```powershell
go build -o .cache\bubble-layout\probe-detect.exe .\cmd\probe-detect
.\.cache\bubble-layout\probe-detect.exe -sweep `
  '.cache\bubble-layout\selected.cbz' `
  '.cache\bubble-layout\detect-sweep'
```

## 还没有完成的工作

### 重新构建正式引擎

17:20 已重新构建并替换正式二进制。后续修改后可用以下命令重新构建：

```powershell
$env:CGO_ENABLED = '1'
$env:CC = 'E:\work\transl\runtime\toolchain\mingw64\mingw64\bin\gcc.exe'
$env:PATH = 'E:\work\transl\runtime\toolchain\mingw64\mingw64\bin;' + $env:PATH
go build -trimpath -o .\bin\mangatrans.exe .\cmd\mangatrans
.\bin\mangatrans.exe -h
```

构建后必须重新检查：

- `bin\mangatrans.exe` 的 `-h` 包含现有参数。
- 主流程输出的文本块数量与输入工作量一致。
- 发生排版错误时整页回退，不出现擦字后无译文。
- 气泡区域没有覆盖人物、插画或页框。

### 真实引擎验证

固定译文工具已经验证了布局，但没有验证完整的 OCR、翻译、擦字和写包链路。下一步需要用新构建的 `bin\mangatrans.exe` 对测试资源运行至少一轮，并保存 `-work` 目录。

建议先限制页面：

```powershell
.\bin\mangatrans.exe `
  -in 'TestResourse\Training set\彼方のアストラ\1話.cbz' `
  -out '.cache\bubble-layout\engine-check.cbz' `
  -limit 20 `
  -work '.cache\bubble-layout\engine-work'
```

检查：

- 日志中的 `translated` 和 `rendered` 数量。
- `engine-work\blocksNN.json` 的框和译文。
- 输出 CBZ 的页数、页名、CRC 和完整解码。
- 用户提供的连体气泡样例页。

### 质量验收

目前只抽看了固定译文对比图，没有完成整个测试章节的视觉验收。优先查看：

- 页面 `03`：上方两个相邻气泡。
- 页面 `07`：顶部大气泡、右下大气泡和多个小气泡。
- 页面 `08`：多个短竖排对白。
- 页面 `10`：两个大气泡和机器人场景。
- 页面 `11`：顶部连体对白及人物介绍。
- 页面 `12`：多人对白和底部气泡。

重点检查：

- 同一气泡内的文本是否互相挤压。
- 竖排文本是否出现单字孤列。
- 长横排文本是否进入人物或气泡边界。
- 气泡内的封闭插画区域是否被当作文字空间。
- 不可信区域回退后，原文字是否仍然存在。

## 已知限制和风险

- 检测器目前没有参数可以稳定拆分这些连体文本框。不要把阈值扫描结果当作解决方案。
- `FitAreas` 依赖白色连通区域和 OCR 框位置。特殊背景、黑色气泡、渐变气泡和开口气泡可能回退到 OCR 框。
- 形状排版只保证不越过安全 mask，不保证语义分段正确。一个 OCR 框跨越多个视觉气泡时，代码不会猜测如何拆译文。
- 译文批次上下文会改变模型输出。之前的性能对比中，153 条译文有 90 条措辞变化。数量一致不等于语义一致。
- `internal/translate` 的既有单元测试曾被安全软件拦截，不能把它算作通过。
- 漏检问题按用户要求暂不处理。
- 当前这些改动尚未提交。接手者应先查看 `git diff` 和 `git status`，再决定是否拆分提交。

## 接手顺序

1. 阅读 `internal/bubble/bubble.go`、`internal/render/render.go`、`internal/render/shape.go` 和 `cmd/mangatrans/main.go` 的 Stage C。
2. 查看 `.cache/bubble-layout/fixed-v2/compare-03.png`、`compare-07.png`、`compare-08.png`、`compare-10.png`、`compare-11.png`、`compare-12.png`。
3. 运行 `go test ./internal/bubble ./internal/render ./cmd/mangatrans -count=1`。
4. 后续源码修改通过检查后，再重新构建 `bin/mangatrans.exe`。
5. 用新二进制运行 20 页真实链路，确认日志和输出。
6. 根据视觉验收结果决定是否继续调整 `FitAreas` 或 `shape.go`。
7. 只有确认真实链路后，才更新 README 或性能报告中的生产版本信息。

## 交接结论

下一步应以 OCR 前的独立对白分框为核心，检查真实识别结果与气泡对应关系。气泡形状排版只是后续步骤，旧固定译文图不能替代分框验收。正式引擎已更新；完整模型链路和整章视觉验收仍未完成。
