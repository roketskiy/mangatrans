# 连体气泡全链路优化记录

本文记录 2026 年 10 月 18 日晚针对「连体气泡」问题的全链路优化。所有结论都有 `E:\work\transl\.cache\fullchain` 下的产物支撑。前一轮工作见 `docs/handoff-connected-bubbles.md`（已标记过时）。

## 用户目标

- 每段翻译放在自己的气泡上，不再多段挤一个区域。
- 字体大小合适，与原文协调。

## 根因

基线运行 `TestResourse/Training set/bubble.cbz`（6 页，57 个检测框）确认：

1. 一个检测框覆盖多个对白时（黑气泡、重叠气泡都会发生），旧 `SplitTextRegions` 只会拆「白色区域 + 细连接」，上述情况全部漏过。OCR、翻译、渲染以「多对白合并块」运行，译文只能挤进一个区域。
2. 检测器的文本掩膜里本来就有分离信号：不同对白的文字像素之间有完整空白带（见 `baseline-work/mask00.png` 左上框内的三个分离簇），旧代码没有用它。
3. 字号从固定 `MaxSize=16` 顶格试起，大区域必然顶满，与原文大小无关。

## 改动

### 1. 文本掩膜间隙拆分（`internal/bubble/splitmask.go`）

- `splitBoxByMaskGaps` 把检测框内的文本掩膜沿阅读轴投影，在静默带处切开，每段成为一个独立对白。
- 两个投影轴都尝试后择优：两个竖排气泡左右并排会形成宽框，只看框宽高比会选错轴。段数多者胜，同段数比像素均衡度。
- 静默不足 10px 的相邻带合并回一组，防止单个对白的列组被拆散；带的最小宽度 16px，每段至少占全框墨水的 1/8。
- 竖排段按日文阅读序（右到左）返回，保证翻译批次的上下文连贯。
- `growSegmentMask` 用 BFS 把裁剪窗内每个像素分给最近的段，OCR 裁图不会看到邻居的文字。
- 子段在原图上各自 `FitAreas`，渲染直接拿到气泡形状约束。

### 2. 字号先验（`cmd/mangatrans/helpers.go` + `internal/render/render.go`）

- `estimateFontSize` 用墨水密度估算原文字号：掩膜面积 ÷ OCR 字符数 ÷ 0.33（日文字形的 em 覆盖率），开平方。
- `render.Block.SizeHint` 把估算值带进渲染；二分搜索的上限从固定 `Max` 收紧为 `1.2 × SizeHint`，译文大小跟随原文。
- `MaxSize` 从 16 提到 18，给大字号原文留空间。

### 接入顺序

`SplitTextRegions` 先试掩膜间隙拆分，不成立再走旧的白色区域拆分，最后回退整框。黑气泡在白色路径拿不到区域，现在由掩膜路径接管。

## 验证

单元测试：`go test ./internal/bubble ./internal/render ./cmd/mangatrans -count=1` 全部通过（`internal/translate` 的测试二进制仍被杀软拦截，属已知环境问题）。新增用例覆盖双对白、三对白、横排、单对白不拆、杂点忽略、所有权掩码互斥、字号估算。

真实链路（`bubble.cbz`，6 页）：

| 指标 | 基线 | 优化后 |
| --- | ---: | ---: |
| OCR 区域数 | 62 | 70 |
| 翻译块 | 61 | 69 |
| 总耗时 | 44.3s | 41.5s |

- 页 1 左上黑框拆成 3 段，各自落在自己的黑气泡里（`regions-v2/overlay-01.png`）。
- 页 4 左上合并块和底部两组对白全部拆开（`regions-v2/overlay-04.png`）。
- 字号不再顶满大气泡（`compare-p00.png`、`compare-p03.png`、`compare-p04.png` 为前后对比）。

## 气泡分割模型（同日晚追加）

在几何启发式之上接入了实例分割模型 `huyvux3005/manga109-segmentation-bubble`（MangaLens，YOLO11n-seg，Apache-2.0，Manga109 训练）：

- 模型文件 `models/mangalens-bubble-seg.onnx`（11.9MB，1600 输入，`scripts/export_mangalens.py` 一条命令重导出；导出不能开 onnxslim 简化，它会折错 anchor reshape）。
- `internal/segment` 负责 ONNX 前向、NMS 和原型掩膜合成，输出页面绝对坐标的实例掩膜。
- `bubble.SplitWithBubbles` 把每个文本框的文字像素按实例归属分组：单实例框保留内部投影拆分、子段继承实例掩膜做排版区域；跨实例框在实例边界拆开；无实例解释的框走原几何链路。
- `render.Block.Anchor` 兑现：排版 lane 窗口锚定原文段框，多段共享一个气泡形状时不再向形状中心漂移。

验收（`bubble.cbz` 六页）：实例数 8/9/10/13/8/10，掩膜贴合云朵锯齿边和黑色气泡内部；分段数与纯投影方案一致（69 块），但深色和异形气泡首次有了精确形状；页 1 与页 4 的多瓣连体气泡全部按瓣落位。分割耗时约 0.3 秒/页。叠加图在 `.cache/fullchain/seg-probe/`，渲染对比在 `v4-work/`。

## 遗留问题

- 「ワンピース」被译成「哭哭奕」一类的 OCR/翻译错误仍在，属于识别与翻译环节，本轮未动。
- 左右投影重叠的两个对白（间隙里两段文字在阅读轴上互相覆盖）仍拆不开，可用连通域聚类增强。
- 拟声词（SFX）仍会进入对白检测框，用户已声明漏检/误检问题暂不处理。
- 探针产物在 `.cache/fullchain/`：`baseline-work/`（改动前全链路）、`v3-work/`（当前引擎）、`regions-v2/`（分框叠加图）。
