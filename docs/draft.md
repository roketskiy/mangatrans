# 本地漫画翻译桌面应用 技术草案

版本 0.2。日期 2026-10-17。状态 待评审。

## 1. 目标与硬约束

做一个 Windows 桌面应用。用户拖入漫画文件夹或 CBZ，本机完成版面分析、OCR、翻译、擦字、排版，输出中文版。

硬约束来自需求。

- 外壳与后端用 Go，框架 mygo。
- 前端是 web 页面。
- 神经网络推理用 ONNX Runtime，NVIDIA GPU 加速。
- 翻译用 llama.cpp 部署 `unsloth/Hy-MT2-1.8B-GGUF` 的 3-bit 量化版。
- 只用 HuggingFace 上现成的 ONNX 文件，不做模型转换。
- 每个流水线阶段只选一个模型。
- 流程与模型选择参考本地 `koharu-slim`。

开发期约束。模型与缓存一律落在 `E:\work\transl\.cache`，不写 C 盘。

## 2. 目标机器

本机实测。

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| 显存 | 6141 MiB |
| 驱动 | 591.86，报告 CUDA 13.1 |
| Go | 1.26.5 windows/amd64 |
| Bun | 1.4.2 |
| Node | 24.14.1 |
| CUDA Toolkit | 13.0，只有编译器，无运行时 DLL |
| CUDA 运行时 | `E:\Python312\Lib\site-packages\torch\lib`，cu13 加 cuDNN 9，37 个 DLL，2611 MB |
| C 编译器 | winlibs mingw-w64 gcc 16.2，装在 `E:\work\transl\runtime\toolchain` |

6 GB 显存是本方案最强的约束。第 8 节按它分配。

### 2.1 CUDA 环境的真实情况

三条实测结论，都会影响方案。

第一，CUDA Toolkit 13.0 的 `bin` 只有 14 个编译工具（nvcc、ptxas、nvlink 等），`lib\x64` 只有 `.lib` 导入库。它没有 `cublas64_13.dll` 这类运行时 DLL。

第二，运行时 DLL 在 PyTorch 里。`torch 2.12.0+cu130` 的 `lib` 目录含全套 cu13 与 cuDNN 9。

| 文件 | 体积 |
| --- | --- |
| `cublasLt64_13.dll` | 455.8 MB |
| `cublas64_13.dll` | 48.0 MB |
| `cudart64_13.dll` | 0.4 MB |
| `cudnn_engines_precompiled64_9.dll` | 197.2 MB |
| `cudnn_adv64_9.dll` | 102.0 MB |
| `cudnn_heuristic64_9.dll` | 56.2 MB |
| `cudnn_ops64_9.dll` | 35.0 MB |
| 其余 cuDNN 子库 | 约 30 MB |

第三，Python 的 `onnxruntime-gpu 1.26` 是 cu12 构建，在本机必然失败。它的 `onnxruntime_providers_cuda.dll` 硬依赖 `cublasLt64_12.dll`，本机只有 cu13，加载失败后静默回落到 CPU。实测跑一次 574 ms，那是纯 CPU 速度。

所以 ORT 必须用 **cuda13 变体**。选 ORT 1.31.0，因为它是当前最新，且官方提供 `onnxruntime-win-x64-gpu_cuda13-1.31.0.zip`。

运行时策略按产品决定执行，自用场景直接引用 torch 的 DLL 路径，不复制 2.6 GB。启动时把两个目录加入 DLL 搜索路径。

- `E:\work\transl\runtime\ort-cuda13`，ORT 的三个 DLL。
- `E:\Python312\Lib\site-packages\torch\lib`，CUDA 与 cuDNN。

代价是依赖本机 Python 环境不被移动或升级。若要迁移，只需把上面 8 个 DLL 复制到项目 `runtime/cuda13`，约 895 MB。

### 2.2 已跑通的 CUDA 验证

2026-10-17 实测，Go 程序直接加载 ORT 1.31 cuda13。

`comictextdetector.onnx`，输入 `[1,3,1024,1024]`，预热两次后取 5 次最快值。

| 后端 | 单次推理 |
| --- | --- |
| CUDA EP | 42.6 ms |
| CPU EP | 8734.9 ms |
| 加速比 | 205.2x |

这个 205 倍差距是 CUDA EP 真的生效的证据，不是静默回落。代码在 `spike/ort/epcompare/main.go`。

`aot.onnx`，输入 512×512，CUDA 预热后 62 到 68 ms。

### 2.3 工具链的硬要求

Go 在 Windows 上编译 cgo 必须用 gcc 风格的编译器。实测三条路都不通。

| 尝试 | 结果 |
| --- | --- |
| MSVC `cl` | Go 给 `runtime/cgo` 传 `-Werror` 等 gcc 参数，`cl` 报 D8021 |
| LLVM `clang` | 缺 `-mthreads` 支持，报 unsupported option |
| 系统自带 | 没有 gcc |

可行路径是 winlibs mingw-w64。已安装 gcc 16.2.0 到 `E:\work\transl\runtime\toolchain\mingw64\mingw64\bin`。

构建时设置两个环境变量。

```
CGO_ENABLED=1
CC=E:\work\transl\runtime\toolchain\mingw64\mingw64\bin\gcc.exe
```

### 2.4 onnxruntime_go 的实际 API

版本 v1.36.0。几个名字与常见文档不同，实测确认。

| 用途 | 实际名字 |
| --- | --- |
| 设置 CUDA EP | `AppendExecutionProviderCUDA(*CUDAProviderOptions)` |
| 设置优化级别 | `SetGraphOptimizationLevel` |
| 建会话 | `NewAdvancedSession(path, inNames, outNames, inValues, outValues, opts)` |
| 预分配张量 | `NewEmptyTensor[T](shape)` |

不存在 `SetCUDAProvider`、`GetProviderName`、`AppendExecutionProviderCPU`。CPU 是默认回退，不显式设置。

CUDA 选项用字符串 map 传入。

```go
cuda.Update(map[string]string{
    "device_id":                 "0",
    "arena_extend_strategy":     "kSameAsRequested",
    "gpu_mem_limit":             "2147483648",
    "cudnn_conv_algo_search":    "HEURISTIC",
    "do_copy_in_default_stream": "1",
})
```

## 3. 结论先行

| 项 | 选择 | 理由 |
| --- | --- | --- |
| 桌面外壳 | mygo，web 前端加 Go 后端 | 需求指定。Go 直接持有推理运行时。 |
| 前端 | Vite 加 React 加 TypeScript | mygo 默认模板，适合大量缩略图与虚拟列表。 |
| IPC | `mygo.Bind` 加 `Channel` 传进度，自定义协议传图片 | 图片不走 base64。 |
| 推理绑定 | `yalue/onnxruntime_go` v1.36.0 加 ORT 1.31 cuda13 | 实测 CUDA EP 生效，比 CPU 快 205 倍。 |
| GPU 后端 | CUDA EP，运行库引用 torch 的 cu13 DLL | 本机已有全套，不复制文件。 |
| 翻译引擎 | `llama-server.exe` 子进程加 OpenAI 兼容 HTTP | 崩溃隔离，流式输出，显存可控。 |
| 翻译模型 | `Hy-MT2-1.8B-UD-Q3_K_XL.gguf` 990 MB | 3-bit 中质量最好的一档，比 Q3_K_S 只多 118 MB。 |
| OCR 引擎 | `llama-server.exe` 加 `--mmproj` | llama.cpp 已封装多模态生成循环，不用写解码。 |
| OCR 模型 | `PaddleOCR-VL-manga-Q4_K_M` 286 MB 加 mmproj 570 MB | 专为日文漫画微调，有现成 GGUF。 |
| 显存策略 | OCR 与翻译串行，先跑完全部 OCR 再加载翻译模型 | 6 GB 装不下两个大模型常驻。 |
| 排版渲染 | `tdewolff/canvas` | 竖排、断行、整形、光栅化都现成，已实测竖排出图。 |
| C 编译器 | winlibs mingw-w64 gcc 16.2 | Go 在 Windows 上编译 cgo 的唯一可行路径。 |

## 4. 流水线，每阶段唯一模型

顺序参考 `koharu-slim/src/pipeline.rs` 的 `prepare_page` 与 `finish_page_inner`。

八个阶段，其中六个跑模型，两个纯规则。阶段编号按执行顺序。

| # | 阶段 | 模型 | 运行时 |
| --- | --- | --- | --- |
| 1 | 文本与气泡检测 | ComicTextDetector | ORT |
| 2 | 版面分类 | PP-DocLayoutV3 | ORT |
| 3 | 字体属性 | YuzuMarker int8 | ORT |
| 4 | 气泡分割 | YOLOv8m-seg | ORT |
| 5 | OCR | PaddleOCR-VL-For-Manga | llama.cpp |
| 6 | 翻译 | Hy-MT2-1.8B | llama.cpp |
| 7 | 擦字修补 | LaMa manga | ORT |
| 8 | 排版渲染 | 无 | 纯 Go |

阶段 5 与 6 都跑 llama.cpp，所以按书级串行，见 4.6 节。

所有 ORT 模型的形状都是 2026-10-17 本机实测，脚本 `.cache/probe_onnx.py`，结果 `.cache/onnx-probe/probe.json`。

### 阶段 1 文本与气泡检测

| 项 | 内容 |
| --- | --- |
| 模型 | ComicTextDetector |
| 来源 | `mayocream/koharu` 的 `comictextdetector.onnx` |
| 体积 | 90.3 MB，单文件，无外部权重 |
| opset | ai.onnx 11，388 节点 |
| 输入 | `images` FLOAT `[1,3,1024,1024]` |
| 输出 | `blk` FLOAT `[1,64512,7]`，`seg` FLOAT `[1,1,1024,1024]`，`det` FLOAT `[1,2,1024,1024]` |
| 算子 | 只有 Conv、ConvTranspose、BatchNorm、Resize、Slice、Transpose 等基础算子，无 GridSample，无自定义算子 |

这一个模型干三件事。

- `blk` 是 YOLOv5 输出。64512 = (128² + 64² + 32²) × 3，即三个 stride 的锚点。7 = 4 个框坐标 + 1 个置信度 + 2 个类别。经置信度 0.4 阈值与 NMS 0.35 得到文字块。
- `seg` 是 UNet 输出的像素级文字 mask。这是擦字阶段的输入。
- `det` 是 DBNet 的 shrink map 与 threshold map，两通道。用于细化文字区域。

`koharu-slim` 把检测与文本分割拆成两个阶段，是因为它按模块拆分。模型本身已经合并。我们合并成一次前向，省一次全页推理。

后处理参数直接沿用 `koharu-slim/crates/koharu-ml/src/comic_text_detector/mod.rs`。

- 置信度阈值 0.4，NMS 阈值 0.35，框外扩 1.0 像素。
- mask 二值化阈值 60，膨胀半径 3，闭运算半径 10，最终膨胀 2。
- 预处理为等比缩放到 1024 长边后左上角补零，除以 255，不做均值方差归一化。

### 阶段 2 版面分类

| 项 | 内容 |
| --- | --- |
| 模型 | PP-DocLayoutV3 |
| 来源 | `PaddlePaddle/PP-DocLayoutV3_onnx` 的 `inference.onnx` |
| 体积 | 124.5 MB |
| opset | ai.onnx 17，2818 节点，含 GridSample |
| 输入 | `im_shape` FLOAT `[N,2]`，`image` FLOAT `[N,3,800,800]`，`scale_factor` FLOAT `[N,2]` |
| 输出 | `fetch_name_0` FLOAT `[N,7]`，`fetch_name_1` INT32 `[N]`，`fetch_name_2` INT32 `[N,200,200]` |

作用只有一个。给每个检测框一个语义标签，用于过滤页眉、页脚、页码、版权信息这类噪声框。

页面类型分类（封面、目录、正文、后记）不走模型，走规则。见 `koharu-slim/src/pipeline.rs` 的 `classify_page_kind`，依据框数量、OCR 长度、气泡数量、目录特征文本。

这是可选项。若实测在漫画页上收益不明显，可以只保留阶段 1 加规则分类。决策点见第 11 节。

### 阶段 3 字体属性

| 项 | 内容 |
| --- | --- |
| 模型 | YuzuMarker FontDetection |
| 来源 | `ogkalu/yuzumarker-font-detection-onnx` 的 `font-detector_int8.onnx` |
| 体积 | 34.7 MB |
| opset | ai.onnx 17，504 节点 |
| 输入 | `input` FLOAT `[batch,3,512,512]` |
| 输出 | `output` FLOAT `[batch,6162]` |

6162 维向量包含字体分类、文字颜色、描边颜色、字号、角度等属性。`koharu-slim` 的 `apply_font_detection` 把输出拆成方向、颜色、字号、描边、倾角五组。

这是可选项。跳过它时用默认值，排版质量下降但不影响正确性。

### 阶段 4 气泡分割

| 项 | 内容 |
| --- | --- |
| 模型 | YOLOv8m-seg speech bubble |
| 来源 | `kitsumed/yolov8m_seg-speech-bubble` 的 `model_dynamic.onnx` |
| 体积 | 103.9 MB |
| opset | ai.onnx 19，565 节点 |
| 输入 | `images` FLOAT `[batch,3,H,W]`，动态尺寸 |
| 输出 | `output0` FLOAT `[batch,4+32+1,anchors]`，`output1` FLOAT `[batch,32,mh,mw]` |

作用是给擦字阶段提供气泡 mask。`koharu-slim` 的 `apply_bubble_fill` 用它在纯色气泡里直接填满，跳过 LaMa 推理。这一步在速度上很值。

可选。跳过时全部走 LaMa。

### 阶段 5 OCR

| 项 | 内容 |
| --- | --- |
| 模型 | PaddleOCR-VL-For-Manga |
| 来源 | `LunarOilRig/PaddleOCR-VL-manga-GGUF` |
| 文件 | `PaddleOCR-VL-manga-Q4_K_M.gguf` 286.3 MB，`mmproj-Q8_0.gguf` 569.9 MB |
| 上游 | `jzhang533/PaddleOCR-VL-For-Manga`，日文漫画微调 |
| 引擎 | `llama-server.exe` 加 `--mmproj` |
| 提示词 | `OCR:`，模型按这个提示词训练 |
| 许可 | Apache-2.0 |

选择理由。三条，按重要性排序。

第一，不用写解码循环。llama.cpp 已经实现多模态模型的完整生成流程，视觉编码、图像特征注入、自回归解码、KV cache 全在里面。我们只发 HTTP 请求。

第二，专为日文漫画微调，比通用 OCR 更合适。

第三，有现成 GGUF，符合“只用现成模型”的约束。作者实测在 7 个裁剪区域的日文漫画页上，Q4_K_M 与 F16 输出逐字相同，只有一处模糊手写汉字不同。

输入是一张裁剪图，不是整页。所以阶段 1 的检测框要先裁剪并外扩，再送进来。

`llama-mtmd-cli` 的等价参数用于 `llama-server`。

```
--model PaddleOCR-VL-manga-Q4_K_M.gguf
--mmproj mmproj-Q8_0.gguf
--temp 0
--jinja
```

`--temp 0` 是必须的。OCR 不需要随机性。

备选路线，已评估并放弃。

- Manga OCR（`l0wgear/manga-ocr-2025-onnx`，21.3 + 112.6 MB）。质量好，但 ONNX 导出只有编码器与解码器两张静态图，beam search 循环要在 Go 里自己写。既然 llama.cpp 已有现成方案，没必要。
- PP-OCRv3/v4 japan（`tobiichioriguchi/japan_PP-OCRv3_mobile_rec_onnx`，9.6 MB）。CTC 一次前向，最省显存，但通用日文模型，漫画场景未验证。

保留这两条作为 M3 的对照基线，不写进主流程。

### 阶段 6 翻译

| 项 | 内容 |
| --- | --- |
| 模型 | Hy-MT2-1.8B |
| 来源 | `unsloth/Hy-MT2-1.8B-GGUF` |
| 文件 | `Hy-MT2-1.8B-UD-Q3_K_XL.gguf`，990 MB |
| 备选 | `Hy-MT2-1.8B-Q3_K_M.gguf` 951 MB，`Hy-MT2-1.8B-Q3_K_S.gguf` 872 MB |
| 引擎 | `llama-server.exe` |
| 许可 | Apache-2.0 |

三档 3-bit 的取舍。UD-Q3_K_XL 用 unsloth 动态量化，同尺寸下关键层保留更多精度，只比 Q3_K_S 大 118 MB。选它。

一个备注。`IQ4_XS` 是 1.03 GB，只比 UD-Q3_K_XL 多 40 MB，质量明显更稳。如果 4-bit 也在允许范围内，值得实测对比。

### 阶段 5 与阶段 6 的串行安排

两个阶段都跑 llama.cpp，但 6 GB 显存装不下两份模型常驻。所以按书级串行，不按页级。

```
整本书：
  1. 加载 OCR 模型（llama-server A，286 + 570 MB）
  2. 所有页跑完检测、版面、字体、气泡、OCR
  3. 关闭 A
  4. 加载翻译模型（llama-server B，990 MB）
  5. 所有页跑翻译
  6. 关闭 B
  7. 所有页跑擦字与渲染（ORT 模型）
```

这样任一时刻只有一个 llama-server 活着。峰值取最大值，不叠加。

代价是多一次模型加载，3 到 8 秒，一本书只付一次。

代价之二，页级流水线被打断。检测与 OCR 可以流水，翻译必须等全部 OCR 完成。对单本书影响不大，对“打开一页就想看结果”的交互有影响。

交互场景的退路。若用户只想翻当前一页，走单页快路径：OCR 与翻译都用一个临时 llama-server，用完即关。这个路径慢，但只处理一页。

### 阶段 7 擦字修补

| 项 | 内容 |
| --- | --- |
| 模型 | LaMa manga |
| 来源 | `mayocream/lama-manga-onnx` 的 `lama-manga.onnx` |
| 体积 | 197.9 MB |
| opset | ai.onnx 17，17472 节点 |
| 输入 | `image` FLOAT `[batch,3,512,512]`，`mask` FLOAT `[batch,1,512,512]` |
| 输出 | `output` FLOAT `[batch,3,512,512]` |

输入固定 512，所以全页要切块。`koharu-slim` 的 `run_inpaint_with_windows` 用 1.7 倍窗口比例切块，块间重叠，边界羽化。

备选 `ogkalu/aot-inpainting` 的 `aot.onnx`，22 MB，输入尺寸动态 `[batch,3,h,w]`，速度快九倍。低显存模式用它，但同一时刻只启用一个。

### 阶段 8 排版渲染

无模型。用 `github.com/tdewolff/canvas`，产品决定。

选它的理由。它把排版封装完整，竖排、断行、字形整形、字体子集化都现成，底层正好也是 `go-text/typesetting`。自己写这些要几周。

2026-10-18 已实测通过，验证程序 `spike/canvas/main.go`。

- 竖排 `VerticalRL` 加 `Upright`：文字从上到下，列从右到左推进，汉字直立。
- 断行用 Knuth 算法，`TextOptions.Linebreaker` 默认就是它。
- 横排对照正常。
- 二进制 13.4 MB，没有拉进 GUI 框架。

实测发现的一个 API 陷阱。`NewTextBox` 在内部立刻完成排版，之后修改返回值的 `WritingMode` 无效。必须用 `NewRichText` 先 `SetWritingMode` 再 `ToText`。

能力对照。

| 能力 | Rust 实现 | canvas 实现 |
| --- | --- | --- |
| 断行 | ICU 加 jieba 加 DP | KnuthLinebreak，内置 |
| 字形整形 | HarfRust | 内置 shaper，底层 go-text/typesetting |
| 竖排 | HarfBuzz TTB 加 vert/vrt2 | `SetWritingMode(VerticalRL)`，已实测 |
| 竖排标点 | 字体 GSUB 替换 | `TextOrientation` 处理 |
| 自动字号 | 二分搜索 | 无，要自己包一层二分 |
| 气泡避让 | 无，只接矩形 | 无，同样只接矩形 |
| 光栅化 | skrifa 加 tiny-skia | `renderers/rasterizer`，已实测出 PNG |
| 黑白反色 | 不在渲染器内 | 我们自己实现 |

仍要自己写的部分。

- 自动字号搜索。在气泡尺寸上二分字号，调 `NewRichText` 与 `ToText`，直到结果放进内缩后的矩形。
- 气泡多边形内缩。canvas 只接受矩形框，漫画气泡常是椭圆，先内切成矩形再用。
- 黑白气泡反色。检测气泡内背景亮度，选黑字或白字。
- 中日文标点的禁则微调。canvas 有 `PunctuationInMargins` 选项，够不够要看实测。

字体沿用系统字体即可，自用场景不需要子集化。验证程序直接用了 `msyh.ttc`。

## 5. 核心数据模型

数据形状先定，逻辑后写。

```go
package project

type Kind uint8

const (
	KindUnknown Kind = iota
	KindCover
	KindToc
	KindBody
	KindAfterword
)

type Direction uint8

const (
	DirHorizontal Direction = iota
	DirVertical
)

type Rect struct{ X, Y, W, H float32 }

type Block struct {
	ID          int
	Box         Rect
	Quad        [4][2]float32
	Direction   Direction
	LayoutLabel string

	OCRText     string
	Translation string
	SkipReason  string

	Font *FontHint
}

type FontHint struct {
	SizePx      float32
	TextColor   [3]uint8
	StrokeColor [3]uint8
	StrokePx    float32
	AngleDeg    float32
}

type Bubble struct {
	Box  Rect
	Mask []byte
}

type Page struct {
	Index      int
	SourcePath string
	Width      int
	Height     int
	Kind       Kind
	Blocks     []Block
	Bubbles    []Bubble
	Stage      Stage
	Err        string
}

type Stage uint8

const (
	StageQueued Stage = iota
	StageDetected
	StageOCRed
	StageTranslated
	StageInpainted
	StageRendered
	StageDone
	StageSkipped
	StageFailed
)
```

`Stage` 是状态机，不是一组布尔标志。它保证“已翻译但未 OCR”这类非法状态无法出现。断点恢复只读 `Stage`。

## 6. 推理层

### 6.1 接口

```go
package infer

type ModelID string

const (
	ModelDetect  ModelID = "detect"
	ModelLayout  ModelID = "layout"
	ModelFont    ModelID = "font"
	ModelOCREnc  ModelID = "ocr-enc"
	ModelOCRDec  ModelID = "ocr-dec"
	ModelBubble  ModelID = "bubble"
	ModelInpaint ModelID = "inpaint"
)

type Spec struct {
	ID       ModelID
	Repo     string
	File     string
	SHA256   string
	Bytes    int64
	VRAM     int64
	Optional bool
}

type Session interface {
	Run(ctx context.Context) error
	Close() error
}

type Engine interface {
	Acquire(ctx context.Context, id ModelID) (Session, error)
	Release(s Session)
	Stats() Stats
}
```

上层只写 `Acquire(ModelDetect)`。加载、显存、回退都在 `Engine` 里。加模型是往注册表加一行，不是加一个分支。

### 6.2 张量复用

`onnxruntime_go` 的 `NewTensor` 把 Go slice 指针直接注册进 OrtValue，零拷贝。所以每个 session 预分配输入输出张量，每次 `Run` 前 `copy` 进输入，不再分配。

这对阶段 1 很重要。`blk` 输出是 64512×7×4 字节，约 1.8 MB，每次重新分配会明显拖慢。

### 6.3 并发

ORT 官方约定，同一 session 多 goroutine 并发 `Run` 是线程安全的，前提是每次用不同张量。

实际做法。单 session，每个 worker 一组独立张量，不加锁。6 GB 显存下不要每 worker 一个 session，权重与 arena 会翻倍。

### 6.4 回退链

产品决定只做 Windows 加 CUDA，所以不做 DirectML。回退只有两级。

1. CUDA EP 成功，记录 EP 为 CUDA。
2. CUDA EP 初始化失败，降级 CPU，UI 明确提示并给出原因。

每个模型单独记录实际 EP。ORT 在算子不支持时会把节点分给 CPU，日志里要能看见。

CUDA EP 的初始化失败原因要分类记录，因为这决定用户该修什么。

| 现象 | 原因 |
| --- | --- |
| 缺 `cublasLt64_13.dll` 等 | torch 的 DLL 路径没加进搜索路径 |
| 缺 `cudnn64_9.dll` | 同上，或 cuDNN 未随 torch 安装 |
| 显存不足 | `gpu_mem_limit` 设太高，或桌面程序占用了显存 |
| 驱动过旧 | 需要 550 以上 |

## 7. 翻译层

### 7.1 进程管理

`llama-server.exe` 随应用分发，放在 `resources/windows-amd64/bin/`。

启动参数，按 6 GB 显存。

```
--model models/Hy-MT2-1.8B-UD-Q3_K_XL.gguf
--host 127.0.0.1
--port <Go 分配的空闲端口>
-ngl 99
-c 8192
-b 2048
-ub 512
--flash-attn
--cache-type-k q8_0
--cache-type-v q8_0
--parallel 1
--jinja
--no-webui
```

显存估算。权重 1.0 GB，KV cache 8192 上下文 q8_0 约 0.5 GB，计算缓冲与 CUDA 上下文 0.4 到 0.7 GB。合计约 2 到 2.3 GB。

启动流程。

1. Go 选空闲端口。
2. 启动子进程，stdout 重定向到日志。
3. 轮询 `/health`。加载中返回 503 加 `{"error":{"code":503,"message":"Loading model"}}`，就绪返回 200 加 `{"status":"ok"}`。
4. 就绪后通知 UI。超时 120 秒。
5. 退出时发 SIGTERM，等 5 秒，再强杀。

崩溃隔离。子进程退出后，翻译阶段的页标记 `StageFailed` 并带原因。重启子进程不影响已完成的 OCR 与擦字。

注意端口。llama.cpp v0.7.0 起默认端口从 8080 改成 9931。我们显式指定端口，不依赖默认值。

### 7.2 提示词

Hy-MT2 官方单条模板。

```
将以下文本翻译为 `简体中文`，注意**只需要输出翻译后的结果，不要额外解释**：

`{source_text}`
```

官方声明模型没有默认 system prompt。GGUF 内置 chat template，用 `--jinja` 直接生效。

漫画要一次翻多个气泡并保持编号对应。`koharu-slim` 的做法是编号标签加 JSON 输出，解析失败回退标签解析。

我们的做法。

- 系统提示词固定，说明任务、格式、两条 few-shot 示例。
- 用户消息是编号列表。
- 用 `response_format` 的 `json_schema` 约束输出结构，llama-server 支持 schema 约束生成。
- 解析失败重试一次，仍失败回退编号标签解析。

批大小按 token 预算，不按页数。目标输入 1500 token 以内，约 3 到 5 页。`koharu-slim` 配置里的 20 页是给百万上下文模型的，不适用。

### 7.3 前缀复用

系统提示词固定且放最前，llama-server 会复用这段 KV cache。多批请求之间不要改系统提示词，否则每批都重新 prefill。

术语表（人名、专有名词）通过用户消息里的“参考下面的翻译”段注入，不放进系统提示词。

## 8. 显存与性能预算

因为 OCR 与翻译串行，任一时刻只有一个 llama-server 活着。所以峰值取最大值，不叠加。

| 时刻 | 组成 | 合计 |
| --- | --- | --- |
| 阶段 1 到 4 | 检测 90 MB 加版面 125 MB 加字体 35 MB 加气泡 104 MB，加激活峰值 | 约 2.4 GB |
| 阶段 5 OCR | OCR 权重 856 MB 加视觉缓存约 400 MB 加工作区 | 约 1.6 GB |
| 阶段 6 翻译 | 翻译权重 990 MB 加 KV cache 500 MB 加工作区 | 约 2.3 GB |
| 阶段 7 擦字 | LaMa 198 MB 加激活 | 约 1.0 GB |

最大值约 2.4 GB，6 GB 卡留出 0.8 GB 给桌面与驱动后仍有约 2.8 GB 余量。这个预算比之前乐观得多。

但这是估算，不是实测。真实峰值取决于 ORT 的 arena 策略与 llama.cpp 的批处理设置。M6 必须记录。

缓解手段，以防实测超预算。

- ORT arena 设 `kSameAsRequested` 加 `gpu_mem_limit`，避免 `kNextPowerOfTwo` 的倍增。
- `cudnn_conv_algo_search` 用 `HEURISTIC`，不用 `EXHAUSTIVE`。后者首次搜索很慢且 workspace 占用大。
- 阶段 1 的输入尺寸在显存不足时从 1024 降到 640。`koharu-slim` 在 CPU 模式下就是这么做的。
- 阶段 7 的 LaMa 分块从 512 降到 384。

并发模型。

- 页内严格串行，同一张卡。
- 同一阶段内页间流水线深度 2。第 N 页在做 OCR 时，第 N+1 页在做检测。
- 阶段之间按书级串行，因为要切换 llama-server 的模型。
- CPU 侧排版、PNG 编码、CBZ 打包走独立 worker 池，与 GPU 阶段重叠。

页间流水线是“高性能”的主要来源。逐页串行跑完，GPU 会在渲染与编码时空转。

目标数字。

- 单页 8 个气泡，端到端 6 到 10 秒。
- 其中翻译 2 到 4 秒，OCR 1 到 2 秒，擦字 1 到 2 秒。
- 一本书 30 页，含一次模型切换，总时长约 4 到 6 分钟。

这些是估算，必须在目标机器上实测。测量方法见第 12 节。

## 9. 前端与 IPC

### 9.1 绑定

```go
type Library struct{}

func (l *Library) Open(path string) (project.Project, error)
func (l *Library) Pages(id string) ([]project.PageView, error)

func (l *Library) Run(ctx context.Context, id string, progress *mygo.Channel[Progress]) error
func (l *Library) Cancel(id string) error

func (l *Library) UpdateBlock(id string, page, block int, text string) error
func (l *Library) Rerender(id string, page int) (string, error)
func (l *Library) Export(id, outPath string) error
```

`Run` 用 `Channel` 推进度，前端 `for await` 消费。`ctx` 在窗口关闭或前端关闭 channel 时取消，流水线随之停止。

`mygo generate` 从 Go 方法签名生成 TypeScript 客户端。`[]byte` 映射为 base64 字符串，所以图片不走这个方法。

### 9.2 图片传输

注册两个自定义协议，用 `mygo.Protocol.Handle` 加 `mygo.FileServer`。

```go
mygo.Protocol.Handle("page", mygo.FileServer(os.DirFS(cacheDir)))
mygo.Protocol.Handle("out", mygo.FileServer(os.DirFS(outputDir)))
```

前端直接 `<img src="page://localhost/12/original.webp">`。Windows 上 WebView2 把自定义 scheme 挂在 `http://page.localhost/`，`mygo-runtime` 提供平台判断。

这样一页 2 MB 的图不产生 2.7 MB base64，也不占 channel 的 1 MiB 背压窗口。

### 9.3 界面

三个区域。

- 左侧页面列表，虚拟滚动，显示阶段徽标。
- 中间画布，叠加框选与译文，双击可编辑。
- 右侧面板，模型状态、显存占用、单页耗时、批处理设置。

## 10. 打包与分发

### 10.1 运行时文件

两个来源，都是本机现成的。

ORT 1.31 cuda13，从官方 zip 提取三个 DLL 到 `runtime/ort-cuda13`。已提取完成。

| 文件 | 体积 |
| --- | --- |
| `onnxruntime_providers_cuda.dll` | 187.7 MB |
| `onnxruntime.dll` | 16.9 MB |
| `onnxruntime_providers_shared.dll` | 0.04 MB |

zip 本身 295.1 MB，解压 1257.5 MB，其中 1025 MB 是 pdb 调试符号，必须剔除。

CUDA 与 cuDNN 运行时直接引用 torch 的 `lib` 目录，不复制。若将来需要脱离 Python 环境，只复制这 8 个文件，约 895 MB。

| 文件 | 体积 |
| --- | --- |
| `cublasLt64_13.dll` | 455.8 MB |
| `cudnn_engines_precompiled64_9.dll` | 197.2 MB |
| `cudnn_adv64_9.dll` | 102.0 MB |
| `cudnn_heuristic64_9.dll` | 56.2 MB |
| `cublas64_13.dll` | 48.0 MB |
| `cudnn_ops64_9.dll` | 35.0 MB |
| `cudnn_engines_runtime_compiled64_9.dll` | 23.7 MB |
| `cudart64_13.dll` 与其余 cuDNN 子库 | 约 8 MB |

关键事实修正。llama.cpp 的 CUDA 包不含运行库，它需要 `cublas64_13.dll`、`cublasLt64_13.dll`、`cudart64_13.dll`，正好是上面同一套。所以 ORT 与 llama.cpp 共用一份 cu13 运行库，不需要两套。这是选 CUDA 13 而不是 CUDA 12 的主要理由。

安装包体积预估，不含模型权重。

- 应用本体加前端，约 15 MB。
- ORT 三个 DLL，约 205 MB。
- CUDA 与 cuDNN 若引用 torch，为 0。
- 合计约 220 MB。

这是自用场景的好处。若将来要做可分发的安装包，才需要把 895 MB 的 CUDA DLL 打进去。

### 10.2 目录

```
resources/
├─ windows-amd64/
│  ├─ bin/llama-server.exe
│  ├─ bin/*.dll
│  ├─ ort/*.dll
│  └─ runtime/*.dll
├─ fonts/NotoSansSC-subset.otf
└─ models/manifest.json
```

`mygo build` 会合并平台目录进安装包，并对 DLL 签名。

### 10.3 模型下载

首次启动下载，不进安装包。默认目录 `%LOCALAPPDATA%\MangaTranslator\models`，设置里可改到任意盘。

清单含文件名、大小、SHA256。支持断点续传，校验失败删除重下。

网络实测结论。

| 目标 | 代理 | 速度 |
| --- | --- | --- |
| hf-mirror.com | 开 | 5.61 MB/s |
| hf-mirror.com | 关 | 0，连不上 |
| huggingface.co | 开 | 7.35 MB/s |
| huggingface.co | 关 | 0，连不上 |

所以下载器必须支持代理配置，且必须有退避重试。直连 huggingface.co 连续请求会触发 429 限流。

### 10.4 WebView2

Windows 10 1803 之后多数机器自带 Evergreen Runtime。安装包检测，缺失时提示安装。mygo 文档没有明确说明这一点，需要实测。

## 11. 里程碑

| 阶段 | 交付 | 验证方式 |
| --- | --- | --- |
| M0 | 环境验证 | 已完成。CUDA EP 42.6 ms，CPU 8734.9 ms，205 倍 |
| M1 | 骨架 | mygo 项目跑起来，`CGO_ENABLED=1` 加 mingw 构建成功，前端能调 Bind 方法 |
| M2 | 检测链路 | 检测加 NMS 加 mask，输出可视化叠加图，对比 `koharu-slim` 的框 |
| M3 | OCR 链路 | PaddleOCR-VL-For-Manga 通过 llama-server 跑通，与 Manga OCR、PP-OCR 对比准确率与耗时 |
| M4 | 翻译链路 | llama-server 拉起，批量 JSON 输出成功率与单页耗时，验证与 OCR 的串行切换 |
| M5 | 擦字与渲染 | LaMa 加排版，人工对比 20 页 |
| M6 | 端到端 | 一本 30 页 CBZ，记录每阶段耗时与显存峰值 |
| M7 | 打包 | 安装并跑通 |

M0 已完成，结论见第 2.2 节。原来的最大风险点已经消除。

现在最大的风险是 M6，6 GB 显存能否装下 ORT 全部模型，以及 OCR 与翻译串行切换的实际开销。

## 12. 已实测的事实

2026-10-17 本机实测，可复现。

模型 IO 全部十个已探测，见第 4 节各阶段表格。脚本 `.cache/probe_onnx.py`，输出 `.cache/onnx-probe/probe.json`，模型缓存 `.cache/onnx-probe/`。

- 阶段 1 的 ONNX 只有基础算子，无 GridSample，无自定义算子，ORT 直接可跑。已在 CUDA EP 上实测通过，42.6 ms。
- 阶段 2、3、5、7 都含 GridSample 或量化算子。CUDA EP 对 GridSample 的支持要逐个确认。RT-DETR 与 YuzuMarker 用了 `ConvInteger` 与 `MatMulInteger`，CUDA EP 对 int8 量化算子可能回退到 CPU。

这一条很重要。如果 int8 模型在 CUDA 上大量回退，要改用 fp32 版本。`ogkalu/comic-text-and-bubble-detector` 有 fp32 的 `detector.onnx`，`ogkalu/yuzumarker-font-detection-onnx` 有 fp32 的 `font-detector.onnx`。

CUDA EP 已跑通，代码与数据。

- 验证程序 `spike/ort/main.go`，加载 ORT 1.31 cuda13，跑通 `aot.onnx` 与 `comictextdetector.onnx`。
- 对照基准 `spike/ort/epcompare/main.go`，CUDA 42.6 ms，CPU 8734.9 ms。
- 运行时包 `onnxruntime-win-x64-gpu_cuda13-1.31.0.zip`，295.1 MB，已提取三个 DLL 到 `runtime/ort-cuda13`。
- 编译器 winlibs gcc 16.2，已装到 `runtime/toolchain`。

## 13. 风险与未决

按严重程度排序。

1. **显存只有 6 GB，且 OCR 与翻译都要跑 llama.cpp。** 已定串行策略，但串行切换的实际开销未测。若 ORT 的激活峰值与 llama-server 叠加后仍不够，要降阶段 1 输入到 640，或把翻译模型换成 Q3_K_S。
2. **ORT CUDA EP 对 int8 与 GridSample 的支持。** 阶段 2、3 含 `ConvInteger`、`MatMulInteger`、`GridSample`，可能大量回退 CPU。备选是换 fp32 模型，体积与显存都上升。
3. **mygo 在 `CGO_ENABLED=1` 下构建。** 文档说默认 `CGO_ENABLED=0`，可改。M1 实测确认。cgo 编译器已确定为 mingw-w64。
4. **PaddleOCR-VL-For-Manga 的实际识别质量。** 上游是第三方微调，作者只给了定性结论。M3 要用真实漫画页对比 Manga OCR 与 PP-OCR。
5. **Hy-MT2 的批量 JSON 输出稳定性。** 官方模板面向单条文本，批量格式需要 few-shot 加 schema 约束。成功率待实测。
6. **竖排排版细节。** canvas 已解决主体，剩余是自动字号二分、气泡内缩、黑白反色。纯工程量，无技术风险。
7. **依赖本机 torch 的 DLL。** 自用场景可接受。若 torch 被升级到 cu14，DLL 名会变，需要重新指向。缓解办法是记下当前版本 `2.12.0+cu130`，或改为复制 895 MB 到项目目录。

## 14. 已定的产品决策

1. 只做 Windows，只做自用。CUDA 是唯一 GPU 路径，不做 DirectML 与 Metal 适配。
2. OCR 用 PaddleOCR-VL-For-Manga，跑在 llama.cpp 上。不用 Manga OCR，因为那要在 Go 里写自回归解码，而 llama.cpp 已有现成方案。
3. OCR 与翻译按书级串行。先跑完全部 OCR，卸载模型，再加载翻译模型。不并行常驻。
4. CUDA 运行时直接引用本机 torch 的 DLL 路径，不复制文件。
5. 需要人工校对界面，因为要处理 OCR 与翻译的错误。
6. 排版渲染用 `tdewolff/canvas`，已实测竖排可行。
7. 不用 FLUX.2 做生成式修补。Koharu 里有 `flux2_klein`（4B 扩散模型）但 CLI 主流程没引用它。LaMa 加气泡填色已够，省 2.3 GB 显存。

## 15. 复现清单

本机已就位的资源，全部在 E 盘。

| 路径 | 内容 |
| --- | --- |
| `runtime/ort-cuda13/` | ORT 1.31 的三个 DLL |
| `runtime/toolchain/mingw64/` | winlibs gcc 16.2 |
| `.cache/onnx-probe/` | 10 个探测过的 ONNX 模型，约 0.9 GB |
| `.cache/probe_onnx.py` | 模型 IO 探测脚本 |
| `.cache/canvas-out/` | canvas 竖排验证输出图 |
| `spike/ort/` | cgo 加 CUDA EP 验证程序 |
| `spike/ort/epcompare/` | CUDA 与 CPU 对照基准 |
| `spike/canvas/` | tdewolff/canvas 竖排验证程序 |

构建命令。

```
set CGO_ENABLED=1
set CC=E:\work\transl\runtime\toolchain\mingw64\mingw64\bin\gcc.exe
go build -o ortspike.exe .
```

运行前需要两个 DLL 目录在 PATH 里，程序内部已设置。
