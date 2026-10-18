# mangatrans

把日文漫画（`.cbz` / `.zip`）翻译成简体中文，输出 `.cbz`。全部推理在本机完成，使用 CUDA。

## 流水线

1. **检测**：ComicTextDetector（ONNX）一次前向得到文本框、文本分割图和 DBNet 概率图。掩码按 koharu-slim 的做法融合 `max(sigmoid((shrink-thresh)*50), unet)`，再 close(10) + dilate(3)。
2. **OCR**：PaddleOCR-VL-For-Manga（GGUF，llama.cpp）逐框识别，保留竖排和拟声词。
3. **翻译**：Hy-MT2-1.8B（GGUF，llama.cpp）按页批量翻译，JSON schema 约束输出；缺失的 id 单独重试。仍为空、原样返回或带假名的回复会被拒绝，该框保留原文，避免把日文当成译文画上去。
4. **擦字**：LaMa（ONNX）按连通域分窗修补；纯色气泡走快速填充，不跑模型。单页擦字或排版失败时整页回退为原图，不会在未擦干净的原图上叠字，也不会中断整本任务。
5. **排版**：先用连通域找出气泡内部，把文字排进气泡而不是 OCR 的紧框；再用 tdewolff/canvas 逐框渲染。竖排用 `VerticalRL + Upright`，自动选字号、按字数折列、避头尾标点，并按背景明暗自动选黑/白字。
6. **打包**：保持原文件名和顺序写回 CBZ。

OCR 和翻译串行使用 GPU：OCR 模型先卸载，再加载翻译模型。

## 环境要求

- Windows
- CUDA 13 + cuDNN（本机用的是 `E:\Python312\Lib\site-packages\torch\lib` 里的 DLL）
- Go 1.26，cgo 用 mingw-w64（MSVC 不行）
- llama.cpp 的 `llama-server.exe`（CUDA 构建）
- ONNX Runtime **cuda13** 构建（`onnxruntime.dll`、`onnxruntime_providers_cuda.dll`、`onnxruntime_providers_shared.dll`）

## 模型

默认目录是 `E:\work\transl\models` 和 `E:\work\transl\.cache\onnx-probe`，可以用环境变量改。

| 用途 | 文件 | 来源 |
| --- | --- | --- |
| 检测 | `comictextdetector.onnx` (90 MB) | `mayocream/koharu` |
| 擦字 | `lama-manga.onnx` (198 MB) | `mayocream/lama-manga-onnx` |
| OCR | `PaddleOCR-VL-manga-Q4_K_M.gguf` (286 MB) | `LunarOilRig/PaddleOCR-VL-manga-GGUF` |
| OCR 视觉塔 | `mmproj-Q8_0.gguf` (570 MB) | 同上 |
| 翻译 | `Hy-MT2-1.8B-UD-Q4_K_XL.gguf` (1.1 GB) | `unsloth/Hy-MT2-1.8B-GGUF` |

下载走代理：

```powershell
$env:HTTP_PROXY='http://127.0.0.1:7897'; $env:HTTPS_PROXY=$env:HTTP_PROXY
curl.exe -L -o models\Hy-MT2-1.8B-UD-Q4_K_XL.gguf `
  https://huggingface.co/unsloth/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-UD-Q4_K_XL.gguf
```

## 构建

```powershell
.\build.ps1
```

脚本会设置 mingw 的 `CC` 和 `PATH`，然后产出 `bin\mangatrans.exe`。

手动构建：

```powershell
$env:CGO_ENABLED='1'
$env:CC='E:\work\transl\runtime\toolchain\mingw64\mingw64\bin\gcc.exe'
$env:PATH='E:\work\transl\runtime\toolchain\mingw64\mingw64\bin;'+$env:PATH
go build -o bin\mangatrans.exe .\cmd\mangatrans
```

## 使用

```powershell
.\bin\mangatrans.exe -in "book.cbz" -out "book.zh.cbz"
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `-limit N` | 只处理前 N 页，调试用 |
| `-work DIR` | 保存每页输出、擦字掩码和 OCR/译文 JSON |
| `-conf 0.35` | 检测置信度阈值 |
| `-mt-model PATH` | 翻译 GGUF |
| `-font PATH` | 译文字体，默认 `msyhbd.ttc` |
| `-glossary PATH` | 术语表，JSON 映射或每行 `原文=译文` |
| `-bench-json PATH` | 结束时把分阶段耗时写成 JSON |
| `-perf-json PATH` | 高精度事件、请求和逐页结果，供新性能脚本分析 |
| `-ocr-workers N` | OCR 请求并发数，范围 1 到 4，默认 1；ONNX 检测会话仍串行 |
| `-mt-batch-pages N` | 一个 12 块翻译批次最多覆盖的连续页数，范围 1 到 4，默认 2 |

环境变量（都可选）：

`MANGATRANS_LLAMA`、`MANGATRANS_OCR_MODEL`、`MANGATRANS_MMPROJ`、`MANGATRANS_DET_MODEL`、`MANGATRANS_LAMA_MODEL`、`MANGATRANS_ORT_DIR`、`MANGATRANS_CUDA_DIR`

术语表示例：

```powershell
.\bin\mangatrans.exe -in book.cbz -out book.zh.cbz -glossary glossary.example.json
```

只把当前批次里出现的词条注入提示词，不会拖慢其他页。

## 桌面界面

`mangatrans-ui.exe` 是 MyGo 写的原生窗口，只负责选文件、显示进度和日志；翻译仍由同目录的 `mangatrans.exe` 完成。

```powershell
.\build.ps1       # 先构建引擎
.\build-ui.ps1    # 再构建界面
.\bin\mangatrans-ui.exe
```

界面按 Windows GUI 子系统构建（`-ldflags '-H=windowsgui'`），启动时不会创建控制台。引擎和它拉起的 `llama-server.exe` 都使用 `CREATE_NO_WINDOW`；点“停止”或关闭窗口时会取消任务，并用 `taskkill /PID <pid> /T /F` 清理整棵进程树。`.\test-shutdown.ps1` 会启动一次引擎、等到 `llama-server` 子进程出现，再执行同样的清理并检查残留。

UI 还创建带 `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 的 Job Object，并把引擎加入该 job。即使从任务管理器强制结束 UI，Windows 关闭 job handle 时也会连带结束引擎和 `llama-server`。`.\bin\mangatrans-ui.exe -job-selftest` 会做一次隐藏的进程树测试，结果写到 `.cache\out\job-selftest.json`。

界面构建需要 Go 1.27（脚本使用 `.cache\go1.27`），引擎仍用 Go 1.26 + cgo。两个 exe 必须放在同一目录，或者用 `MANGATRANS_EXE` 指向引擎。

进度不是从日志里的英文单词猜的：引擎会发出 `progress ocr|translate|inpaint|save|done N/M` 事件，界面按阶段权重和当前页数换算。四个阶段（识别文字、翻译、擦除排版、保存校验）分别显示为进行中、完成或失败；失败时进度条变红并停在真实阶段，不会停在虚假的 90%。日志自动跟随末尾，向上滚动时会暂停跟随；长路径和 panic 栈会换行显示，失败原因也会在左侧卡片中给出摘要。

中间产物只有两类，都紧挨输出文件：原子写入的 `<输出>.part`，以及本次运行的日志目录 `.<输出文件名>.mangatrans`（放 OCR/翻译服务器日志）。正常结束、出错和 panic 都会删除；即使 UI 被强制结束，下一次开始翻译前也会先清掉这两个残留。页面图像只在内存中处理，不会解压到临时目录。

## 性能测试

独立测量脚本接受任意 CBZ/ZIP：

```powershell
python -m pip install -r .\scripts\requirements-bench.txt
# 先重新构建引擎，使 exe 支持 -perf-json
python .\scripts\bench.py "D:\漫画\输入.cbz" --runs 5
```

脚本读取新加入执行路径的单调时钟事件，不使用旧分阶段统计。它同时记录引擎及子进程的 CPU、I/O、内存和整卡 GPU 采样，校验实际输出，并单列失败、回退、未译块及未埋点时间。每轮保留原始数据，生成 JSON 和 Markdown 报告。

配对比较请用 `scripts/compare_bench.py`，它交错 before/after 顺序并剔除逐页工作量不一致的配对。至少完成 5 个有效配对后再看描述性结果；该报告不验证译文语义，也只记录整卡 GPU 数据。

参数、构建命令、测量口径与限制见 [性能测试说明](docs/performance.md)。

## 已知限制

- 标题、装饰字、拟声词、手写招牌常常检测不到，会保留日文。
- 检测框可能把相邻两个气泡合并成一个，译文会混在一起。
- 1.8B 翻译模型的润色能力有限，长句和语气词偶尔生硬。
- 译文按原文字框排版；原文很短的框装不下长译文时，字号会偏小。
- 不规则气泡（例如测试书第 6 页顶部）里的超长译文仍可能压到黑色背景；矩形排版无法完全贴合斜边。
- 输出统一 JPEG 质量 92，不保留 PNG 无损。
