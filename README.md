# mangatrans

把日文漫画（`.cbz` / `.zip`）翻译成简体中文，输出 `.cbz`。全部推理在本机完成，使用 CUDA。

## 流水线

1. **检测**：ComicTextDetector（ONNX）一次前向得到文本框、文本分割图和 DBNet 概率图。掩码按 koharu-slim 的做法融合 `max(sigmoid((shrink-thresh)*50), unet)`，再 close(10) + dilate(3)。
2. **OCR**：PaddleOCR-VL-For-Manga（GGUF，llama.cpp）逐框识别，保留竖排和拟声词。
3. **翻译**：Hy-MT2-1.8B（GGUF，llama.cpp）按页批量翻译，JSON schema 约束输出；缺失的 id 单独重试，仍失败就保留原文。
4. **擦字**：LaMa（ONNX）按连通域分窗修补；纯色气泡走快速填充，不跑模型。
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

界面构建需要 Go 1.27（脚本使用 `.cache\go1.27`），引擎仍用 Go 1.26 + cgo。两个 exe 必须放在同一目录，或者用 `MANGATRANS_EXE` 指向引擎。

## 性能基准

整本 11 页（822x1200），CUDA 13，Hy-MT2-1.8B Q4：

| 阶段 | 耗时 | 吞吐 |
| --- | --- | --- |
| 检测 | 1.73 s | 157 ms/页 |
| OCR | 8.30 s | 133 ms/框，7.5 框/s |
| 翻译 | 8.46 s | 136 ms/块，7.3 块/s |
| 擦字 | 12.81 s | 1.16 s/页 |
| 排版 | 0.59 s | 53 ms/页 |
| 模型加载 | 12.63 s | 一次性 |
| 总计 | 45.4 s | 4.13 s/页，14.5 页/分钟 |

`.\bench.ps1` 会重跑这本测试书，并在输出 CBZ 旁写一份 `bench.json`，字段是各阶段毫秒数、页数、框数和字数。

## 实测

测试书 1：`11歳になってしまう`，11 页 822x1200。OCR 约 1 秒/页，整本约 1 分钟。
测试书 2：`『刀剣乱舞-花丸-』第2話`，34 页，整本约 2 分钟。

## 已知限制

- 标题、装饰字、拟声词、手写招牌常常检测不到，会保留日文。
- 检测框可能把相邻两个气泡合并成一个，译文会混在一起。
- 1.8B 翻译模型的润色能力有限，长句和语气词偶尔生硬。
- 译文按原文字框排版；原文很短的框装不下长译文时，字号会偏小。
- 输出统一 JPEG 质量 92，不保留 PNG 无损。
