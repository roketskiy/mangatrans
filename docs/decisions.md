# 设计决策记录

## 2026-10-18 启动实现

产品要求：自主迭代出功能齐全的成品，不做 UI 测试，端到端输入输出都是 zip/cbz，必须亲自检查翻译效果。

### 架构选择

单进程 Go 程序，不先做 mygo 外壳。理由：

先做出能跑的端到端流水线，再包 UI。UI 是薄壳，流水线是核心。如果先做 UI，每次验证都要点界面，迭代慢。

程序形态：`mangatrans.exe <input.cbz> <output.cbz>`。后续加 `--serve` 给 mygo 前端用。

### 流水线阶段

| # | 阶段 | 实现 | 状态 |
| --- | --- | --- | --- |
| 0 | 解包 CBZ | archive/zip 标准库 | 待做 |
| 1 | 文本检测 | ORT + comictextdetector.onnx | 待做 |
| 2 | 气泡分割 | ORT + YOLOv8m-seg | 待做 |
| 3 | 擦字 | ORT + LaMa | 待做 |
| 4 | OCR | llama-server + PaddleOCR-VL-For-Manga | 待做 |
| 5 | 翻译 | llama-server + Hy-MT2 | 待做 |
| 6 | 排版渲染 | tdewolff/canvas | 待做 |
| 7 | 打包 CBZ | archive/zip 标准库 | 待做 |

阶段 2 版面分类（PP-DocLayoutV3）与阶段 3 字体属性（YuzuMarker）先跳过，用规则替代。理由：它们各占 100 到 125 MB 显存，而收益是过滤噪声框和优化排版。先用简单规则，看效果再决定。

### 关键约束

- 模型与缓存只在 E 盘。
- 测试素材 `E:\edge_download\少年ジャンプ＋ (Shonen Jump +)\11歳になってしまう\11歳になってしまう.cbz`。
- 端到端输入输出都是 cbz。
- 不做 UI 测试。

### 待验证

- PaddleOCR-VL-For-Manga 在真实漫画页上的识别质量。
- Hy-MT2 批量 JSON 翻译的稳定性。
- 显存峰值。

## 技术事实（已实测）

| 事实 | 证据 |
| --- | --- |
| ORT 1.31 cuda13 加 torch 的 cu13 DLL，CUDA EP 生效 | `spike/ort/epcompare`，42.6 ms 对 CPU 8734.9 ms |
| Go cgo 必须用 mingw-w64，不能用 MSVC 或 clang | 三次构建失败，`runtime/cgo` 传 gcc 参数 |
| canvas 竖排必须先 `SetWritingMode` 再 `ToText` | `spike/canvas`，第一次失败第二次成功 |
| llama.cpp 0.7.0 CUDA 版在本机可用 | `llama-server.exe --version` |
