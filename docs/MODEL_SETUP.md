# 模型与运行时准备

核心包不包含神经网络权重。没有权重时 TBE、Telea 和启发式检测仍可运行，但复杂字幕建议至少安装 PP-OCRv6 detection-only。

## 1. OCR

CPU ONNX Runtime：

```bash
python -m pip install -e '.[ocr]'
vsrx models install-ocr
```

CUDA：

```bash
python -m pip install -e '.[ocr-gpu]'
vsrx models install-ocr --gpu
```

OpenVINO：

```bash
python -m pip install -e '.[openvino]'
```

运行 `vsrx doctor` 确认 provider。不要同时安装互相冲突版本的 `onnxruntime` 与 `onnxruntime-gpu`。
CUDA 12.x 环境使用 `onnxruntime-gpu>=1.21,<1.27`；RapidOCR 的 CUDA 候选会显式启用
`CUDAExecutionProvider`，初始化失败时再降级到 CPU。

## 2. LaMa ONNX

VSR-X 接受带 image 与 mask 两个输入的常见 LaMa ONNX 图，支持动态尺寸或固定 NCHW 尺寸。

```bash
vsrx models install-lama --source /trusted/lama.onnx
export VSRX_LAMA_MODEL=$PWD/models/lama.onnx
```

也可直接设置：

```bash
export VSRX_LAMA_MODEL=/absolute/path/lama.onnx
```

安装器拒绝异常小文件。首次正式使用前必须用 `vsrx models verify` 记录实际 SHA-256，并把哈希写进 `configs/model_manifest.yaml` 的本地副本。

本项目已验证 `Carve/LaMa-ONNX` 的 `lama_fp32.onnx` 双输入图。官方 LaMa 仓库不发布
ONNX 权重，替换导出图时仍需核对 `image`、`mask` 输入和输出数值范围。

## 3. MI-GAN

可选 CPU 经济模型：

```bash
export VSRX_MIGAN_MODEL=/absolute/path/migan.onnx
```

并启用：

```bash
--set spatial_inpainting.migan.enabled=true
```

不同导出图的输入/输出可能不同；必须先用测试样本确认。默认关闭。
`migan_pipeline_v2.onnx` 使用 uint8 RGB 输入，并以 mask 的 255 表示保留区、0 表示修复区；
适配器会自动完成该语义转换。

## 4. 官方 ProPainter

用途：低覆盖、人物穿越字幕、复杂运动等困难 ROI。仅限其许可证允许的非商业研究/个人场景。

```bash
vsrx models install-propainter --root ./models
export VSRX_PROPAINTER_REPO=$PWD/models/ProPainter
vsrx models verify --root ./models
```

准备完成的目录至少应有：

```text
ProPainter/
├── inference_propainter.py
└── weights/
    └── ...官方权重文件...
```

VSR-X 不修改上游模型代码；通过其 CLI 运行。上游参数变化时，需要更新 `OfficialProPainterInpainter` 适配器并锁定 commit。

建议：

- PyTorch/CUDA 环境单独使用虚拟环境或容器；
- 8 GB 显存严格使用 ROI 和 16～32 帧 chunk；
- 不要整帧处理 1080p；
- 首次运行用 5～10 秒样本校准显存。

## 5. 官方 STTN

```bash
vsrx models install-sttn \
  --checkpoint /trusted/sttn.pth \
  --root ./models
export VSRX_STTN_REPO=$PWD/models/STTN
export VSRX_STTN_CHECKPOINT=$PWD/models/STTN/checkpoints/sttn.pth
```

STTN 是紧急降级，不应替代默认 TBE 主路径。
原始仓库脚本硬编码第二张 GPU；VSR-X 会在临时副本中改为当前可见的第一张 GPU，并从
原始脚本生成的 `masks_result.mp4` 读取结果，不修改上游仓库文件。

## 6. Windows 本地环境

完整安装后可用以下脚本同时激活虚拟环境和模型路径：

```powershell
.\scripts\activate.ps1
vsrx doctor --models-root .\models
```

## 7. 模型身份与缓存

作业缓存包含：

- `configs/model_manifest.yaml` 内容；
- 已安装 OCR、ONNX Runtime、OpenVINO、OpenCV、PyTorch 版本；
- LaMa/MI-GAN/STTN 权重 SHA-256；
- ProPainter/STTN 仓库 Git commit；
- 权重树文件大小和修改时间身份。

更换权重或上游仓库后会生成新的作业身份，避免误用旧检查点。

## 8. 供应链建议

- 只从上游官方仓库或可信内部镜像下载；
- 保存 URL、commit、文件大小、SHA-256、许可证快照；
- 不执行来源不明的模型下载脚本；
- 模型仓库建议只读挂载；
- 不把第三方权重提交到本项目 Git。
