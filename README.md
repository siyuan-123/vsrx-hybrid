# VSR-X Hybrid 1.0.0

VSR-X Hybrid 是一套面向**大量本地视频**的硬字幕去除系统。设计目标不是让单一大模型处理所有帧，而是用最低成本完成大多数片段，只把真正困难的区域交给较重的模型。

核心策略：

```text
媒体探测与软字幕快速路径
        ↓
镜头切分
        ↓
PP-OCRv6 detection-only + 多 ROI 发现
        ↓
Hungarian + Kalman 字幕轨迹
        ↓
描边/阴影/发光/底板精细掩膜
        ↓
运动补偿 TBE / Clean Plate，优先恢复真实背景
        ↓
按片段路由：COPY / TBE / Telea / LaMa / ProPainter / STTN
        ↓
残字、闪烁、接缝、锐度和掩膜外像素质量门
        ↓
局部重试、FFV1 检查点、断点续跑
        ↓
一次最终有损编码并复用原音频和元数据
```

## 适用场景

- 电影、电视剧、课程、访谈、Vlog、动画、游戏录屏等硬字幕视频。
- 单行、双行、双语、顶部、竖排、移动、卡拉 OK、带描边/阴影/底板的字幕。
- 一次处理大量文件，需要作业数据库、断点续跑、局部失败重试和审计报告。
- 非商业研究或个人使用，可以另行安装官方 ProPainter 作为困难片段后端。

## 关键特点

- **真实背景优先**：相邻帧可见时恢复真实像素，不让生成模型无谓猜测。
- **低资源路由**：空掩膜直接复制；小洞用 OpenCV；普通残洞才用 LaMa；困难 ROI 才用 ProPainter。
- **ROI-only**：OCR、光流和重模型尽量只处理字幕附近区域。
- **检测与识别分离**：默认只运行 PP-OCRv6 文字检测，不加载文字识别网络。
- **掩膜外严格保护**：所有修复后端最终通过精确合成，硬掩膜外像素在无损检查点中保持不变。
- **可恢复**：SQLite WAL、分段 FFV1 检查点、输入/配置/模型/外部掩膜哈希。
- **容器安全编码**：硬件编码失败自动转软件；音频 copy 不兼容时自动转 AAC/Opus。
- **无权重也可运行**：缺少 OCR/LaMa/ProPainter 时有启发式检测和 OpenCV 降级，不会让整个系统不可用。
- **外部掩膜模式**：可绕过 OCR，适合确定性处理、人工修正、测试和第三方检测器接入。

## 当前实现状态

项目已实现并验证：

- FFprobe 完整流探测、VFR/PTS/HDR 字段读取。
- 软字幕移除与提取。
- PySceneDetect 自适应切镜。
- PP-OCRv6 detection-only RapidOCR 适配器与启发式降级。
- 多区域发现、Hungarian + Kalman 跟踪、字幕/Logo/场景文字保护。
- 字形、描边、阴影、发光、半透明底板与时间并集掩膜。
- 相位相关、ECC、ORB、DIS 局部光流、曝光补偿与加权中位数 TBE。
- Telea、Navier-Stokes、LaMa ONNX、MI-GAN ONNX、官方 ProPainter、官方 STTN 适配器。
- 显存预算、单 GPU 跨进程锁和 OOM 降级。
- 自动质量门与重试阶梯。
- SQLite 状态机、FFV1 检查点、输出 SHA-256 清单。
- CLI、目录批处理、REST API、模型诊断和安装命令。
- 纯 Markdown 架构规范和实现文档。

测试状态见 [docs/TESTING.md](docs/TESTING.md)。

## 环境要求

基础要求：

- Python 3.11 或更高版本。
- FFmpeg 和 FFprobe 可在 `PATH` 中执行。
- 8 GB 以上系统内存；长视频推荐 16 GB 以上。
- NVIDIA GPU 不是必需。TBE、Telea、CPU ONNX 路径可以纯 CPU 运行。

可选：

- RapidOCR + ONNX Runtime/OpenVINO：PP-OCRv6 detection-only。
- LaMa/MI-GAN ONNX 权重：残洞修补。
- 官方 ProPainter 仓库和权重：困难片段高质量修复。
- 官方 STTN 仓库和 checkpoint：低显存视频模型降级。

神经网络权重未包含在代码包中，原因包括文件体积、来源校验和第三方许可证。安装后应运行 `vsrx models verify` 并在 `configs/model_manifest.yaml` 固定 SHA-256。

## 安装

### Linux/macOS

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv git

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[ocr,test]'
```

纯 CPU/OpenVINO 可改为：

```bash
python -m pip install -e '.[openvino,test]'
```

仅安装基础功能：

```bash
python -m pip install -e .
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ocr,test]"
```

另行安装 FFmpeg，并确认：

```powershell
ffmpeg -version
ffprobe -version
```

## 快速开始

### 单文件

```bash
vsrx process input.mkv -o output.mkv --profile balanced
```

### 批量目录

```bash
vsrx process ./videos --output ./cleaned --profile fast
```

### 固定字幕区域

ROI 使用 `x1,y1,x2,y2`：

```bash
vsrx process input.mp4 -o output.mp4 \
  --roi 80,720,1840,1060 \
  --profile balanced
```

多个区域：

```bash
vsrx process input.mp4 -o output.mp4 \
  --roi 80,40,1840,260 \
  --roi 80,720,1840,1060
```

### 外部逐帧掩膜

```bash
vsrx process input.mkv -o output.mkv \
  --mask-dir ./masks \
  --profile fast
```

推荐命名：

```text
masks/
├── 00000000.png
├── 00000001.png
├── frame_00000002.png
└── pts_125000.png
```

外部掩膜模式会跳过 OCR 和轨迹构建，缺失帧视为空掩膜。VFR 视频推荐使用 `pts_<微秒>.png` 或 `masks.json`。详见 [docs/EXTERNAL_MASKS.md](docs/EXTERNAL_MASKS.md)。

### 分析但不处理

```bash
vsrx analyze input.mkv --profile balanced
```

### 配置覆盖

```bash
vsrx process input.mkv -o output.mkv \
  --profile balanced \
  --set clean_plate.max_selected_reference_frames=6 \
  --set quality_control.max_automatic_attempts_per_segment=3
```

### 查看与取消作业

```bash
vsrx jobs
vsrx cancel <job_id>
```

## 模型管理

```bash
vsrx models status
vsrx models verify
vsrx models install-ocr
vsrx models install-ocr --gpu
vsrx models install-lama --source /path/to/lama.onnx
vsrx models install-propainter
vsrx models install-sttn --checkpoint /path/to/sttn.pth
```

模型环境变量：

```bash
export VSRX_LAMA_MODEL=/models/lama.onnx
export VSRX_MIGAN_MODEL=/models/migan.onnx
export VSRX_PROPAINTER_REPO=/models/ProPainter
export VSRX_STTN_REPO=/models/STTN
export VSRX_STTN_CHECKPOINT=/models/STTN/checkpoints/sttn.pth
```

详见 [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md)。

## REST API

```bash
vsrx serve --host 127.0.0.1 --port 8765 --profile balanced
```

提交：

```bash
curl -X POST http://127.0.0.1:8765/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "input_path": "/data/input.mkv",
    "output_path": "/data/output.mkv",
    "codec": "auto",
    "device_index": 0
  }'
```

服务端建议设置允许访问的根目录：

```bash
export VSRX_ALLOWED_ROOTS=/data:/mnt/archive
```

接口详见 [docs/API_REFERENCE.md](docs/API_REFERENCE.md)。

## 配置档位

| 档位 | 默认检测 | 参考帧 | 光流 | 质量门 | 适用场景 |
|---|---|---:|---|---|---|
| `fast` | Small，较低频 | 最多 6 | DIS ultrafast | 完整但轻量 | 海量普通视频 |
| `balanced` | Small | 最多 8 | DIS fast | 完整 | 默认推荐 |
| `quality` | Small + Medium 重检 | 最多 12 | DIS medium | 更严格 | 精品片段 |
| `cpu_economy` | Tiny | 较少 | CPU | 降低重试 | 无独显设备 |

配置层级：

```text
内置 balanced.yaml
  + profile overlay
  + --overlay YAML
  + --set key=value
```

## 项目结构

```text
src/vsrx/
├── app/          CLI、API、配置加载、总控制器、模型管理
├── media/        probe、解码、检查点编码、最终编码、字幕流
├── scene/        镜头切分与转场处理
├── detection/    PP-OCRv6 detection-only 与 ROI 发现
├── tracking/     Hungarian、Kalman、轨迹分类
├── mask/         精细掩膜、面板、时间并集、外部掩膜
├── motion/       全局配准、DIS 光流与置信度
├── cleanplate/   参考选择、曝光补偿、鲁棒融合
├── routing/      特征、路由、显存预算
├── inpaint/      Telea、LaMa、MI-GAN、ProPainter、STTN
├── quality/      残字、闪烁、接缝、保护不变量和重试
├── scheduler/    SQLite、状态机、检查点、GPU 锁
└── reporting/    JSON/Markdown 审计
```

详细架构见：

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/ALGORITHM_DETAILS.md](docs/ALGORITHM_DETAILS.md)
- [docs/spec/VSRX_Hybrid_Architecture_Spec_v1.1_CN.md](docs/spec/VSRX_Hybrid_Architecture_Spec_v1.1_CN.md)

## 测试与质量检查

```bash
ruff check src tests
pytest -q
python -m build
```

当前自动测试覆盖：

- 配置合并与稳定哈希。
- 外部掩膜索引、PTS、Manifest 和 JPEG 噪声阈值。
- 全局运动配准。
- TBE 真实背景恢复。
- 路由决策。
- Telea 与掩膜外像素保护。
- 轨迹关联和缺帧插值。
- 质量门。
- SQLite 幂等与输出路径更新。
- FFV1 检查点往返。
- API 健康检查。
- 真实小视频首次处理与断点复用集成测试。

## 性能原则

1. 先检测是否有独立字幕流；能 remux 就不运行 AI。
2. 所有参考帧严格限制在同一镜头。
3. 相邻帧运动只计算一次，跨帧变换通过组合得到。
4. 全局配准足够好时不运行局部稠密光流。
5. 达到覆盖率和置信度阈值时提前停止参考搜索。
6. 重模型只处理残余掩膜的联合 ROI。
7. 每张 GPU 同一时刻只允许运行一个 GPU 修复后端。
8. 每个片段完成后立即生成可验证检查点。
9. 只有最终输出做一次有损编码。

## 重要限制

- 当字幕在整个镜头中永久遮挡某区域，且真实内容从未在其他帧出现时，任何方案都只能生成合理内容，无法知道原始真实像素。
- 满屏弹幕、大面积滚动字幕、长时间半透明遮罩、快速复杂前景穿越会明显增加计算量，并更依赖 ProPainter。
- HDR 处理会保留已探测到的色彩元数据，但模型本身通常在 8-bit BGR 域运行；严格 HDR 母版工作流需要额外的线性光/高位深模型校准。
- 第三方模型和权重需单独遵守其许可证。本项目不重新分发这些权重。
- 自动质量门用于阻止明显失败静默通过，不等于数学上保证每个生成像素与被遮挡原始画面完全一致。

## 许可证

VSR-X Hybrid 自有代码使用 Apache-2.0。第三方项目与权重许可证独立生效，详见：

- [NOTICE.md](NOTICE.md)
- [docs/THIRD_PARTY_LICENSES.md](docs/THIRD_PARTY_LICENSES.md)
