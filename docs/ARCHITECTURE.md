# VSR-X Hybrid 实现架构

本文描述当前代码的真实实现，不是愿景文档。完整研究与决策依据位于 `docs/spec/`。

## 1. 设计目标与不变量

VSR-X 的优化目标按优先级排序如下：

1. 不静默交付明显失败结果；
2. 能从检查点恢复，单个片段失败不丢失整条视频；
3. 优先恢复相邻帧中的真实像素；
4. 只在必要的帧和 ROI 上运行神经网络；
5. 掩膜外像素必须保持不变；
6. 音频、PTS、VFR、HDR 标记和元数据尽可能保留；
7. 没有可选模型时仍能运行，并明确记录降级路径。

任何修改都不得破坏以下不变量：

- 镜头之间不共享参考帧、轨迹、光流或生成状态；
- 生成模型输出只能通过 `composite_exact()` 写入掩膜区域；
- 作业缓存键必须包含输入、配置、模型与外部掩膜身份；
- 完成的输出必须有 SHA-256 产物记录；
- OOM 只能触发当前片段降级，不能破坏已完成片段；
- VFR 视频的时间身份使用微秒 PTS，而不是假设固定帧率。

## 2. 逻辑分层

```text
CLI / REST API
    |
    v
VSRXPipeline：作业编排、状态转换、检查点、错误边界
    |
    +-- Media：probe / decode / soft subtitle / final encode
    +-- Scene：镜头检测与过渡保护
    +-- Detection：抽样、PP-OCRv6 detection-only、ROI 发现
    +-- Tracking：Hungarian + Kalman、轨迹分类
    +-- Mask：精细概率掩膜、底板、时间并集、外部掩膜
    +-- Motion：全局配准、DIS 光流与置信度
    +-- CleanPlate：参考帧选择、曝光补偿、鲁棒融合
    +-- Routing：特征、显存预算、最低成本路径选择
    +-- Inpaint：Telea / LaMa / MI-GAN / ProPainter / STTN
    +-- Quality：残字、闪烁、接缝、锐度、保护区检查
    +-- Scheduler：SQLite WAL、GPU 锁、FFV1 检查点
    +-- Reporting：JSON/Markdown 审计报告
```

## 3. 代码目录与职责

| 路径 | 责任 | 禁止承担的责任 |
|---|---|---|
| `src/vsrx/app/pipeline.py` | 流水线编排、状态机、重试、最终输出 | 不直接实现 OCR、光流或模型算法 |
| `src/vsrx/domain/` | 稳定领域对象、枚举、错误类型 | 不访问文件、数据库、模型 |
| `src/vsrx/media/` | FFmpeg/PyAV 输入输出、PTS 与容器兼容性 | 不决定字幕路由 |
| `src/vsrx/scene/` | 镜头边界 | 不跨镜头传播状态 |
| `src/vsrx/detection/` | 文字检测与 ROI 发现 | 不做文字识别或背景生成 |
| `src/vsrx/tracking/` | 检测关联与轨迹分类 | 不直接修改图像 |
| `src/vsrx/mask/` | 字幕 matte | 不选择修复模型 |
| `src/vsrx/motion/` | 几何与稠密运动 | 不执行生成式修复 |
| `src/vsrx/cleanplate/` | 真实背景恢复 | 不读取外部数据库 |
| `src/vsrx/routing/` | 路由和预算 | 不实现具体模型 |
| `src/vsrx/inpaint/` | 后端适配与精确合成 | 不决定业务重试 |
| `src/vsrx/quality/` | 片段质量门 | 不直接写最终视频 |
| `src/vsrx/scheduler/` | 状态、检查点、资源锁 | 不包含算法阈值 |
| `src/vsrx/reporting/` | 审计报告 | 不改变作业状态 |

## 4. 一条作业的实际执行顺序

1. 解析配置档位与 CLI 覆盖，计算配置哈希。
2. `ffprobe` 读取全部流、时间基、帧率、色彩和字幕轨。
3. 计算输入身份以及已安装模型的运行时身份。
4. 创建或复用 SQLite 作业；验证已有最终输出是否属于该作业。
5. 若只有软字幕且未强制扫描，直接 remux 删除字幕流。
6. 解码视频，保留每帧的 `pts_us`、持续时间和关键帧属性。
7. 按镜头切分；每个镜头独立建立检测、轨迹和参考帧范围。
8. 使用固定 ROI、外部掩膜或自动发现路径得到字幕区域。
9. 构建字幕轨迹并保护 Logo、场景文字和低置信度轨迹。
10. 生成逐帧概率掩膜及硬掩膜。
11. 将镜头拆成有上下文的时间片段。
12. 为片段计算相邻帧变换缓存与 clean plate。
13. 根据覆盖率、置信度、残洞和运动特征选择路径。
14. 运行最轻可行后端；质量门失败时只升级当前片段。
15. 每个片段输出 FFV1 无损检查点，并将路径、哈希和 QC 写库。
16. 按原始顺序重组片段，一次最终有损编码，复用原音频与元数据。
17. 重新 probe 输出，写入 `audit.json`、`audit.md` 和产物 SHA-256。

## 5. 进程与资源模型

基础实现是单作业进程内编排，重模型通过外部子进程隔离：

```text
主进程
  - 解码 / OCR / TBE / QC / SQLite
  - LaMa/MI-GAN ONNX Runtime 会话惰性加载
  - GPU 文件锁

ProPainter 子进程
  - 只读取当前困难 ROI 的帧序列和掩膜
  - CUDA_VISIBLE_DEVICES 固定设备
  - 超时、OOM、退出码均转为结构化错误

STTN 子进程
  - 仅作低显存降级
  - 与 ProPainter 使用同一 GPU 独占锁策略
```

默认每张 GPU 同时只允许一个 GPU 修复片段。CPU 分析可以并行，但不得让多个 ProPainter 实例争用显存。

## 6. 数据持久化

SQLite 使用 WAL 与 `synchronous=FULL`。主要表：

- `jobs`：作业身份、状态、输出、错误；
- `media_probe`：输入媒体结构；
- `shots`：镜头边界；
- `subtitle_tracks`：轨迹和分类；
- `segments`：路由、尝试、检查点；
- `quality_reports`：每次尝试的质量结果；
- `model_runs`：实际引擎、模型身份、耗时和显存；
- `artifacts`：检查点与最终产物哈希；
- `events`：状态审计与取消请求。

检查点必须写临时文件、`fsync`、原子改名，再更新数据库；绝不能先把数据库标为完成再写文件。

## 7. 扩展点

新增检测器：实现现有 detection adapter 的输出契约，输出 `TextDetection` 与可选概率图。

新增空间修复器：继承 `BaseInpainter`，实现 `available()` 与 `inpaint()`；最终必须调用 `composite_exact()`。

新增视频模型：优先采用独立进程适配器，限定 ROI、超时、设备和输出帧数；不要把模型仓库复制进核心包。

新增质量指标：返回归一化指标，并在 `QualityGate` 中明确阈值、严重级别和对应重试动作。

## 8. 安全边界

REST API 仅适合可信局域网。必须设置 `VSRX_ALLOWED_ROOTS`，否则 API 能访问进程用户可读写的路径。不要直接暴露到公网；公网部署需另加认证、限流、作业配额和反向代理。
