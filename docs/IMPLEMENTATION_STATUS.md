# 实现状态与边界

## 已完成

- 可安装 Python `src/` 包、CLI 和 FastAPI；
- FFprobe 全流探测、PyAV PTS/VFR 解码；
- 软字幕移除/提取快速路径；
- PySceneDetect 镜头切分；
- PP-OCRv6 detection-only RapidOCR 适配；
- 无 OCR 时启发式降级；
- 自动多 ROI、固定 ROI、外部 mask；
- Hungarian + Kalman 跟踪和轨迹保护；
- 描边、阴影、发光、底板、时间并集 mask；
- 相邻运动缓存、phase/ECC/ORB、DIS 与置信度；
- 曝光补偿、参考帧评分、加权中位数 clean plate；
- 片段路由和 VRAM 预算；
- Telea、NS、LaMa ONNX、MI-GAN ONNX；
- 官方 ProPainter/STTN 子进程适配；
- QC 和局部重试；
- SQLite WAL、FFV1 检查点、取消、恢复；
- 最终容器兼容编码与音频 fallback；
- JSON/Markdown 审计和产物哈希；
- 自动测试、合成数据和部署文件。

## 需要用户提供的外部资产

- OCR runtime/模型缓存；
- LaMa 或 MI-GAN ONNX 权重；
- 官方 ProPainter 仓库及权重；
- 官方 STTN checkpoint。

这些资产受体积、环境和许可证约束，不能合理地内置到代码包。

## 本交付环境中已验证

- 纯 CPU 基础路径；
- 外部 mask 完整视频处理；
- FFV1 分段检查点与恢复；
- 更换输出路径复用检查点；
- H.264 MKV/MP4 最终编码；
- 输入 60 帧，输出保持 60 帧和时长；
- 合成字幕区 OCR 复检从有检测降为 0；
- 单元与集成测试；
- Ruff 静态检查；
- 包构建与安装验证将在 `VALIDATION_REPORT.md` 记录。

## 尚未在本交付环境实测

- 官方 ProPainter 权重推理；
- 官方 STTN 权重推理；
- 任意第三方 LaMa/MI-GAN ONNX 导出图；
- NVIDIA/Intel/Apple 硬件编码；
- 真实 HDR10/Dolby Vision 全链路；
- 多小时 4K 素材；
- 分布式多机队列。

适配器和错误边界已实现，但这些能力必须在目标硬件和固定权重上做验收，不能用“代码已接入”代替实际模型验证。

## 物理限制

若被字幕遮挡的背景在整个镜头中从未出现，算法只能生成合理内容，无法知道真实原像素。系统通过 QC 与 `review_required` 显式暴露这种困难，而不是承诺不可能的百分之百还原。
