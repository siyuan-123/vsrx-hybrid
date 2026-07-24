# Changelog

## 1.0.0 — 2026-07-20

首个完整实现版本：

- 完成可恢复的端到端硬字幕去除流水线。
- 完成软字幕快速移除、VFR/PTS 感知媒体处理。
- 完成 PP-OCRv6 detection-only、启发式降级、多 ROI、轨迹和精细掩膜。
- 完成运动补偿 TBE、跨帧变换组合、局部光流自动门控和鲁棒融合。
- 完成 Telea、LaMa、MI-GAN、官方 ProPainter、STTN 适配器与自动路由。
- 完成显存预测、跨进程 GPU 独占、自动降级。
- 完成质量门、局部重试、FFV1 检查点、SQLite WAL 和输出清单。
- 完成 CLI、REST API、模型管理、Docker、systemd、测试和纯 Markdown 文档。
- 增加外部逐帧掩膜模式。
- 输出缓存包含输入、配置、处理选项、掩膜和实际模型运行时指纹。
- 通过 Ruff 静态检查与完整自动测试套件。
