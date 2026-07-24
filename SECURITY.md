# 安全说明

- REST API 默认只监听 `127.0.0.1`。
- 部署服务时必须设置 `VSRX_ALLOWED_ROOTS`，限制输入、输出和外部掩膜路径。
- 不要从不可信地址安装 ONNX、PyTorch checkpoint 或第三方仓库。
- 模型安装后使用 `vsrx models verify`，并在 Manifest 中固定 SHA-256。
- ProPainter/STTN 通过外部 Python 进程运行；建议在单独容器或低权限账户中执行。
- 处理目录可能包含可恢复的视频帧，应设置合适的文件权限，并配置成功任务清理策略。
- FFmpeg、OpenCV、PyAV、ONNX Runtime 和 PyTorch 应保持安全更新。

安全问题请在不公开真实素材的前提下报告，并附最小复现步骤、版本、平台和日志中的错误代码。
