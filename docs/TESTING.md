# 测试、基准与验收

## 1. 自动测试

```bash
python -m pip install -e '.[test,dev]'
ruff check src tests tools
ruff format --check src tests tools
pytest -q
```

当前测试覆盖：

- 配置哈希和覆盖；
- 外部 mask 目录与身份；
- 全局配准和 clean plate；
- 路由阈值；
- 精确合成的掩膜外不变量；
- 轨迹插值；
- 质量门；
- SQLite 幂等和检查点；
- API；
- 小视频首次处理和改变输出路径后的检查点复用。

## 2. 合成端到端测试

```bash
python tools/generate_synthetic_video.py --output-dir /tmp/vsrx-smoke
vsrx process /tmp/vsrx-smoke/burned.mkv \
  -o /tmp/vsrx-smoke/output.mkv \
  --mask-dir /tmp/vsrx-smoke/masks \
  --profile fast
```

生成器同时保存 clean ground truth，因此可以计算字幕区误差和掩膜外像素变化。

## 3. Golden Set

正式批量前，从真实素材抽取至少 200～500 个 5～15 秒片段，覆盖：

- 单行、双行、双语；
- 中文、英文、日文、艺术字；
- 动画、真人、PPT、游戏；
- 顶部、竖排、移动、卡拉 OK；
- 描边、阴影、发光、半透明底板；
- 人物穿越字幕；
- 高运动、低照度、严重压缩；
- VFR、HDR、隔行和多音轨；
- 字幕始终遮挡和满屏文字。

每类至少保留：输入、人工 mask、可用时的无字幕原版、预期处理策略和人工评分。

## 4. 指标

系统指标：

- 作业完成率；
- 崩溃后恢复率；
- OOM 是否只影响片段；
- 每分钟视频处理耗时；
- 峰值 RSS/VRAM；
- 重模型帧/片段占比；
- 人工复核率。

质量指标：

- OCR 残留率；
- 描边/阴影残留率；
- 光流对齐时间误差；
- 接缝 Delta/梯度比；
- 掩膜外像素差；
- 有 ground truth 时的 PSNR/SSIM/LPIPS；
- 盲测主观偏好。

## 5. 发布门槛

最低要求：

- 静默失败为 0；
- 所有掩膜外保护测试通过；
- 检查点恢复集成测试通过；
- FFmpeg 输出帧数/时长/音频流验证通过；
- 8 GB 配置不会因单片段 OOM 丢失整条作业；
- Golden Set 的人工复核率达到项目目标；
- 所有实际使用模型哈希已固定。

## 6. 性能基准

```bash
python tools/benchmark.py input.mkv \
  --mask-path ./masks \
  --profile balanced \
  --output benchmark.json
```

基准必须记录：硬件、操作系统、Python、FFmpeg、OpenCV、运行时 provider、模型哈希、输入分辨率/帧数、路由占比和输出哈希。不要把低分辨率合成测试速度外推为 1080p 真实素材速度。
