# 故障排查

## `ffmpeg` 或 `ffprobe` 找不到

运行：

```bash
vsrx doctor
which ffmpeg
which ffprobe
```

也可在配置中设置 `probe.ffmpeg_path` 和 `probe.ffprobe_path`。

## OCR 很慢

- 确认 `input_limit_type: max`；
- 设置固定 ROI；
- 使用 `fast`；
- 检查是否错误安装 CPU provider 而期望 CUDA/OpenVINO；
- 降低全画面复扫频率；
- 不要开启文字识别。

## OCR 漏字幕

- 使用 `quality`；
- 增大 `unclip_ratio`；
- 降低 detection threshold；
- 增加全画面复扫；
- 对艺术字片段启用 Medium 重检；
- 使用外部 mask 作为确定性修正。

## 擦除后有白边/黑边

- 增大掩膜膨胀；
- 检查阴影/发光扩展；
- 对半透明底板启用 panel；
- 查看 `audit.json` 中 residual text 与 seam 指标；
- 让当前片段重跑，不需要重跑整条视频。

## 闪烁

- 增加参考帧；
- 提高局部光流质量；
- 降低 TBE-only 接受阈值；
- 避免逐帧全量 LaMa；
- 困难片段升级 ProPainter；
- 检查分块重叠和镜头边界。

## 重影或把人物带进背景

- 提高正反向光流一致性要求；
- 降低动态遮挡参考权重；
- 提高 foreground crossing 触发敏感度；
- 缩短参考窗口；
- 升级视频修复后端。

## 显存不足

- 确认只处理 ROI；
- 降低 ProPainter `subvideo_length` 和 chunk；
- 减少 neighbor length；
- 使用 `--device` 选择空闲 GPU；
- 关闭其他 GPU 作业；
- 配置 STTN 或 LaMa 降级；
- 已完成片段会保留，可直接恢复。

## 输出已有但拒绝覆盖

系统会保护与当前作业身份无关的文件。使用新路径，或确认后加 `--overwrite`。合法的已完成输出会根据 manifest 自动复用。

## API 提交后找不到 submission

API 重启会丢失内存 `submission_id`。使用 `/v1/jobs` 查询持久作业；作业和检查点没有丢失。

## ProPainter 显示 unavailable

检查：

```bash
vsrx models verify
ls "$VSRX_PROPAINTER_REPO/inference_propainter.py"
find "$VSRX_PROPAINTER_REPO/weights" -type f
```

仓库存在但没有权重不会被视为可用。
