# CLI 参考

入口：`vsrx`。开发环境也可运行 `PYTHONPATH=src python -m vsrx.app.cli`。

## `vsrx process`

处理单文件或目录。

```bash
vsrx process INPUT [OPTIONS]
```

关键参数：

| 参数 | 说明 |
|---|---|
| `-o, --output PATH` | 单文件输出；目录输入时为输出目录 |
| `--profile NAME` | `fast`、`balanced`、`quality`、`cpu_economy` |
| `--config PATH` | 自定义基础 YAML |
| `--overlay PATH` | YAML 覆盖，可重复 |
| `--set key=value` | 点路径覆盖，可重复；值按 YAML 解析 |
| `--roi x1,y1,x2,y2` | 固定 ROI，可重复 |
| `--mask-path PATH` | 外部 mask 目录或 NPZ；别名 `--mask-dir` |
| `--work-dir PATH` | 本次作业目录根 |
| `--force-hard-scan` | 存在软字幕也继续检测烧录字幕 |
| `--aggressive` | 删除不确定文字轨迹，增加误删风险 |
| `--overwrite` | 允许覆盖现有输出 |
| `--no-resume` | 禁用检查点复用 |
| `--dry-run` | 只探测、切镜、发现字幕 |
| `--codec NAME` | `auto`、`h264`、`h265`、`av1` |
| `--device N` | CUDA 设备序号 |
| `--keep-intermediates` | 保留成功作业中间产物 |

批量目录保持相对文件名的基本语义，输出命名为 `<stem>.subtitle_removed.<ext>`。批量命令在同一进程复用模型实例，避免每个文件重复加载。

示例：

```bash
vsrx process ./input --output ./output --profile fast

vsrx process input.mkv -o clean.mkv \
  --roi 100,700,1820,1060 \
  --set subtitle_discovery.periodic_full_frame_rescan_seconds=4

vsrx process input.mkv -o clean.mkv \
  --mask-dir ./masks \
  --profile balanced
```

## `vsrx analyze`

只执行媒体分析、镜头和 ROI 发现，不写修复视频。

```bash
vsrx analyze input.mkv --profile balanced
```

## `vsrx jobs`

列出状态库中的最近作业：

```bash
vsrx jobs --limit 200
```

## `vsrx job`

读取单个作业。`--audit` 会附带 probe、QC、模型运行、产物和事件。

```bash
vsrx job JOB_ID
vsrx job JOB_ID --audit
```

## `vsrx cancel`

写入协作式取消请求。当前片段在下一次取消轮询点停止；已完成检查点保留。

```bash
vsrx cancel JOB_ID
```

## `vsrx serve`

启动 FastAPI：

```bash
VSRX_ALLOWED_ROOTS=/data:/archive \
vsrx serve --host 127.0.0.1 --port 8765 --profile balanced
```

## `vsrx doctor`

检查 FFmpeg、FFprobe、OCR runtime、ONNX providers、GPU 与可选后端。缺少可选模型不会失败，缺少 FFmpeg/FFprobe 返回退出码 2。

```bash
vsrx doctor --models-root ./models
```

## 模型命令

```bash
vsrx models status
vsrx models verify
vsrx models install-ocr
vsrx models install-ocr --gpu
vsrx models install-lama --source /trusted/path/lama.onnx
vsrx models install-propainter
vsrx models install-sttn --checkpoint /trusted/path/sttn.pth
```

`install-propainter` 只使用官方仓库提供的下载脚本；若该 revision 没有脚本，命令会停止并要求按官方 README 手工放置权重，不会抓取未知镜像。

## 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 成功或完成但带 `review_required`（审计中明确记录） |
| 1 | 至少一个处理任务失败 |
| 2 | 输入为空、参数错误或缺少必需运行工具 |
