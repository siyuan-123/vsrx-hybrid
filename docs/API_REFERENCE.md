# REST API 参考

API 由 `vsrx serve` 启动。它适合可信本地网络，不包含认证、计费和租户隔离。

## 路径安全

生产环境必须设置：

```bash
export VSRX_ALLOWED_ROOTS=/data/input:/data/output:/data/masks
```

请求中的输入、输出和 mask 路径都必须位于允许根目录内。未设置时不做根目录限制，因此不得直接公网暴露。

## `GET /health`

返回服务版本和当前可用修复后端。

## `GET /v1/config`

返回解析后的配置、档位和配置哈希。注意其中可能包含本地路径；不要向不可信客户端开放。

## `POST /v1/jobs`

请求：

```json
{
  "input_path": "/data/input/movie.mkv",
  "output_path": "/data/output/movie.mkv",
  "fixed_rois": [[80, 700, 1840, 1060]],
  "external_mask_path": null,
  "force_hard_subtitle_scan": false,
  "overwrite": false,
  "aggressive_uncertain_removal": false,
  "codec": "auto",
  "device_index": 0
}
```

响应为 HTTP 202：

```json
{
  "submission_id": "...",
  "state": "queued",
  "job_id": null,
  "output_path": null,
  "error": null
}
```

`submission_id` 是当前 API 进程内的提交标识；持久作业建立后会出现 `job_id`。

## `GET /v1/submissions/{submission_id}`

查询内存提交状态。API 重启后该映射会丢失，但 SQLite 中的 `job_id`、检查点和作业状态仍保留。

## `GET /v1/jobs?limit=100`

列出持久作业。

## `GET /v1/jobs/{job_id}`

返回作业基本字段和片段列表。

## `GET /v1/jobs/{job_id}/audit?event_limit=1000`

返回完整机器可读审计：

- `job`；
- `probe`；
- `segments`；
- `quality_reports`；
- `model_runs`；
- `artifacts`；
- `events`。

## `POST /v1/jobs/{job_id}/cancel`

写入取消请求，返回 202。

## 并发模型

API 使用有界线程池提交作业，线程数来自 `scheduler.cpu_analysis_workers`。GPU 修复仍通过跨进程文件锁串行。对于多 GPU，应启动多个 API/worker 实例并为每个实例固定默认设备和独立工作目录，或在上层队列中显式分配 `device_index`。

## 反向代理建议

- 请求体限制为很小，因为 API 只接收路径，不上传视频；
- 超时只影响提交接口，不应等待整个视频完成；
- 禁止目录穿越并使用容器只读挂载输入；
- 给输出与 work 目录独立磁盘配额；
- 用 systemd 或容器重启策略恢复服务。
