# 批处理、运维与资源控制

## 1. 推荐目录

```text
/data/input       只读输入
/data/output      最终输出
/data/work        SQLite、检查点、审计与临时文件
/data/models      只读模型与上游仓库
```

工作目录需要足够空间。FFV1 检查点可能接近或高于原始解码数据的压缩体积。

## 2. 作业恢复

默认启用恢复。作业身份由输入、配置、模型和外部 mask 决定。再次处理相同身份时：

- 验证最终输出 manifest，合法则直接复用；
- 否则验证已完成片段检查点；
- 只重新执行丢失、哈希不匹配或未完成片段；
- 可改变输出路径而复用同一组片段。

不要手工修改 `work/<job_id>/segments`。要强制重跑可使用 `--no-resume` 或改变配置；覆盖已有最终文件需 `--overwrite`。

## 3. 并发

单 GPU：

- GPU 重模型 worker：1；
- OCR 可与 CPU 分析并行；
- 编码可以与下一个文件分析重叠，但基础 CLI 为稳定性采用顺序文件循环；
- 不建议同时启动多个独立 `vsrx process` 指向同一 GPU，文件锁会串行，但会增加内存和磁盘压力。

多 GPU：

- 每个进程固定 `--device N`；
- 最好使用独立 `work_dir` 或确保共享 SQLite 位于可靠本地文件系统；
- 不建议把 SQLite WAL 放到不支持可靠锁语义的网络文件系统。

## 4. 磁盘与清理

成功作业默认可清理临时帧，只保留数据库、审计和必要检查点策略。`--keep-intermediates` 用于诊断，不适合长期批量默认开启。

定期监控：

```bash
du -sh work output models
find work -type f -name '*.tmp' -mtime +1 -delete
```

不要在作业运行时删除 lock 或当前 segment 目录。

## 5. 日志与审计

每个作业输出：

- `audit.json`：机器读取；
- `audit.md`：人工/AI 阅读；
- SQLite 事件与片段状态；
- 最终产物 SHA-256。

重点指标：

- `review_required_count`；
- `heavy_model_segment_rate`；
- route/engine histogram；
- QC 失败原因；
- 每阶段耗时；
- 峰值 RSS/VRAM。

## 6. 资源档位

### CPU 经济

```bash
vsrx process input.mkv -o output.mkv --profile cpu_economy
```

使用较少参考帧、Tiny OCR 和轻量后端。复杂运动质量低于 GPU 配置。

### 默认平衡

```bash
vsrx process input.mkv -o output.mkv --profile balanced
```

推荐绝大多数批处理。

### 快速

适合普通固定对白字幕；通过 Golden Set 证明误漏率可接受后再批量采用。

### 质量

增加检测重试、参考帧和 QC 严格度。只应用于精品输出或自动筛出的困难片段。

## 7. 容器

CPU Docker 镜像可直接运行 OCR/TBE/Telea。GPU 模型的 CUDA、PyTorch 和上游依赖差异较大，推荐为 ProPainter 单独构建固定 revision 的镜像，并把其仓库/权重只读挂载到主服务。

## 8. systemd

提供 `services/vsrx-api.service` 示例。部署前修改用户、工作目录、虚拟环境和允许根目录。服务不应以 root 运行。

## 9. 备份

需要备份：

- SQLite 主文件及 `-wal`/`-shm`（最好先停止服务或执行 checkpoint）；
- 配置与模型 manifest；
- 审计报告；
- 最终输出。

检查点可视磁盘成本选择性备份。
