---
doc_id: vsrx-reference-04_MODEL_MANIFEST_REFERENCE
version: 1.1
language: zh-CN
format: markdown-only
---

# 第三方模型 Manifest 模板（YAML 源码）

> 单文件模型填写 `bytes` 与 `sha256`；仓库型模型同时固定 `git_commit`，并在 `files`
> 中逐个登记运行所需权重。未知或不匹配的哈希不得进入稳定任务。

```yaml
schema_version: 1
research_snapshot: YYYY-MM-DD
models:
  ppocr_v6_small_det:
    role: default_text_detector
    source: https://可信下载地址/PP-OCRv6_det_small.onnx
    runtime: rapidocr_3.9.2_onnxruntime
    license: verify_with_downloaded_weight
    filename: PP-OCRv6_det_small.onnx
    bytes: REPLACE_WITH_FILE_SIZE
    sha256: REPLACE_WITH_SHA256
  lama:
    role: residual_hole_inpainting
    source: https://可信下载地址/lama_fp32.onnx
    upstream_source: https://github.com/advimman/lama
    runtime: onnxruntime
    license: Apache-2.0
    bytes: REPLACE_WITH_FILE_SIZE
    sha256: REPLACE_WITH_SHA256
  propainter_official:
    role: hard_roi_video_inpainting
    source: https://github.com/sczhou/ProPainter
    git_commit: REPLACE_WITH_GIT_COMMIT
    runtime: isolated_pytorch_cuda_worker
    license: noncommercial_research_only_unless_separately_authorized
    files:
      weights/ProPainter.pth:
        bytes: REPLACE_WITH_FILE_SIZE
        sha256: REPLACE_WITH_SHA256
      weights/recurrent_flow_completion.pth:
        bytes: REPLACE_WITH_FILE_SIZE
        sha256: REPLACE_WITH_SHA256
      weights/raft-things.pth:
        bytes: REPLACE_WITH_FILE_SIZE
        sha256: REPLACE_WITH_SHA256
  sttn:
    role: emergency_low_vram_fallback
    source: https://可信下载地址/sttn.pth
    upstream_source: https://github.com/researchmm/STTN
    git_commit: REPLACE_WITH_GIT_COMMIT
    runtime: isolated_pytorch_worker
    license: MIT
    bytes: REPLACE_WITH_FILE_SIZE
    sha256: REPLACE_WITH_SHA256
```

RapidOCR 的 `filename` 相对于其安装包的 `models/` 目录解析；LaMa、MI-GAN 和 STTN
相对于 `--models-root` 的约定路径解析。ProPainter 的 `files` 键相对于仓库根目录解析。

