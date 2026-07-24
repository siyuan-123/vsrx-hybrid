# 外部逐帧掩膜

外部掩膜是最确定、最高效的输入方式：跳过 OCR、轨迹分类和自动字幕发现，只执行运动恢复、路由、修复和质量门。

## 1. 目录格式

支持灰度/彩色图或 `.npy`：

```text
masks/
├── 00000000.png
├── 00000001.png
├── frame_00000002.png
├── mask_00000003.npy
└── pts_266667.png
```

索引命名按解码顺序；`pts_` 后为微秒。VFR 视频优先使用 PTS。

掩膜语义：

- 0：保留；
- 非 0：删除并恢复；
- 浮点数组范围 0～1；
- 8 位图建议 0/255。

JPEG 会产生非零噪声，不推荐；加载器会应用阈值，但 PNG/NPY 更稳定。

## 2. `masks.json`

可显式映射：

```json
{
  "schema_version": 1,
  "time_base": "microseconds",
  "frames": [
    {"frame_index": 0, "pts_us": 0, "path": "00000000.png"},
    {"frame_index": 1, "pts_us": 66667, "path": "00000001.png"}
  ]
}
```

相对路径以 manifest 目录为基准。

## 3. NPZ

单个 `.npz` 可包含：

- `masks`：`[T,H,W]`；
- 可选 `frame_indices`；
- 可选 `pts_us`。

大视频不建议把全部 mask 压进单个 NPZ，因为加载峰值内存较高。

## 4. 尺寸与缺失帧

- mask 尺寸不同于视频时，使用最近邻缩放；
- 缺失帧视为空 mask；
- 多个候选同时命中时，优先显式 manifest/PTS，再使用帧索引；
- 外部 mask 文件内容参与作业身份哈希；修改任何 mask 会生成新作业。

## 5. 固定矩形掩膜

可用 `tools/export_masks_from_roi.py` 生成：

```bash
python tools/export_masks_from_roi.py input.mkv ./masks \
  --roi 100,720,1820,1060
```

矩形 mask 会删除整个区域，适合字幕始终固定且背景可从时序恢复的素材；若区域过大，会增加生成量和误删风险。

## 6. 调试建议

先用 5～10 秒片段检查：

- 字幕描边是否完全包含；
- 是否误覆盖人物或 UI；
- 出现/消失边界帧是否同步；
- VFR 是否使用正确 PTS；
- 审计中的 `mask_ratio`、`coverage` 与路由是否合理。
