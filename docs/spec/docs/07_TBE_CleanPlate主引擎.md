---
doc_id: vsrx-07_TBE_CleanPlate主引擎
version: 1.1
language: zh-CN
format: markdown-only
source_of_truth: VSRX_Hybrid_Architecture_Spec_v1.1_CN.md
---

# 运动补偿 TBE / Clean-Plate 主引擎

> 本文是完整规范的模块化摘录。实现冲突时，以根目录单文件完整规范和版本更高的 ADR 为准。

# 9. 运动补偿 TBE / Clean-Plate 主引擎

## 9.1 设计目标

对目标帧 t，在同一镜头前后搜索字幕未遮挡的参考像素，将其几何和曝光对齐到 t，再进行鲁棒融合。输出：

- `clean_plate`；
- 每像素 coverage；
- 每像素 confidence；
- residual mask；
- 参考帧清单和评分。

## 9.2 两级配准

### 全局配准

顺序：

1. ECC affine；
2. ORB + RANSAC affine；
3. identity。

配准特征排除字幕 mask 与画面边缘低信息区域。最大边降到 960 计算，提高速度。

### ROI 局部光流

全局变换后，只在字幕 ROI + 96 px 上下文计算 OpenCV DIS：

- 默认 `PRESET_FAST`；
- 质量重试切换 `PRESET_MEDIUM`；
- 可选 RAFT-Small 作为 GPU 插件，但不应默认依赖。

使用正反向一致性：

```text
fb_error(x) = ||F(t->j, x) + F(j->t, x + F(t->j,x))||
```

`fb_error > 1.8 px`、越界、参考帧被 mask 或发生遮挡的像素均判为无效。

## 9.3 参考帧选择

默认前后各 0.9 秒，最少 0.4 秒，最大 1.5 秒；每侧最多 36 帧。评分：

```text
S_j(x) = 0.30*flow_consistency
       + 0.25*photometric_ring_similarity
       + 0.15*gradient_ring_similarity
       + 0.15*exposure_fit_quality
       + 0.15*temporal_proximity
```

参考帧要求：

- 与目标帧同镜头；
- 目标像素在参考帧未被字幕 mask；
- 光流正反向一致；
- mask 外环的亮度、梯度与目标帧一致；
- 不被新前景遮挡。

## 9.4 曝光与色彩补偿

在 mask 外 8-24 px ring 上拟合每通道仿射：

```text
I_target ~= a * I_warped_reference + b
```

使用 Huber/截断最小二乘，避免人物或高光干扰。只在拟合残差低于阈值时应用。

## 9.5 鲁棒融合

对每个待恢复像素 x，收集有效候选 `v_j(x)` 和权重 `w_j(x)`。默认使用 weighted median，因为它能抵抗少量错配、运动物体和曝光异常。至少 3 个有效参考；5 个以上更可靠。

```text
coverage(x) = min(valid_count(x) / preferred_count, 1.0)
confidence(x) = weighted_mean(S_j(x)) * occlusion_validity(x)
```

残洞定义：

```text
residual(x) = original_mask(x)
              AND (coverage(x) < c_min OR confidence(x) < q_min)
```

## 9.6 为什么这个引擎最符合需求

- 字幕变化会周期性暴露不同的真实背景；
- 绝大多数计算在小 ROI 中进行；
- 不加载大视频生成模型；
- 真实像素比生成内容更稳定、可复现；
- 对皮肤、衣服、头发、地面等细节不会凭空重绘；
- 即使失败，也能通过 coverage/confidence 明确知道失败位置。

---
