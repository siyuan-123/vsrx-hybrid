# 贡献指南

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[ocr,dev]'
```

## 必须通过的检查

```bash
ruff format src tests tools
ruff check src tests tools
pytest -q
python -m build
```

## 模块边界

- `domain/` 只放稳定契约，不依赖具体模型。
- `media/` 不应依赖 OCR 或修复模型。
- `detection/` 输出统一 `TextDetection`。
- `mask/` 输出统一 `MaskFrame`，不得直接编码视频。
- `cleanplate/` 只恢复真实背景和 residual mask。
- `inpaint/` 必须保证硬掩膜外像素精确保留。
- `quality/` 不负责修改帧，只给出报告和重试建议。
- `scheduler/` 不包含视觉算法。
- 新模型必须通过 `BaseInpainter` 接口接入，并提供缺权重时的明确错误。

## 测试要求

修复 bug 时必须增加能在无神经网络权重环境中运行的回归测试。涉及外部模型的测试应使用环境变量显式启用，不能成为默认 CI 的硬依赖。

## 数据与隐私

不要提交真实用户视频、帧、字幕或模型权重。测试素材应由代码生成，或明确具备可重新分发许可。
