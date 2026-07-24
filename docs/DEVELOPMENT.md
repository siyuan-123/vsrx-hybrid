# 开发指南

## 1. 环境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,ocr]'
pre-commit  # 本项目未强制依赖，可自行接入
```

## 2. 常用命令

```bash
make test
make lint
make format
make package

python -m compileall -q src
python tools/validate_installation.py
```

## 3. 开发顺序

修改算法时遵守：

1. 先把失败样本加入 Golden Set 或测试 fixture；
2. 写失败测试；
3. 修改单一模块，不在 pipeline 中塞算法；
4. 验证掩膜外像素不变量；
5. 运行单元、集成、格式和包安装测试；
6. 更新配置、文档和 CHANGELOG；
7. 用真实短片比较性能与 QC，不能只看单帧视觉效果。

## 4. 领域对象

跨模块传递 `src/vsrx/domain/contracts.py` 中的对象，不用匿名 dict 作为算法接口。新增字段要保持 JSON 可序列化，并考虑 SQLite/审计兼容性。

## 5. 错误分类

- 配置/输入错误：立即失败，不自动重试；
- 模型不可用：按 registry fallback；
- OOM：降低 chunk/后端，只重试片段；
- 外部工具退出：保留 stderr 摘要和临时产物策略；
- QC 不通过：走 retry ladder；
- 无法恢复：`review_required`，不伪装成功。

## 6. 新增模型适配器检查表

- `available()` 不应触发大模型加载；
- 模型惰性加载；
- 明确设备选择；
- 输入 ROI 和帧数上限；
- 超时；
- OOM 识别；
- 输出帧数/尺寸校验；
- `composite_exact()`；
- 模型 hash/commit；
- 临时文件清理；
- 单元测试和至少一个真实集成测试。

## 7. 版本与兼容性

配置 `schema_version` 当前为 1。破坏性配置或数据库变化必须：

- 提升 schema 版本；
- 提供迁移脚本；
- 防止旧缓存被误复用；
- 更新 `docs/spec` 与实现文档。
