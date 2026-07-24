# VSR-X Hybrid 1.1：纯 Markdown AI 开发包

本目录只包含 Markdown 文件。没有图片、PDF、DOCX、Graphviz、二进制附件或外部图像依赖。

## 推荐读取方式

1. AI 一次性读取：先加载 [`VSRX_Hybrid_Architecture_Spec_v1.1_CN.md`](VSRX_Hybrid_Architecture_Spec_v1.1_CN.md)。
2. AI 分阶段开发：先加载 [`00_AI_INDEX.md`](00_AI_INDEX.md)，再按当前模块加载 `docs/` 中的一到三个文件。
3. 编写代码前：额外加载 `reference/` 中对应的配置、接口或数据库文档。

## 模块化文档

- [`01_最终决策与需求约束.md`](docs/01_最终决策与需求约束.md)：最终决策与需求约束
- [`02_开源方案调研与对比.md`](docs/02_开源方案调研与对比.md)：开源方案调研与对比
- [`03_总体技术架构.md`](docs/03_总体技术架构.md)：总体技术架构
- [`04_仓库与模块设计.md`](docs/04_仓库与模块设计.md)：仓库与模块设计
- [`05_媒体探测与镜头切分.md`](docs/05_媒体探测与镜头切分.md)：媒体探测与镜头切分
- [`06_字幕检测跟踪与精细掩膜.md`](docs/06_字幕检测跟踪与精细掩膜.md)：字幕检测、跟踪与精细掩膜
- [`07_TBE_CleanPlate主引擎.md`](docs/07_TBE_CleanPlate主引擎.md)：运动补偿 TBE / Clean-Plate 主引擎
- [`08_片段路由与修复模型.md`](docs/08_片段路由与修复模型.md)：片段路由与修复模型
- [`09_自动质量门与重试闭环.md`](docs/09_自动质量门与重试闭环.md)：自动质量门与重试闭环
- [`10_运行时部署与资源控制.md`](docs/10_运行时部署与资源控制.md)：运行时部署与资源控制
- [`11_作业状态数据契约与API.md`](docs/11_作业状态数据契约与API.md)：作业状态、数据契约与 API
- [`12_编码色彩与配置档位.md`](docs/12_编码色彩与配置档位.md)：编码、色彩与配置档位
- [`13_测试基准与发布门槛.md`](docs/13_测试基准与发布门槛.md)：测试基准与发布门槛
- [`14_开发路线图.md`](docs/14_开发路线图.md)：开发路线图
- [`15_失败模式实施清单与结论.md`](docs/15_失败模式实施清单与结论.md)：失败模式、实施清单与研究结论
- [`16_来源许可证与版本说明.md`](docs/16_来源许可证与版本说明.md)：来源、许可证与版本说明

## 参考源码文档

- [`reference/01_CONFIG_REFERENCE.md`](reference/01_CONFIG_REFERENCE.md)：YAML 配置源码。
- [`reference/02_INTERFACES_REFERENCE.md`](reference/02_INTERFACES_REFERENCE.md)：Python 接口源码。
- [`reference/03_DATABASE_SCHEMA_REFERENCE.md`](reference/03_DATABASE_SCHEMA_REFERENCE.md)：SQLite SQL 源码。
- [`reference/04_MODEL_MANIFEST_REFERENCE.md`](reference/04_MODEL_MANIFEST_REFERENCE.md)：模型清单模板。

## 文档约束

- 架构图统一使用 Mermaid 文本，并附纯文字说明。
- 单文件规范是总体事实源；模块文档便于上下文按需加载。
- Markdown 中的 YAML/Python/SQL 代码块是可提取的实现基线。
- 任何模型、阈值或依赖升级都要在同一 Golden Set 上重跑基准。
- 所有恢复失败必须进入自动重试或 `REVIEW_REQUIRED`，禁止静默通过。

## 版本变化

1.1 相对 1.0：删除 PDF、DOCX、PNG、DOT；将图表转为 Mermaid；将配置、接口、数据库和 manifest 转为 Markdown 文档；增加 AI 索引与模块化拆分。
