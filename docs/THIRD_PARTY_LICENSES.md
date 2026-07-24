# 第三方组件与许可证注意事项

本项目核心代码使用 Apache-2.0。第三方库、模型代码和权重各自受其许可证约束；外层许可证不会覆盖模型许可证。

| 组件 | 用途 | 许可证/要求 |
|---|---|---|
| FFmpeg | 探测、转码、复用 | 取决于发行构建及启用 codec；检查本机构建 |
| PyAV | 解码/编码接口 | BSD 类；同时受 FFmpeg 构建影响 |
| OpenCV | 图像、光流、Telea | Apache-2.0 |
| NumPy/SciPy | 数值计算 | BSD 类 |
| PySceneDetect | 镜头检测 | BSD 类 |
| FastAPI/Uvicorn/Typer/Rich | 服务和 CLI | 各自宽松许可证 |
| RapidOCR / PP-OCR 权重 | detection-only | 对安装版本和具体权重逐项核查 |
| LaMa | residual inpainting | 上游代码 Apache-2.0；具体导出权重需核查来源 |
| MI-GAN | 可选经济后端 | 核查仓库与权重许可证 |
| ProPainter | 困难视频修复 | 官方代码/模型面向非商业研究；商业需另行授权 |
| STTN | 低显存降级 | 上游仓库 MIT；checkpoint 来源仍需记录 |

## 必做登记

每个部署环境保存：

```text
模型名称
来源 URL
Git commit 或 release
权重文件名与字节数
SHA-256
下载日期
许可证文件快照
用途
是否允许当前使用方式
```

`configs/model_manifest.yaml` 是技术身份清单，不替代法律审查。

## 不包含内容

发布包不包含任何第三方模型权重或上游仓库，因此下载包本身不会把 ProPainter、STTN 或其他权重重新分发给用户。
