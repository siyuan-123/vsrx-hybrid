from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from vsrx.app.api import create_app
from vsrx.app.config_loader import load_runtime_config
from vsrx.app.model_manager import ModelManager
from vsrx.app.options import ProcessOptions
from vsrx.app.pipeline import VSRXPipeline
from vsrx.app.resources import model_manifest_path
from vsrx.scheduler import JobRepository
from vsrx.utils.logging import configure_logging

app = typer.Typer(
    no_args_is_help=True, help="VSR-X Hybrid：高效率、低资源、可恢复的批量视频硬字幕去除。"
)
models_app = typer.Typer(no_args_is_help=True, help="模型与运行环境管理。")
app.add_typer(models_app, name="models")
console = Console()


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        parts = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise typer.BadParameter("ROI 必须是 x1,y1,x2,y2") from exc
    if len(parts) != 4 or parts[2] <= parts[0] or parts[3] <= parts[1]:
        raise typer.BadParameter("ROI 必须是有效的 x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def _config(
    config: Path | None,
    profile: str,
    overlay: list[Path],
    set_values: list[str],
):
    cfg = load_runtime_config(config, profile=profile, overlay_paths=overlay, overrides=set_values)
    configure_logging(
        str(cfg.get("runtime.log_level", "INFO")), bool(cfg.get("observability.json_logs", True))
    )
    return cfg


def _inputs(path: Path, cfg) -> list[Path]:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise typer.BadParameter(f"输入不存在：{resolved}")
    extensions = {"." + str(item).lower().lstrip(".") for item in cfg.get("input.extensions", [])}
    iterator = resolved.rglob("*") if bool(cfg.get("input.recursive", True)) else resolved.glob("*")
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in extensions)


@app.command("process")
def process_command(
    input_path: Annotated[Path, typer.Argument(help="视频文件或目录")],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="单文件输出路径；批量时为输出目录")
    ] = None,
    config: Annotated[Path | None, typer.Option("--config", help="基础 YAML 配置")] = None,
    profile: Annotated[
        str, typer.Option("--profile", help="fast/balanced/quality/cpu_economy")
    ] = "balanced",
    overlay: Annotated[list[Path], typer.Option("--overlay", help="附加 YAML 覆盖，可重复")] = [],
    set_value: Annotated[list[str], typer.Option("--set", help="key=value 配置覆盖，可重复")] = [],
    roi: Annotated[list[str], typer.Option("--roi", help="固定区域 x1,y1,x2,y2，可重复")] = [],
    mask_path: Annotated[
        Path | None, typer.Option("--mask-path", "--mask-dir", help="外部逐帧掩膜目录或 VSR-X NPZ")
    ] = None,
    work_dir: Annotated[Path | None, typer.Option("--work-dir")] = None,
    force_hard_scan: Annotated[
        bool, typer.Option("--force-hard-scan", help="即使存在软字幕也继续扫描硬字幕")
    ] = False,
    aggressive: Annotated[
        bool, typer.Option("--aggressive", help="自动删除不确定文字轨迹；可能增加误删风险")
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    no_resume: Annotated[bool, typer.Option("--no-resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="仅探测、切镜和发现字幕区域")] = False,
    codec: Annotated[str, typer.Option("--codec", help="auto/h264/h265/av1")] = "auto",
    device: Annotated[int, typer.Option("--device", help="CUDA 设备序号")] = 0,
    keep_intermediates: Annotated[bool, typer.Option("--keep-intermediates")] = False,
) -> None:
    cfg = _config(config, profile, overlay, set_value)
    files = _inputs(input_path, cfg)
    if not files:
        console.print("[red]没有找到可处理的视频。[/red]")
        raise typer.Exit(2)
    fixed_rois = [_parse_roi(item) for item in roi]
    batch = len(files) > 1 or input_path.expanduser().resolve().is_dir()
    if batch and output is not None:
        output.expanduser().mkdir(parents=True, exist_ok=True)
    failures = 0
    table = Table("输入", "状态", "输出", "审计")
    pipeline = VSRXPipeline(cfg, model_manifest_path=model_manifest_path())
    for source in files:
        if batch:
            out_dir = (
                output.expanduser().resolve()
                if output
                else Path(str(cfg.get("runtime.output_dir", "./output"))).expanduser().resolve()
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            extension = (
                source.suffix
                if source.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}
                else ".mkv"
            )
            output_path = out_dir / f"{source.stem}.subtitle_removed{extension}"
        else:
            output_path = output
        try:
            result = pipeline.process(
                source,
                ProcessOptions(
                    output_path=output_path,
                    work_dir=work_dir,
                    fixed_rois=fixed_rois,
                    external_mask_path=mask_path,
                    force_hard_subtitle_scan=force_hard_scan,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    resume=not no_resume,
                    codec=codec,
                    device_index=device,
                    keep_intermediates=keep_intermediates,
                    aggressive_uncertain_removal=aggressive,
                ),
            )
            state_style = "yellow" if result.review_required else "green"
            table.add_row(
                str(source),
                f"[{state_style}]{result.state}[/{state_style}]",
                str(result.output_path or "-"),
                str(result.audit_markdown),
            )
        except Exception as exc:
            failures += 1
            table.add_row(str(source), "[red]failed[/red]", "-", f"{type(exc).__name__}: {exc}")
    console.print(table)
    if failures:
        raise typer.Exit(1)


@app.command("analyze")
def analyze_command(
    input_path: Path,
    output: Path | None = typer.Option(None, "--output", "-o"),
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
    roi: list[str] = typer.Option([], "--roi"),
) -> None:
    cfg = _config(config, profile, [], [])
    result = VSRXPipeline(cfg, model_manifest_path=model_manifest_path()).process(
        input_path,
        ProcessOptions(
            output_path=output, fixed_rois=[_parse_roi(item) for item in roi], dry_run=True
        ),
    )
    console.print(f"分析完成：{result.audit_markdown}")


@app.command("jobs")
def jobs_command(
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
    limit: int = typer.Option(100, "--limit"),
) -> None:
    cfg = _config(config, profile, [], [])
    work = Path(str(cfg.get("runtime.work_dir", "./work"))).expanduser().resolve()
    db_cfg = Path(str(cfg.get("runtime.state_db", "vsrx.sqlite3")))
    repo = JobRepository(db_cfg if db_cfg.is_absolute() else work / db_cfg.name)
    try:
        table = Table("Job ID", "状态", "输入", "输出", "更新时间")
        for job in repo.list_jobs(limit=limit):
            table.add_row(
                job["job_id"],
                job["state"],
                job["input_path"],
                job.get("output_path") or "",
                job["updated_at"],
            )
        console.print(table)
    finally:
        repo.close()


@app.command("job")
def job_command(
    job_id: str,
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
    audit: bool = typer.Option(False, "--audit", help="包含探测、QC、模型运行、产物和事件"),
) -> None:
    cfg = _config(config, profile, [], [])
    work = Path(str(cfg.get("runtime.work_dir", "./work"))).expanduser().resolve()
    db_cfg = Path(str(cfg.get("runtime.state_db", "vsrx.sqlite3")))
    repo = JobRepository(db_cfg if db_cfg.is_absolute() else work / db_cfg.name)
    try:
        job = repo.get_job(job_id)
        if job is None:
            raise typer.BadParameter("job_id 不存在")
        payload: dict[str, object] = {"job": job, "segments": repo.list_segments(job_id)}
        if audit:
            payload.update(
                probe=repo.get_probe(job_id),
                quality_reports=repo.list_quality_reports(job_id),
                model_runs=repo.list_model_runs(job_id),
                artifacts=repo.list_artifacts(job_id),
                events=repo.list_events(job_id),
            )
        console.print_json(json.dumps(payload, ensure_ascii=False))
    finally:
        repo.close()


@app.command("cancel")
def cancel_command(
    job_id: str,
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
) -> None:
    cfg = _config(config, profile, [], [])
    work = Path(str(cfg.get("runtime.work_dir", "./work"))).expanduser().resolve()
    db_cfg = Path(str(cfg.get("runtime.state_db", "vsrx.sqlite3")))
    repo = JobRepository(db_cfg if db_cfg.is_absolute() else work / db_cfg.name)
    try:
        if repo.get_job(job_id) is None:
            raise typer.BadParameter("job_id 不存在")
        repo.request_cancel(job_id)
        console.print("已写入取消请求。")
    finally:
        repo.close()


@app.command("serve")
def serve_command(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
) -> None:
    import uvicorn

    cfg = _config(config, profile, [], [])
    uvicorn.run(create_app(cfg), host=host, port=port)


@app.command("doctor")
def doctor_command(
    root: Path = typer.Option(Path("./models"), "--models-root"),
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
) -> None:
    """检查基础运行环境和可选模型；缺少可选模型不会让命令失败。"""

    cfg = _config(config, profile, [], [])
    manager = ModelManager(cfg, root, model_manifest_path())
    status = manager.status()
    verification = manager.verify()
    payload = {"config_hash": cfg.hash, "status": status, "verification": verification}
    console.print_json(json.dumps(payload, ensure_ascii=False))
    mandatory_missing = [name for name in ("ffmpeg", "ffprobe") if not status.get(name)]
    if mandatory_missing:
        console.print(f"[red]缺少必需工具：{', '.join(mandatory_missing)}[/red]")
        raise typer.Exit(2)


@models_app.command("status")
def models_status(
    root: Path = typer.Option(Path("./models"), "--root"),
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
) -> None:
    cfg = _config(config, profile, [], [])
    console.print_json(
        json.dumps(ModelManager(cfg, root, model_manifest_path()).status(), ensure_ascii=False)
    )


@models_app.command("verify")
def models_verify(
    root: Path = typer.Option(Path("./models"), "--root"),
    config: Path | None = typer.Option(None, "--config"),
    profile: str = typer.Option("balanced", "--profile"),
) -> None:
    cfg = _config(config, profile, [], [])
    console.print_json(
        json.dumps(ModelManager(cfg, root, model_manifest_path()).verify(), ensure_ascii=False)
    )


@models_app.command("install-ocr")
def models_install_ocr(
    gpu: bool = typer.Option(False, "--gpu"),
    root: Path = typer.Option(Path("./models"), "--root"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    cfg = _config(config, "balanced", [], [])
    ModelManager(cfg, root, model_manifest_path()).install_ocr(gpu=gpu)
    console.print("OCR 运行时安装完成。模型会由 RapidOCR 在首次使用时按其机制准备。")


@models_app.command("install-lama")
def models_install_lama(
    source: str = typer.Option(..., "--source", help="本地 ONNX 文件或可信 URL"),
    root: Path = typer.Option(Path("./models"), "--root"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    cfg = _config(config, "balanced", [], [])
    path = ModelManager(cfg, root, model_manifest_path()).install_lama(source)
    console.print(f"LaMa 已安装：{path}\n设置环境变量 VSRX_LAMA_MODEL={path}")


@models_app.command("install-propainter")
def models_install_propainter(
    root: Path = typer.Option(Path("./models"), "--root"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    cfg = _config(config, "balanced", [], [])
    path = ModelManager(cfg, root, model_manifest_path()).install_propainter()
    console.print(f"ProPainter 已安装：{path}\n设置环境变量 VSRX_PROPAINTER_REPO={path}")


@models_app.command("install-sttn")
def models_install_sttn(
    checkpoint: str | None = typer.Option(None, "--checkpoint", help="本地文件或可信 URL"),
    root: Path = typer.Option(Path("./models"), "--root"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    cfg = _config(config, "balanced", [], [])
    path = ModelManager(cfg, root, model_manifest_path()).install_sttn(checkpoint)
    console.print(f"STTN 仓库已准备：{path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
