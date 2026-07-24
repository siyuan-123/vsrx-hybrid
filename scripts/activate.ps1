$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

. (Join-Path $Root ".venv\Scripts\Activate.ps1")
$env:VSRX_LAMA_MODEL = Join-Path $Root "models\lama.onnx"
$env:VSRX_MIGAN_MODEL = Join-Path $Root "models\migan.onnx"
$env:VSRX_PROPAINTER_REPO = Join-Path $Root "models\ProPainter"
$env:VSRX_STTN_REPO = Join-Path $Root "models\STTN"
$env:VSRX_STTN_CHECKPOINT = Join-Path $Root "models\STTN\checkpoints\sttn.pth"

Set-Location $Root
Write-Host "VSR-X 环境已激活：$Root"
