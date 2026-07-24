$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = if ($env:VSRX_VENV) { $env:VSRX_VENV } else { Join-Path $Root ".venv" }
$Extra = if ($env:VSRX_INSTALL_EXTRA) { $env:VSRX_INSTALL_EXTRA } else { "ocr,test" }
$Python = if ($env:PYTHON) { $env:PYTHON } else { "py" }

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is required" }
if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) { throw "ffprobe is required" }

& $Python -3.11 -m venv $Venv
$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate
python -m pip install --upgrade pip
python -m pip install -e "$Root[$Extra]"
vsrx doctor --models-root (Join-Path $Root "models")
Write-Host "Installed. Activate with: $Activate"
