<#
.SYNOPSIS
    Builds the self-contained offline runtime and model folders inside portable_ocr/
    on an internet-connected Windows x64 staging machine before copying to an air-gapped PC.
#>
$ErrorActionPreference = "Stop"

$RepoRoot     = Split-Path -Parent $PSScriptRoot
$PortableRoot = Join-Path $RepoRoot "portable_ocr"
$PythonDir    = Join-Path $PortableRoot "runtime\python"
$LlamaDir     = Join-Path $PortableRoot "runtime\llama"
$DoclingDir   = Join-Path $PortableRoot "models\docling"
$VlmDir       = Join-Path $PortableRoot "models\vlm"

New-Item -ItemType Directory -Force -Path $PythonDir, $LlamaDir, $DoclingDir, $VlmDir | Out-Null

Write-Host "=== [1/5] Downloading Relocatable Python 3.11.9 (Windows x64) ===" -ForegroundColor Cyan
$PyZip = Join-Path $env:TEMP "python-3.11.9-embed-amd64.zip"
if (-not (Test-Path (Join-Path $PythonDir "python.exe"))) {
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -OutFile $PyZip
    Expand-Archive -Path $PyZip -DestinationPath $PythonDir -Force
}

# Configure relative python311._pth
@"
python311.zip
.
DLLs
Lib
Lib\site-packages
..\..\src
import site
"@ | Set-Content -Path (Join-Path $PythonDir "python311._pth") -Encoding ASCII

$PyExe = Join-Path $PythonDir "python.exe"
if (-not (Test-Path (Join-Path $PythonDir "Scripts\pip.exe"))) {
    $GetPip = Join-Path $env:TEMP "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
    & $PyExe $GetPip --no-warn-script-location
}

Write-Host "=== [2/5] Installing Offline Python Packages (CPU PyTorch + Docling + RapidOCR) ===" -ForegroundColor Cyan
& $PyExe -m pip install --no-warn-script-location -r (Join-Path $PortableRoot "requirements.txt")

Write-Host "=== [3/5] Bundling Microsoft VC++ / OpenMP Runtime DLLs ===" -ForegroundColor Cyan
$VcDlls = @(
    "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll",
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
    "concrt140.dll", "vccorlib140.dll", "vcomp140.dll"
)
$TargetDllDirs = @(
    $PythonDir,
    (Join-Path $PythonDir "DLLs"),
    (Join-Path $PythonDir "Lib\site-packages\torch\lib"),
    (Join-Path $PythonDir "Lib\site-packages\onnxruntime\capi"),
    $LlamaDir
)
foreach ($d in $TargetDllDirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
    foreach ($dll in $VcDlls) {
        $sysDll = Join-Path $env:SystemRoot "System32\$dll"
        if (Test-Path $sysDll) {
            Copy-Item -Path $sysDll -Destination (Join-Path $d $dll) -Force
        }
    }
}

Write-Host "=== [4/5] Pre-fetching Offline IBM Docling Models (DocLayNet, TableFormer, RapidOCR) ===" -ForegroundColor Cyan
$env:HF_HUB_DISABLE_XET = "1"
& $PyExe -c "from docling.utils.model_downloader import download_models; from pathlib import Path; download_models(output_dir=Path(r'$DoclingDir'), with_layout=True, with_tableformer=True, with_rapidocr=True, with_easyocr=False)"

Write-Host "=== [5/5] Downloading Portable llama.cpp CPU Server & Qwen3.5-4B GGUF VLM ===" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $LlamaDir "llama-server.exe"))) {
    $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    $Asset = $Release.assets | Where-Object { $_.name -match "bin-win-cpu-x64\.zip$" } | Select-Object -First 1
    $LlamaZip = Join-Path $env:TEMP $Asset.name
    Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $LlamaZip
    Expand-Archive -Path $LlamaZip -DestinationPath $LlamaDir -Force
}

$ModelGguf  = Join-Path $VlmDir "qwen3.5-4b-model.gguf"
$MmprojGguf = Join-Path $VlmDir "qwen3.5-4b-mmproj.gguf"
if (-not (Test-Path $ModelGguf)) {
    Invoke-WebRequest -Uri "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf" -OutFile $ModelGguf
}
if (-not (Test-Path $MmprojGguf)) {
    Invoke-WebRequest -Uri "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf" -OutFile $MmprojGguf
}

Write-Host "`nPortable air-gapped bundle ready at: $PortableRoot" -ForegroundColor Green
