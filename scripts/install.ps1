# Qwen3-TTS Studio installer.
# Everything is installed inside this folder: a private Python, the packages,
# the model weights and all caches. Nothing is written to the C: drive.
#
# Run it through install.bat. Options:
#   -Yes          don't ask any questions
#   -SkipModels   don't download model weights (do it later with install.bat)
#   -AllModels    also download the other clone model size
#   -Reset        replace settings.json with fresh recommendations for this PC
#   -Cpu          install without GPU support (very slow, for testing only)

param(
    [switch]$Yes,
    [switch]$SkipModels,
    [switch]$AllModels,
    [switch]$Reset,
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # makes Invoke-WebRequest much faster
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$TorchVersion = '2.8.0'
$PythonVersion = '3.12'

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($msg) { Write-Host ''; Write-Host "==> $msg" -ForegroundColor Cyan }
function Info($msg) { Write-Host "    $msg" }
function Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }
function Fail($msg) {
    Write-Host ''
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}
function Run($exe, [string[]]$argList, $what) {
    # Native tools print progress on stderr; don't let PowerShell treat that as an error.
    $ErrorActionPreference = 'Continue'
    & $exe @argList
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { Fail "$what failed (exit code $code). Scroll up for details." }
}

Write-Host ''
Write-Host 'Qwen3-TTS Studio installer' -ForegroundColor Green
Write-Host "Install folder: $Root"

# --- 0. Location check ------------------------------------------------------
if ($Root -match '\s') {
    Warn 'The install folder path contains spaces. That usually works, but G:\Qwen3TTS is safest.'
}
$drive = Split-Path -Qualifier $Root
if ($drive -ne 'G:' -and -not $Yes) {
    Warn "This folder is on $drive, not G:."
    Warn 'To install on G:, close this window, move the folder to G:\Qwen3TTS and run install.bat again.'
    $answer = Read-Host '    Install here anyway? (y/N)'
    if ($answer -notmatch '^[yY]') { exit 1 }
}

# Keep every cache inside the install folder.
$env:UV_CACHE_DIR = Join-Path $Root '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root '.python'
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_LINK_MODE = 'copy'
$env:HF_HOME = Join-Path $Root '.cache\huggingface'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:PYTHONUTF8 = '1'

# --- 1. GPU ------------------------------------------------------------------
Step 'Checking your graphics card'
$gpuName = $null
$vramGb = 0
$cc = 0.0
$inv = [Globalization.CultureInfo]::InvariantCulture
function Query-Gpu([string]$fields) {
    $out = & nvidia-smi "--query-gpu=$fields" --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0) { return $null }
    $first = @($out) | Select-Object -First 1
    if (-not $first -or ($first -notmatch ',')) { return $null }
    return @($first.Split(',') | ForEach-Object { $_.Trim() })
}
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $ErrorActionPreference = 'Continue'
    try {
        $parts = Query-Gpu 'name,memory.total,compute_cap'
        # Older drivers don't know compute_cap.
        if (-not $parts) { $parts = Query-Gpu 'name,memory.total' }
        if ($parts) {
            $gpuName = $parts[0]
            $num = 0.0
            if ([double]::TryParse($parts[1], [Globalization.NumberStyles]::Float, $inv, [ref]$num)) {
                $vramGb = [math]::Round($num / 1024, 1)
            }
            if ($parts.Count -ge 3 -and [double]::TryParse($parts[2], [Globalization.NumberStyles]::Float, $inv, [ref]$num)) {
                $cc = $num
            }
        }
    } catch {
        $gpuName = $null
    }
    $ErrorActionPreference = 'Stop'
}

if ($Cpu) {
    $torchIndex = 'cpu'
    Warn 'Installing the CPU-only version (-Cpu). Generation will be very slow.'
} elseif (-not $gpuName) {
    $torchIndex = 'cpu'
    Warn 'No NVIDIA graphics card found (nvidia-smi is missing or failed).'
    Warn 'Qwen3-TTS needs an NVIDIA GPU to be usable. Continuing with the slow CPU version.'
    if (-not $Yes) {
        $answer = Read-Host '    Continue anyway? (y/N)'
        if ($answer -notmatch '^[yY]') { exit 1 }
    }
} else {
    Info "$gpuName, $vramGb GB VRAM"
    # CUDA 12.8 builds cover RTX 20-series and newer (including RTX 50-series).
    # Older cards (GTX 10-series and earlier) use the CUDA 12.6 build.
    if ($cc -ge 7.5) { $torchIndex = 'cu128' }
    elseif ($cc -gt 0) { $torchIndex = 'cu126' }
    elseif ($gpuName -match 'RTX') { $torchIndex = 'cu128' }
    else { $torchIndex = 'cu126' }
    if ($vramGb -gt 0 -and $vramGb -lt 6) { Warn 'Under 6 GB of VRAM: the smaller 0.6B clone model will be used.' }
}
Info "PyTorch build: $torchIndex"

# --- 2. uv (Python + package manager) -----------------------------------------
Step 'Getting the Python package manager (uv)'
$uvDir = Join-Path $Root '.tools\uv'
$uv = Join-Path $uvDir 'uv.exe'
if (-not (Test-Path $uv)) {
    New-Item -ItemType Directory -Force -Path $uvDir | Out-Null
    $zip = Join-Path $uvDir 'uv.zip'
    $url = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    } catch {
        Fail "Could not download uv from $url. Check your internet connection. ($($_.Exception.Message))"
    }
    Expand-Archive -Path $zip -DestinationPath $uvDir -Force
    Remove-Item $zip
    if (-not (Test-Path $uv)) { Fail "uv.exe was not found after extracting $url" }
}
Info (& $uv --version)

# --- 3. Python environment ---------------------------------------------------
Step "Creating a private Python $PythonVersion environment in .venv"
$venv = Join-Path $Root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $py)) {
    Run $uv @('venv', '--python', $PythonVersion, $venv) 'Creating the Python environment'
} else {
    Info 'Already exists, reusing it.'
}

# --- 4. Packages ---------------------------------------------------------------
Step "Installing PyTorch $TorchVersion ($torchIndex). This is a big download (about 3 GB)."
Run $uv @('pip', 'install', '--python', $py,
    "torch==$TorchVersion", "torchaudio==$TorchVersion",
    '--index-url', "https://download.pytorch.org/whl/$torchIndex") 'Installing PyTorch'

Step 'Installing Qwen3-TTS and the app requirements'
Run $uv @('pip', 'install', '--python', $py,
    '-r', (Join-Path $Root 'requirements.txt'),
    '-c', (Join-Path $Root 'constraints.txt')) 'Installing requirements'

# --- 5. Settings ---------------------------------------------------------------
Step 'Choosing settings for your PC'
$cfgArgs = @('-m', 'app.configure')
if ($Reset) { $cfgArgs += '--reset' }
if ($torchIndex -ne 'cpu') { $cfgArgs += '--expect-gpu' }
Run $py $cfgArgs 'Configuring'

# --- 6. Models -------------------------------------------------------------------
if ($SkipModels) {
    Step 'Skipping model download (-SkipModels). Run install.bat again to download them.'
} else {
    Step 'Downloading the Qwen3-TTS models (about 9 GB, resumes if interrupted)'
    $dlArgs = @('-m', 'app.download_models')
    if ($AllModels) { $dlArgs += '--all' }
    Run $py $dlArgs 'Downloading models'
}

Write-Host ''
Write-Host 'All done!' -ForegroundColor Green
Write-Host '  start_studio.bat   design and clone voices'
Write-Host '  start_server.bat   voice server for SillyTavern and other apps'
Write-Host ''
