# Qwen3-TTS Studio installer.
# Everything is installed inside this folder: a private Python, the packages,
# the model weights and all caches. Nothing is written to the C: drive.
# If you already have a Python with GPU PyTorch, the installer offers to reuse
# that PyTorch instead of downloading another copy.
#
# Run it through install.bat. Options:
#   -Yes               don't ask any questions (reuses an existing PyTorch if one is found)
#   -SkipModels        don't download model weights (do it later with install.bat)
#   -SkipGguf          don't download KoboldCpp and the GGUF model (Speak gets them when you pick GGUF)
#   -AllModels         also download the other clone model size
#   -Reset             replace settings.json with fresh recommendations for this PC
#   -Cpu               install without GPU support (very slow, for testing only)
#   -Fresh             always install a private PyTorch, even if one could be reused
#   -UsePython <path>  reuse PyTorch from this python.exe (e.g. a conda env or another app's venv)
#   -KeepCache         keep the download caches instead of deleting them at the end

param(
    [switch]$Yes,
    [switch]$SkipModels,
    [switch]$SkipGguf,
    [switch]$AllModels,
    [switch]$Reset,
    [switch]$Cpu,
    [switch]$Fresh,
    [string]$UsePython,
    [switch]$KeepCache
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # makes Invoke-WebRequest much faster
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$TorchVersion = '2.8.0'
$PythonVersion = '3.12'
# An existing PyTorch is reused only if it is at least this new.
$MinReuseTorch = [version]'2.6.0'

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

$venv = Join-Path $Root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
$reuseMarker = Join-Path $venv 'reused-python.json'
$reuseCfg = $null

# --- 3. Reuse a Python that already has GPU PyTorch? ---------------------------
function Find-PythonCandidates {
    $found = New-Object System.Collections.Generic.List[string]
    if ($UsePython) {
        $found.Add($UsePython)
    } else {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($l in @(& py -0p 2>$null)) {
                if ($l -match '([A-Za-z]:\\.*?python\.exe)\s*$') { $found.Add($Matches[1]) }
            }
        }
        foreach ($cmd in @(Get-Command python, python3 -All -ErrorAction SilentlyContinue)) {
            if ($cmd.Source) { $found.Add($cmd.Source) }
        }
    }
    # Skip the Microsoft Store stub and anything inside this folder.
    return @($found | Where-Object {
        $_ -and (Test-Path $_) -and ($_ -notmatch 'WindowsApps') -and -not $_.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)
    } | Select-Object -Unique)
}

function Probe-Python([string]$exe) {
    $out = & $exe (Join-Path $PSScriptRoot 'probe_python.py') 2>$null
    $line = @($out) | Where-Object { $_ -like 'PROBE:*' } | Select-Object -First 1
    if (-not $line) { return $null }
    try { return ($line.Substring(6) | ConvertFrom-Json) } catch { return $null }
}

function Torch-BaseVersion([string]$v) {
    if ($v -match '^(\d+\.\d+(\.\d+)?)') { return [version]$Matches[1] }
    return $null
}

function Find-ReusableTorch {
    $best = $null
    $ErrorActionPreference = 'Continue'
    foreach ($exe in (Find-PythonCandidates)) {
        Info "Checking $exe"
        $p = Probe-Python $exe
        if (-not $p) { continue }
        if ($p.version -notin @('3.10', '3.11', '3.12', '3.13')) { Info "  Python $($p.version): not supported"; continue }
        if (-not $p.torch) { Info "  Python $($p.version), no PyTorch"; continue }
        $tv = Torch-BaseVersion $p.torch
        if (-not $tv -or $tv -lt $MinReuseTorch) { Info "  PyTorch $($p.torch): too old (need $MinReuseTorch+)"; continue }
        if ($torchIndex -ne 'cpu' -and -not $p.cuda) { Info "  PyTorch $($p.torch) can't use your GPU"; continue }
        Info "  PyTorch $($p.torch), GPU: $($p.cuda)  -> usable"
        if (-not $best -or $tv -gt (Torch-BaseVersion $best.torch)) { $best = $p }
    }
    return $best
}

if ((Test-Path $reuseMarker) -and $Fresh) {
    Step 'Switching to a private PyTorch (-Fresh): removing the old .venv'
    Remove-Item -Recurse -Force $venv
} elseif ($UsePython -and (Test-Path $py)) {
    Step "Rebuilding .venv on top of $UsePython (-UsePython)"
    Remove-Item -Recurse -Force $venv
}

if (Test-Path $reuseMarker) {
    $reuseCfg = Get-Content $reuseMarker -Raw | ConvertFrom-Json
    Step "Reusing PyTorch from $($reuseCfg.exe) (set up on an earlier run)"
} elseif (-not (Test-Path $py) -and -not $Fresh -and ($torchIndex -ne 'cpu' -or $UsePython)) {
    Step 'Looking for a Python that already has GPU PyTorch'
    $cand = Find-ReusableTorch
    if ($cand -and $UsePython) {
        $reuseCfg = $cand
    } elseif ($cand) {
        Write-Host ''
        Info "Found PyTorch $($cand.torch) in $($cand.exe)"
        Info 'Reusing it skips a ~3 GB download. That Python stays untouched; Qwen3-TTS gets its'
        Info 'own .venv that borrows its PyTorch. If you later remove or upgrade that Python,'
        Info 'run install.bat -Fresh.'
        $use = $true
        if (-not $Yes) {
            $answer = Read-Host '    Reuse it? (Y/n)'
            $use = $answer -notmatch '^[nN]'
        }
        if ($use) { $reuseCfg = $cand }
    } elseif ($UsePython) {
        Warn "Can't reuse PyTorch from $UsePython (see above). Installing a private copy instead."
    } else {
        Info 'None found, installing a private copy.'
    }
}

# --- 4. Python environment and packages -----------------------------------------
function Install-Requirements([string[]]$extra) {
    Step 'Installing Qwen3-TTS and the app requirements'
    # yt-dlp has to keep up with YouTube, so running install.bat again always updates it.
    $reqArgs = @('pip', 'install', '--python', $py, '-r', (Join-Path $Root 'requirements.txt'),
        '--upgrade-package', 'yt-dlp', '--upgrade-package', 'yt-dlp-ejs') + $extra
    Run $uv $reqArgs 'Installing requirements'
}

if ($reuseCfg) {
    if (-not (Test-Path $py)) {
        Step "Creating .venv on top of $($reuseCfg.exe)"
        Run $uv @('venv', '--python', $reuseCfg.exe, $venv) 'Creating the Python environment'
        # A .pth file adds the existing site-packages after .venv's own, so .venv's
        # packages win and only what's missing (PyTorch) comes from the other Python.
        $sitePackages = Join-Path $venv 'Lib\site-packages'
        $dirs = @($reuseCfg.site_dirs) -join "`r`n"
        [IO.File]::WriteAllText((Join-Path $sitePackages 'zz_reuse_existing_python.pth'), $dirs + "`r`n")
        ($reuseCfg | ConvertTo-Json) | Set-Content -Path $reuseMarker -Encoding UTF8
    }

    $tv = Torch-BaseVersion $reuseCfg.torch
    $tav = Torch-BaseVersion ([string]$reuseCfg.torchaudio)
    if (-not $tav -or $tav -ne $tv) {
        # torchaudio must match the torch version, so add the matching build to .venv.
        $tag = 'cpu'
        if ($reuseCfg.cuda_version) { $tag = 'cu' + ($reuseCfg.cuda_version -replace '\.', '') }
        Step "Adding torchaudio $tv ($tag) to match your PyTorch"
        Run $uv @('pip', 'install', '--python', $py, '--no-deps', "torchaudio==$tv",
            '--index-url', "https://download.pytorch.org/whl/$tag") 'Installing torchaudio'
    }

    # Leave torch/torchaudio out of resolution so uv doesn't install a second copy.
    $cacheDir = Join-Path $Root '.cache'
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    $excludes = Join-Path $cacheDir 'reuse-excludes.txt'
    $constraints = Join-Path $cacheDir 'reuse-constraints.txt'
    Set-Content -Path $excludes -Value @('torch', 'torchaudio') -Encoding ASCII
    Get-Content (Join-Path $Root 'constraints.txt') | Where-Object { $_ -notmatch '^\s*torch' } |
        Set-Content -Path $constraints -Encoding ASCII
    Install-Requirements @('--excludes', $excludes, '-c', $constraints)

    Step 'Checking that the reused PyTorch works here'
    $ErrorActionPreference = 'Continue'
    $check = & $py -c "import torch, torchaudio, qwen_tts; print('OK', torch.__version__, torch.cuda.is_available())" 2>&1
    $ErrorActionPreference = 'Stop'
    $okLine = @($check) | Where-Object { "$_" -like 'OK *' } | Select-Object -First 1
    if (-not $okLine -or ($torchIndex -ne 'cpu' -and "$okLine" -notmatch 'True$')) {
        Warn 'The reused PyTorch did not work inside Qwen3-TTS:'
        @($check) | Select-Object -Last 5 | ForEach-Object { Warn "  $_" }
        Warn 'Falling back to a private copy.'
        Remove-Item -Recurse -Force $venv
        $reuseCfg = $null
    } else {
        Info "$okLine"
    }
}

if (-not $reuseCfg) {
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    Step "Creating a private Python $PythonVersion environment in .venv"
    if (-not (Test-Path $py)) {
        Run $uv @('venv', '--python', $PythonVersion, $venv) 'Creating the Python environment'
    } else {
        Info 'Already exists, reusing it.'
    }

    Step "Installing PyTorch $TorchVersion ($torchIndex). This is a big download (about 3 GB)."
    Run $uv @('pip', 'install', '--python', $py,
        "torch==$TorchVersion", "torchaudio==$TorchVersion",
        '--index-url', "https://download.pytorch.org/whl/$torchIndex") 'Installing PyTorch'

    Install-Requirements @('-c', (Join-Path $Root 'constraints.txt'))
}

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

# --- 6b. KoboldCpp and the GGUF model, for Speak's GGUF option ------------------------
if ($SkipModels -or $SkipGguf) {
    Step 'Skipping KoboldCpp and the GGUF model. Speak downloads them the first time you pick GGUF.'
} else {
    Step 'Downloading KoboldCpp and the GGUF voice model (about 2.3 GB, resumes if interrupted)'
    $ErrorActionPreference = 'Continue'
    & $py -m app.gguf
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) {
        Warn 'That download failed, but everything else is installed. Run install.bat again to retry,'
        Warn 'or Speak downloads it the first time you pick the GGUF model.'
    }
}

# --- 6c. Music remover and speech-to-text, for cloning from YouTube and transcribing clips ---
if ($SkipModels) {
    Step 'Skipping the music remover and speech-to-text models. They download the first time they are needed.'
} else {
    Step 'Downloading the music remover and speech-to-text models (about 1.8 GB, resumes if interrupted)'
    $ErrorActionPreference = 'Continue'
    & $py -m app.youtube --setup
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) {
        Warn 'That failed, but everything else is installed. Run install.bat again to retry,'
        Warn 'or they download the first time they are needed.'
    }
}

# --- 7. Clean up -------------------------------------------------------------------
# The caches only speed up re-installs; the venv is a full copy (UV_LINK_MODE=copy),
# so deleting them is safe and frees several GB.
if (-not $KeepCache) {
    Step 'Cleaning up download caches'
    $caches = @((Join-Path $Root '.cache\uv'), (Join-Path $env:HF_HOME 'xet'))
    $freed = 0
    foreach ($d in $caches) {
        if (Test-Path $d) {
            $freed += (Get-ChildItem $d -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
        }
    }
    Run $uv @('cache', 'clean') 'Cleaning the package cache'
    if (Test-Path $caches[1]) { Remove-Item -Recurse -Force $caches[1] -ErrorAction SilentlyContinue }
    Info ('Freed {0:N1} GB' -f ($freed / 1GB))
}

Write-Host ''
Write-Host 'All done!' -ForegroundColor Green
Write-Host '  start_studio.bat   design and clone voices'
Write-Host '  start_server.bat   voice server for SillyTavern and other apps'
Write-Host '  start_speak.bat    type something and hear it in one of your voices'
Write-Host ''
