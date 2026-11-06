# Qwen3-TTS Studio installer.
# Everything is installed in one folder: the app, a private Python, the packages,
# the model weights and all caches. Nothing else is written to the C: drive apart
# from the optional desktop and Start menu shortcuts.
# If you already have a Python with GPU PyTorch, the installer offers to reuse
# that PyTorch instead of downloading another copy.
#
# Run it through Qwen3-TTS.bat or install.bat. The first time, it asks where to
# install: in this folder, or in another one (the app is copied there first).
# Options:
#   -InstallDir <path> install into this folder without asking (it is created if needed)
#   -Yes               don't ask any questions (installs here, reuses an existing PyTorch if one is found)
#   -NoShortcut        don't add the desktop and Start menu shortcuts
#   -SkipModels        don't download model weights (do it later with install.bat)
#   -SkipGguf          don't download KoboldCpp and the GGUF model (Speak and the voice server get them when you pick GGUF)
#   -AllModels         also download the other clone model size
#   -Reset             replace settings.json with fresh recommendations for this PC
#   -Cpu               install without GPU support (very slow, for testing only)
#   -Fresh             always install a private PyTorch, even if one could be reused
#   -UsePython <path>  reuse PyTorch from this python.exe (e.g. a conda env or another app's venv)
#   -KeepCache         keep the download caches instead of deleting them at the end
#   -ModelsFrom <path> reuse the models in another Qwen3-TTS Studio folder instead of downloading them
#
# Anything already installed is skipped: Python, PyTorch, packages and models are
# checked first, so running it again is quick and works without internet. Models
# already on this PC (in the Hugging Face cache other AI apps share, or in another
# Qwen3-TTS Studio folder) are linked or copied instead of downloaded.

param(
    [switch]$Yes,
    [switch]$SkipModels,
    [switch]$SkipGguf,
    [switch]$AllModels,
    [switch]$Reset,
    [switch]$Cpu,
    [switch]$Fresh,
    [string]$UsePython,
    [switch]$KeepCache,
    [string]$InstallDir,
    [switch]$NoShortcut,
    [string]$ModelsFrom
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

$Version = 'dev'
$versionFile = Join-Path $Root 'VERSION'
if (Test-Path -LiteralPath $versionFile) { $Version = (Get-Content -LiteralPath $versionFile | Select-Object -First 1).Trim() }

Write-Host ''
Write-Host "Qwen3-TTS Studio $Version installer" -ForegroundColor Green

# --- 0. Where to install ---------------------------------------------------------
# About how much space a full install takes (models, PyTorch, packages, KoboldCpp).
$NeedGb = 24
$ShortcutName = 'Qwen3-TTS Studio.lnk'
$installedHere = Test-Path (Join-Path $Root '.venv\Scripts\python.exe')

function Same-Dir([string]$a, [string]$b) {
    return [string]::Equals($a.TrimEnd('\'), $b.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)
}
function Is-Under([string]$dir, [string]$parent) {
    if (-not $parent) { return $false }
    return ($dir.TrimEnd('\') + '\').StartsWith($parent.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)
}
function Normalize-Dir([string]$p) {
    # "D", "D:" or "D:\" mean a Qwen3TTS folder on that drive. Anything else needs a full path.
    $p = $p.Trim().Trim('"').Trim()
    if ($p -match '^[A-Za-z]:?\\?$') { return $p.Substring(0, 1).ToUpper() + ':\Qwen3TTS' }
    if ($p -notmatch '^[A-Za-z]:\\') { return $null }
    try { return [IO.Path]::GetFullPath($p).TrimEnd('\') } catch { return $null }
}
function Free-Gb([string]$dir) {
    try { return [math]::Floor((New-Object IO.DriveInfo (Split-Path -Qualifier $dir)).AvailableFreeSpace / 1GB) } catch { return $null }
}
function Fixed-Drives {
    return @([IO.DriveInfo]::GetDrives() | Where-Object { $_.DriveType -eq 'Fixed' -and $_.IsReady })
}
function In-OneDrive([string]$dir) {
    foreach ($od in @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
        if (Is-Under $dir $od) { return $true }
    }
    return $false
}
function Downloads-Dir {
    try {
        $p = (New-Object -ComObject Shell.Application).NameSpace('shell:Downloads').Self.Path
        if ($p) { return $p }
    } catch {}
    return (Join-Path $env:USERPROFILE 'Downloads')
}
function Find-PreviousInstall {
    # The shortcuts an earlier install made point at its Qwen3-TTS.bat.
    try {
        $ws = New-Object -ComObject WScript.Shell
        foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
            $lnk = Join-Path $dir $ShortcutName
            if (-not (Test-Path -LiteralPath $lnk)) { continue }
            $t = $ws.CreateShortcut($lnk).TargetPath
            if ($t -and (Test-Path -LiteralPath $t)) { return (Split-Path -Parent $t) }
        }
    } catch {}
    return $null
}
function Folder-Problem([string]$dir) {
    # Why $dir can't be used, or $null when it can.
    $qualifier = Split-Path -Qualifier $dir
    if (-not (Test-Path -LiteralPath "$qualifier\")) { return "There is no $qualifier drive on this PC." }
    if (-not (Same-Dir $dir $Root) -and (Is-Under $dir $Root)) { return 'That folder is inside this one. Pick a folder somewhere else.' }
    if (Is-Under $dir ([IO.Path]::GetTempPath())) { return 'That is a temporary folder, which Windows empties now and then. Pick another one.' }
    return $null
}
function Writable-Problem([string]$dir) {
    try {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $probe = Join-Path $dir ('.write-test-' + [guid]::NewGuid().ToString('N'))
        [IO.File]::WriteAllText($probe, 'ok')
        Remove-Item -LiteralPath $probe -Force
    } catch {
        return "Windows won't let the installer write to $dir. Pick a folder of your own, such as D:\Qwen3TTS (not inside Program Files)."
    }
    return $null
}
function Confirm-Folder([string]$dir) {
    # Point out anything that would make $dir a poor choice and let the user decide.
    if ($dir -match '\s') { Warn 'The path has spaces in it. That should work, but a path without spaces (like D:\Qwen3TTS) is the safest.' }
    if ($dir -match '[^\x00-\x7F]') { Warn 'The path has accented or non-English letters in it. Some tools trip over those; plain letters are the safest.' }
    $notes = @()
    if (In-OneDrive $dir) { $notes += "$dir is synced by OneDrive, which would upload the models (about $NeedGb GB) to the cloud." }
    if (Is-Under $dir (Downloads-Dir)) { $notes += 'That is inside your Downloads folder, which Windows can clean up automatically.' }
    $free = Free-Gb $dir
    $hasInstall = Test-Path (Join-Path $dir '.venv\Scripts\python.exe')
    if (-not $hasInstall -and $null -ne $free -and $free -lt $NeedGb) {
        $notes += "$(Split-Path -Qualifier $dir) has $free GB free, and a full install needs about $NeedGb GB."
    }
    if (-not $notes) { return $true }
    foreach ($n in $notes) { Warn $n }
    $answer = Read-Host '    Use it anyway? (y/N)'
    return ($answer -match '^[yY]')
}
function Write-Marker([string]$dest) {
    # Qwen3-TTS.bat in this folder reads this and opens the installed copy instead.
    $cp = [Globalization.CultureInfo]::CurrentCulture.TextInfo.OEMCodePage
    $enc = if ($cp -eq 65001) { New-Object Text.UTF8Encoding $false } else { [Text.Encoding]::GetEncoding($cp) }
    [IO.File]::WriteAllText((Join-Path $Root '.installed-to.txt'), $dest + "`r`n", $enc)
}
function Copy-AppTo([string]$dest) {
    Step "Copying Qwen3-TTS Studio to $dest"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $opts = @('/R:2', '/W:2', '/NFL', '/NDL', '/NJH', '/NJS', '/NP')
    $ErrorActionPreference = 'Continue'
    # The code folders are mirrored so files dropped in a newer version don't linger.
    # Voices, models, settings and the rest of an existing install are left alone.
    foreach ($sub in 'app', 'scripts') {
        & robocopy (Join-Path $Root $sub) (Join-Path $dest $sub) /MIR /XD __pycache__ @opts | Out-Null
        if ($LASTEXITCODE -ge 8) { Fail "Copying the $sub folder to $dest failed (robocopy code $LASTEXITCODE)." }
    }
    & robocopy $Root $dest /XF settings.json .installed-to.txt @opts | Out-Null
    if ($LASTEXITCODE -ge 8) { Fail "Copying to $dest failed (robocopy code $LASTEXITCODE)." }
    $ErrorActionPreference = 'Stop'
}

$target = $null
if ($InstallDir) {
    $target = Normalize-Dir $InstallDir
    if (-not $target) { Fail "-InstallDir needs a full path such as D:\Qwen3TTS (got: $InstallDir)" }
    if (-not (Same-Dir $target $Root)) {
        $problem = Folder-Problem $target
        if (-not $problem) { $problem = Writable-Problem $target }
        if ($problem) { Fail $problem }
    }
} elseif (-not $installedHere -and -not $Yes) {
    $previous = Find-PreviousInstall
    if ($previous -and (Same-Dir $previous $Root)) { $previous = $null }
    $default = $Root
    $unsafe = (Is-Under $Root ([IO.Path]::GetTempPath())) -or (In-OneDrive $Root) -or (Is-Under $Root (Downloads-Dir))
    if ($previous) {
        $default = $previous
    } elseif ($unsafe) {
        # Suggest the drive with the most free space instead.
        $best = Fixed-Drives | Sort-Object AvailableFreeSpace -Descending | Select-Object -First 1
        if ($best) { $default = $best.Name.Substring(0, 2) + '\Qwen3TTS' }
    }

    Write-Host ''
    if ($previous) {
        Info "Qwen3-TTS Studio is already installed in $previous."
        Info 'Press Enter to update it there (your voices, settings and models are kept),'
        Info 'or type a different folder to install a separate copy.'
    } else {
        Info 'Where should it be installed? Everything goes into one folder: the programs,'
        Info "a private copy of Python and the voice models (about $NeedGb GB in all)."
        Info "Press Enter for $default, or type a different folder (for example D:\Qwen3TTS)."
    }
    $drives = Fixed-Drives | ForEach-Object { '{0} {1:N0} GB' -f $_.Name.Substring(0, 2), [math]::Floor($_.AvailableFreeSpace / 1GB) }
    if ($drives) { Info ('Free space: ' + ($drives -join ',  ')) }
    while (-not $target) {
        Write-Host ''
        $answer = Read-Host "    Install folder [$default]"
        $dir = if ($answer.Trim()) { Normalize-Dir $answer } else { $default }
        if (-not $dir) { Warn 'Type a full path that starts with a drive letter, such as D:\Qwen3TTS.'; continue }
        $problem = Folder-Problem $dir
        if ($problem) { Warn $problem; continue }
        if (-not ((Same-Dir $dir $previous) -or (Confirm-Folder $dir))) { continue }
        $problem = Writable-Problem $dir
        if ($problem) { Warn $problem; continue }
        $target = $dir
    }
} elseif (-not $installedHere) {
    $free = Free-Gb $Root
    if ($null -ne $free -and $free -lt $NeedGb) { Warn "This drive has $free GB free, and a full install needs about $NeedGb GB." }
}

if ($target -and -not (Same-Dir $target $Root)) {
    Copy-AppTo $target
    Write-Marker $target
    # Carry on with the copy's own installer, which installs into its folder.
    $forward = @{}
    foreach ($k in $PSBoundParameters.Keys) { $forward[$k] = $PSBoundParameters[$k] }
    $forward['InstallDir'] = $target
    # The copy can reuse models from this folder if it has any.
    $env:QWEN3TTS_OTHER_INSTALLS = (@($Root, $env:QWEN3TTS_OTHER_INSTALLS) | Where-Object { $_ }) -join ';'
    & (Join-Path $target 'scripts\install.ps1') @forward
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Info "You can delete the folder you downloaded ($Root) now: everything runs from $target."
        Write-Host ''
    }
    exit $code
}

Write-Host "Install folder: $Root"

# Files extracted from a downloaded ZIP are marked as coming from the internet, which
# makes Windows ask before every start. They're installed now, so drop that mark.
$ErrorActionPreference = 'Continue'
Get-ChildItem -LiteralPath $Root -File -Force | Unblock-File -ErrorAction SilentlyContinue
foreach ($sub in 'app', 'scripts') {
    $d = Join-Path $Root $sub
    if (Test-Path -LiteralPath $d) { Get-ChildItem -LiteralPath $d -Recurse -File -Force | Unblock-File -ErrorAction SilentlyContinue }
}
$ErrorActionPreference = 'Stop'

# Models already on this PC are reused instead of downloaded (see app\reuse.py): the
# Hugging Face cache other AI apps share, and other Qwen3-TTS Studio folders.
if (-not $env:QWEN3TTS_USER_HF_HOME) { $env:QWEN3TTS_USER_HF_HOME = $env:HF_HOME }
if (-not $env:QWEN3TTS_USER_HF_HUB_CACHE) { $env:QWEN3TTS_USER_HF_HUB_CACHE = $env:HF_HUB_CACHE }
$others = @($env:QWEN3TTS_OTHER_INSTALLS -split ';')
$others += Find-PreviousInstall
if ($ModelsFrom) {
    if (-not (Test-Path -LiteralPath $ModelsFrom)) { Fail "-ModelsFrom: $ModelsFrom doesn't exist." }
    $others += $ModelsFrom
}
$env:QWEN3TTS_OTHER_INSTALLS = (@($others | Where-Object { $_ -and -not (Same-Dir $_ $Root) }) | Select-Object -Unique) -join ';'

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
function Test-Packages([string[]]$modules) {
    # Whether .venv has these packages (found without importing them, which is slow).
    $list = ($modules | ForEach-Object { "'$_'" }) -join ','
    $ErrorActionPreference = 'Continue'
    & $py -c "import importlib.util as u, sys; sys.exit(any(u.find_spec(m) is None for m in [$list]))" 2>$null | Out-Null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = 'Stop'
    return $ok
}

function Install-Requirements([string[]]$extra) {
    $present = (Test-Path $py) -and (Test-Packages @('qwen_tts', 'gradio', 'fastapi', 'uvicorn', 'yt_dlp', 'demucs', 'imageio_ffmpeg', 'lameenc', 'hf_xet'))
    if ($present) {
        Step 'Checking Qwen3-TTS and the app requirements (already installed; only updates are fetched)'
    } else {
        Step 'Installing Qwen3-TTS and the app requirements'
    }
    # yt-dlp has to keep up with YouTube, so running install.bat again always updates it.
    $reqArgs = @('pip', 'install', '--python', $py, '-r', (Join-Path $Root 'requirements.txt'),
        '--upgrade-package', 'yt-dlp', '--upgrade-package', 'yt-dlp-ejs') + $extra
    $ErrorActionPreference = 'Continue'
    & $uv @reqArgs
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -eq 0) { return }
    if ($present) {
        # Offline, say: what's installed already works, so carry on.
        Warn "Couldn't check for package updates (exit code $code). Keeping the packages already installed."
    } else {
        Fail "Installing requirements failed (exit code $code). Scroll up for details."
    }
}

function Torch-Installed {
    # The version and build of the PyTorch in .venv, like 2.8.0+cu128, or $null.
    if (-not (Test-Path $py)) { return $null }
    $ErrorActionPreference = 'Continue'
    $out = & $py -c "import torch, torchaudio; print('TORCH', torch.__version__, torchaudio.__version__)" 2>$null
    $ErrorActionPreference = 'Stop'
    $line = @($out) | Where-Object { "$_" -like 'TORCH *' } | Select-Object -First 1
    if (-not $line) { return $null }
    $parts = "$line".Split(' ')
    if ($parts[1] -ne $parts[2]) { return $null }  # torchaudio must match torch
    return $parts[1]
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

    $haveTorch = Torch-Installed
    if ($haveTorch -eq "$TorchVersion+$torchIndex") {
        Step "PyTorch $haveTorch is already installed"
    } else {
        Step "Installing PyTorch $TorchVersion ($torchIndex). This is a big download (about 3 GB)."
        if ($haveTorch) { Info "Replacing PyTorch $haveTorch, which isn't the build this PC needs." }
        Run $uv @('pip', 'install', '--python', $py,
            "torch==$TorchVersion", "torchaudio==$TorchVersion",
            '--index-url', "https://download.pytorch.org/whl/$torchIndex") 'Installing PyTorch'
    }

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
    Step 'Getting the Qwen3-TTS models (about 9 GB; models already on this PC are reused, and downloads resume)'
    $dlArgs = @('-m', 'app.download_models')
    if ($AllModels) { $dlArgs += '--all' }
    Run $py $dlArgs 'Downloading models'
}

# --- 6b. KoboldCpp and the GGUF model, for the GGUF option in Speak and the voice server ---
if ($SkipModels -or $SkipGguf) {
    Step 'Skipping KoboldCpp and the GGUF model. They download the first time you pick GGUF.'
} else {
    Step 'Getting KoboldCpp and the GGUF voice model (about 2.3 GB if they are not here yet)'
    $ErrorActionPreference = 'Continue'
    & $py -m app.gguf
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) {
        Warn 'That download failed, but everything else is installed. Run install.bat again to retry,'
        Warn 'or it downloads the first time you pick the GGUF model in Speak or the voice server.'
    }
}

# --- 6c. Music remover and speech-to-text, for cloning from YouTube and transcribing clips ---
if ($SkipModels) {
    Step 'Skipping the music remover and speech-to-text models. They download the first time they are needed.'
} else {
    Step 'Getting the music remover and speech-to-text models (about 1.8 GB if they are not here yet)'
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

# --- 8. Shortcuts -------------------------------------------------------------------
# A first install offers desktop and Start menu shortcuts to the menu (Qwen3-TTS.bat).
# Later runs only refresh the shortcuts that already point at this folder.
$launcher = Join-Path $Root 'Qwen3-TTS.bat'
$haveShortcut = $false
if (-not $NoShortcut -and (Test-Path -LiteralPath $launcher)) {
    try {
        $ws = New-Object -ComObject WScript.Shell
        $links = @(
            (Join-Path ([Environment]::GetFolderPath('Desktop')) $ShortcutName),
            (Join-Path ([Environment]::GetFolderPath('Programs')) $ShortcutName)
        )
        $make = @()
        foreach ($lnk in $links) {
            if ((Test-Path -LiteralPath $lnk) -and (Same-Dir (Split-Path -Parent $ws.CreateShortcut($lnk).TargetPath) $Root)) { $make += $lnk }
        }
        if (-not $installedHere) {
            $add = $true
            if (-not $Yes) {
                Write-Host ''
                $answer = Read-Host '    Add a Qwen3-TTS Studio shortcut to your desktop and Start menu? (Y/n)'
                $add = $answer -notmatch '^[nN]'
            }
            if ($add) { $make = $links }
        }
        $icon = Join-Path $Root 'app\icon.ico'
        foreach ($lnk in $make) {
            $s = $ws.CreateShortcut($lnk)
            $s.TargetPath = $launcher
            $s.WorkingDirectory = $Root
            $s.Description = 'Make voices with Qwen3-TTS and use them in SillyTavern and other apps'
            if (Test-Path -LiteralPath $icon) { $s.IconLocation = "$icon,0" }
            $s.Save()
            $haveShortcut = $true
        }
        if ($make -and -not $installedHere) { Info 'Added the Qwen3-TTS Studio shortcut to your desktop and Start menu.' }
    } catch {
        Warn "Couldn't create the shortcuts ($($_.Exception.Message)). Start Qwen3-TTS.bat from the install folder instead."
    }
}

Write-Host ''
Write-Host 'All done!' -ForegroundColor Green
Info "Qwen3-TTS Studio is installed in $Root"
if ($haveShortcut) {
    Info 'To start it, use the Qwen3-TTS Studio shortcut on your desktop or in the Start menu'
    Info '(or double-click Qwen3-TTS.bat in that folder).'
} else {
    Info 'To start it, double-click Qwen3-TTS.bat in that folder.'
}
Info 'Its menu starts the Voice Studio, the Voice Server and Speak.'
Write-Host ''
exit 0
