@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Qwen3-TTS Studio is not installed yet. Run install.bat first.
    pause
    exit /b 1
)
title Qwen3-TTS Voice Server
"%~dp0.venv\Scripts\python.exe" -m app.server %*
pause
