@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0scripts\install.ps1" goto :not_extracted
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
echo.
pause
exit /b

:not_extracted
echo.
echo   The rest of Qwen3-TTS Studio isn't next to this file. Extract the ZIP first
echo   (right-click it, "Extract All..."), then run Qwen3-TTS.bat from the new folder.
echo.
pause
exit /b 1
