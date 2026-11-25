@echo off
rem Qwen3-TTS Studio: installs everything the first time, then opens the menu that
rem asks which program to start: the Voice Studio, the Voice Server or Speak.
rem
rem   Qwen3-TTS.bat studio | server | speak   starts one of them right here, without
rem                                           the menu (server --engine gguf works too)
setlocal EnableExtensions
cd /d "%~dp0"
title Qwen3-TTS Studio
set PYTHONUTF8=1

if not exist "%~dp0scripts\install.ps1" goto :not_extracted

:begin
rem This copy was installed into another folder, so open that one.
if exist "%~dp0.installed-to.txt" goto :moved
if not exist "%~dp0.venv\Scripts\python.exe" goto :install
if not "%~1"=="" goto :program

"%~dp0.venv\Scripts\python.exe" -m app.menu
set "CODE=%ERRORLEVEL%"
if "%CODE%"=="10" goto :install
if "%CODE%"=="0" exit /b 0
echo.
echo   The menu stopped with an error, shown above. Running install.bat again
echo   usually fixes it.
echo.
pause
exit /b %CODE%

:program
rem One program in this window. The menu starts each one like this, in a window of its own.
"%~dp0.venv\Scripts\python.exe" -m app.menu %*
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%

:install
cls
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" goto :install_failed
echo   Press any key to open the menu.
pause >nul
goto :begin

:install_failed
echo   The install didn't finish. Scroll up to see why, then open Qwen3-TTS.bat
echo   again: it picks up where it left off.
echo.
pause
exit /b %CODE%

:moved
set "TARGET="
set /p TARGET=<"%~dp0.installed-to.txt"
if not defined TARGET goto :stale
if not exist "%TARGET%\Qwen3-TTS.bat" goto :stale
call "%TARGET%\Qwen3-TTS.bat" %*
exit /b %ERRORLEVEL%

:stale
del "%~dp0.installed-to.txt" >nul 2>&1
goto :begin

:not_extracted
echo.
echo   The rest of Qwen3-TTS Studio isn't next to this file. That usually means it
echo   was opened straight from the ZIP file without extracting it first.
echo.
echo   Close this window, right-click the ZIP file, choose "Extract All...", and
echo   then double-click Qwen3-TTS.bat in the folder that comes out.
echo.
pause
exit /b 1
