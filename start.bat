@echo off
setlocal
cd /d "%~dp0"

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo PowerShell was not found. Install PowerShell and try again.
    pause
    exit /b 1
)

echo Starting Torch to Vulcan...
echo WebUI: http://127.0.0.1:5173
echo Press Ctrl+C in this window to stop the development servers.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\dev.ps1"
if errorlevel 1 (
    echo.
    echo Development servers stopped with an error.
    pause
    exit /b 1
)

pause
