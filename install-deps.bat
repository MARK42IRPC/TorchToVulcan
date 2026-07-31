@echo off
setlocal
cd /d "%~dp0"

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo PowerShell was not found. Install PowerShell and try again.
    pause
    exit /b 1
)

echo Installing Torch to Vulcan dependencies...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1"
if errorlevel 1 (
    echo.
    echo Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo Dependencies are ready.
pause
