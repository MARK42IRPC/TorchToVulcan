[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonPath = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher (py) was not found. Install Python 3.11 or later."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js 20 or later."
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "[setup] Creating Python virtual environment..."
    & py -3.11 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python virtual environment." }
}

Write-Host "[setup] Installing Python dependencies..."
& $PythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $PythonPath -m pip install -e "$ProjectRoot[dev,web]"
if ($LASTEXITCODE -ne 0) { throw "Failed to install Python dependencies." }

Write-Host "[setup] Installing WebUI dependencies..."
Push-Location (Join-Path $ProjectRoot "web")
try {
    if (Test-Path -LiteralPath "package-lock.json") {
        & npm ci
    } else {
        & npm install
    }
    if ($LASTEXITCODE -ne 0) { throw "Failed to install WebUI dependencies." }
} finally {
    Pop-Location
}

Write-Host "[setup] Dependencies are ready."
Write-Host "[setup] Start development with: .\scripts\dev.ps1"
