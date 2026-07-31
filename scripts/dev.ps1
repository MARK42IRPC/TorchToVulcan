[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Dependencies are not installed. Run .\scripts\setup.ps1 first."
}

$Backend = Start-Process `
    -FilePath $PythonPath `
    -ArgumentList "-m", "uvicorn", "torch_to_vulcan.api:app", "--host", "127.0.0.1", "--port", $ApiPort, "--reload" `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -PassThru

try {
    Push-Location (Join-Path $ProjectRoot "web")
    & npm run dev -- --host 127.0.0.1 --port $WebPort
    if ($LASTEXITCODE -ne 0) { throw "The WebUI development server exited with an error." }
} finally {
    Pop-Location
    if (-not $Backend.HasExited) {
        Stop-Process -Id $Backend.Id
    }
}
