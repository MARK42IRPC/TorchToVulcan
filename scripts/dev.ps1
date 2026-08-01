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

function Test-ListeningPort {
    param([int]$Port)
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Test-Health {
    param([int]$Port)
    try {
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        return $response.status -eq "ok" -and $response.api_version -eq "0.4"
    } catch {
        return $false
    }
}

function Find-HealthyPort {
    param(
        [int]$StartPort,
        [int]$SearchCount = 10
    )
    for ($candidate = $StartPort; $candidate -lt ($StartPort + $SearchCount); $candidate++) {
        if (Test-Health $candidate) {
            return $candidate
        }
    }
    return $null
}

function Find-FreePort {
    param([int]$StartPort)
    $candidate = $StartPort
    while (Test-ListeningPort $candidate) {
        $candidate++
    }
    return $candidate
}

$Backend = $null
$BackendStarted = $false
$HealthyApiPort = Find-HealthyPort $ApiPort

if ($null -ne $HealthyApiPort) {
    $ApiPort = $HealthyApiPort
    Write-Host "Reusing API: http://127.0.0.1:$ApiPort"
} else {
    if (Test-ListeningPort $ApiPort) {
        $ApiPort = Find-FreePort ($ApiPort + 1)
        Write-Host "API port was occupied. Using $ApiPort instead."
    }

    $Backend = Start-Process `
        -FilePath $PythonPath `
        -ArgumentList "-m", "uvicorn", "torch_to_vulcan.api:app", "--host", "127.0.0.1", "--port", $ApiPort, "--reload" `
        -WorkingDirectory $ProjectRoot `
        -NoNewWindow `
        -PassThru
    $BackendStarted = $true

    $apiReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (Test-Health $ApiPort) {
            $apiReady = $true
            break
        }
        Start-Sleep -Milliseconds 300
    }
    if (-not $apiReady) {
        if ($null -ne $Backend -and -not $Backend.HasExited) {
            Stop-Process -Id $Backend.Id
        }
        throw "API did not become ready on port $ApiPort."
    }
}

try {
    $HealthyWebPort = Find-HealthyPort $WebPort
    if ($null -ne $HealthyWebPort) {
        $WebPort = $HealthyWebPort
        Write-Host "Reusing WebUI: http://127.0.0.1:$WebPort"
        if ($BackendStarted) {
            Write-Host "The API was started and will remain available on port $ApiPort."
            $BackendStarted = $false
        }
        Write-Host "Development services are already running."
        return
    }

    if (Test-ListeningPort $WebPort) {
        $WebPort = Find-FreePort ($WebPort + 1)
        Write-Host "WebUI port was occupied. Using $WebPort instead."
    }

    $env:TTV_API_PORT = "$ApiPort"
    Write-Host "WebUI: http://127.0.0.1:$WebPort"
    Write-Host "API: http://127.0.0.1:$ApiPort"
    Write-Host "Press Ctrl+C in this window to stop newly started services."
    Push-Location (Join-Path $ProjectRoot "web")
    & npm run dev -- --host 127.0.0.1 --port $WebPort
    if ($LASTEXITCODE -ne 0) { throw "The WebUI development server exited with an error." }
} finally {
    Pop-Location
    if ($BackendStarted -and $null -ne $Backend -and -not $Backend.HasExited) {
        Stop-Process -Id $Backend.Id
    }
}
