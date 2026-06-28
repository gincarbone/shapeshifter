#Requires -Version 5.1
<#
.SYNOPSIS
  Starts ContextZip and creates the local .venv on first run if needed.
#>

$ErrorActionPreference = 'Stop'
$root  = $PSScriptRoot
$venv  = Join-Path $root '.venv'
$py    = Join-Path $venv 'Scripts\python.exe'
$pip   = Join-Path $venv 'Scripts\pip.exe'

Set-Location $root

# Create .venv if missing
if (-not (Test-Path $py)) {
    Write-Host '[ContextZip] Creating local .venv...' -ForegroundColor Cyan
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error 'python not found. Install Python 3.11+ and try again.'
        exit 1
    }
    Write-Host '[ContextZip] Installing dependencies...' -ForegroundColor Cyan
    & $py -m pip install --upgrade pip --quiet 2>$null
    & $py -m pip install -r requirements.txt --quiet
    Write-Host '[ContextZip] Dependencies installed.' -ForegroundColor Green
}

# Reinstall dependencies if requirements.txt changed
$reqFile  = Join-Path $root 'requirements.txt'
$stampFile = Join-Path $venv '.installed_stamp'
if ((Test-Path $reqFile) -and (
    -not (Test-Path $stampFile) -or
    (Get-Item $reqFile).LastWriteTime -gt (Get-Item $stampFile).LastWriteTime
)) {
    Write-Host '[ContextZip] requirements.txt changed - reinstalling dependencies...' -ForegroundColor Yellow
    & $py -m pip install -r requirements.txt --quiet
    (Get-Date).ToString() | Out-File $stampFile -Encoding utf8
}

# Create .env if missing
$envFile = Join-Path $root '.env'
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root '.env.example') $envFile
    Write-Host '[ContextZip] Created .env from .env.example - configure UPSTREAM_API_KEY.' -ForegroundColor Yellow
}

# Start server
Write-Host ''
Write-Host '[ContextZip] Starting server...' -ForegroundColor Green
& $py wrapper_server.py
