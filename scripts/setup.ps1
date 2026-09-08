$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        $pythonCommand = $null
        $pythonArgs = @()
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.12 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $pythonCommand = 'py'; $pythonArgs = @('-3.12') }
        }
        if (-not $pythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
            & python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3,12) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { $pythonCommand = 'python' }
        }
        if (-not $pythonCommand) {
            throw 'Install Python 3.12 (64-bit) from python.org, include the Python Launcher, then run SETUP again. See README.md.'
        }
        & $pythonCommand @pythonArgs -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'Could not update pip. Check your internet connection.' }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Keep this window open and show the error to your coding assistant.' }
    if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
    & '.\.venv\Scripts\python.exe' scripts\doctor.py
    if ($LASTEXITCODE -ne 0) { throw 'Setup checks failed. Read the messages above.' }
    Write-Host 'Setup finished. Optional: EDIT_KEYS.cmd. Then double-click START.cmd.' -ForegroundColor Green
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
