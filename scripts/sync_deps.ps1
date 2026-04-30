# scripts/sync_deps.ps1
# Automates the synchronization of .in files to .txt files and installs them.

$ErrorActionPreference = "Stop"

Write-Host "--- Starting Dependency Sync ---" -ForegroundColor Cyan

# 0. Check for missing imports
Write-Host "[0/4] Checking for missing requirements in src/..." -ForegroundColor Yellow
python scripts/check_missing_deps.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 1. Compile production requirements
Write-Host "[1/4] Compiling requirements.txt..." -ForegroundColor Yellow
pip-compile requirements.in --resolver=backtracking --quiet

# 2. Compile development requirements
Write-Host "[2/4] Compiling requirements-dev.txt..." -ForegroundColor Yellow
pip-compile requirements-dev.in --resolver=backtracking --quiet

# 3. Synchronize local environment
Write-Host "[3/4] Synchronizing virtual environment..." -ForegroundColor Yellow
pip-sync requirements-dev.txt

# 4. Verify with Mypy
Write-Host "[4/4] Verifying static analysis..." -ForegroundColor Yellow
mypy src/ --config-file mypy.ini

Write-Host "--- Sync Complete! ---" -ForegroundColor Green
