# scripts/sync_deps.ps1
# Automates the synchronization of .in files to .txt files and installs them.

$ErrorActionPreference = "Stop"

Write-Host "--- Starting Dependency Sync ---" -ForegroundColor Cyan

# 0. Check for missing imports
Write-Host "[0/4] Checking for missing requirements in src/..." -ForegroundColor Yellow
python scripts/check_missing_deps.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# DP-250: three .in files now. requirements.in = lean base (what CI installs via
# requirements-dev.txt); requirements-voice.in = base + heavy voice/STT stack
# (prod, via Dockerfile); requirements-dev.in = base + tooling (CI).
#
# CONSISTENCY: the three locks share the base deps, but compiling them
# independently lets a voice/tooling constraint (e.g. numba capping numpy) pin a
# shared package to a DIFFERENT version than the base lock -> `pip-sync` conflict
# below and CI(dev)-vs-prod(voice) drift (the divergence DP-249/250 fights).
# So compile the base FIRST, then feed requirements.txt as a --constraint to the
# voice and dev compiles: shared deps are forced to the base resolution, and any
# genuine incompatibility surfaces here as a compile error instead of silent skew.
# (This replaces the one-off `--constraint=<temp file>` the committed locks were
# built with, which sync did not reproduce.)

# 1. Compile lean base production requirements
Write-Host "[1/5] Compiling requirements.txt (lean base)..." -ForegroundColor Yellow
pip-compile requirements.in --resolver=backtracking --quiet

# 2. Compile full production requirements (base + voice/STT), pinned to the base lock
Write-Host "[2/5] Compiling requirements-voice.txt (prod, base + voice)..." -ForegroundColor Yellow
pip-compile requirements-voice.in --constraint requirements.txt --resolver=backtracking --quiet

# 3. Compile development/CI requirements (lean base + tooling), pinned to the base lock
Write-Host "[3/5] Compiling requirements-dev.txt (CI, lean)..." -ForegroundColor Yellow
pip-compile requirements-dev.in --constraint requirements.txt --resolver=backtracking --quiet

# 4. Synchronize local environment to the FULL set (tooling + voice) so local
#    dev mirrors prod capabilities; CI installs only requirements-dev.txt.
Write-Host "[4/5] Synchronizing virtual environment..." -ForegroundColor Yellow
pip-sync requirements-dev.txt requirements-voice.txt

# 5. Verify with Mypy
Write-Host "[5/5] Verifying static analysis..." -ForegroundColor Yellow
mypy src/ --config-file mypy.ini

Write-Host "--- Sync Complete! ---" -ForegroundColor Green
