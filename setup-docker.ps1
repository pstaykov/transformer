# One-shot setup for the training + inference containers on Windows.
#
#   Right-click -> "Run with PowerShell", or from a PowerShell prompt:
#     .\setup-docker.ps1
#
# Requires Docker Desktop with the WSL2 backend and, for training, GPU
# support enabled (Docker Desktop > Settings > Resources > WSL Integration,
# plus an up-to-date NVIDIA driver on Windows itself - no separate
# container-toolkit install needed, Docker Desktop handles that).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Fail($msg) {
    Write-Host $msg -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "== Checking Docker Desktop =="
try {
    docker info | Out-Null
} catch {
    Fail "Docker doesn't seem to be running. Start Docker Desktop, wait for it to say 'Engine running', then re-run this script."
}
Write-Host "Docker is running."

Write-Host ""
Write-Host "== Checking GPU support (needed for training) =="
$gpuOk = $false
try {
    docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi | Out-Null
    $gpuOk = $true
    Write-Host "GPU is visible to Docker containers."
} catch {
    Write-Host "GPU is NOT visible to Docker yet." -ForegroundColor Yellow
    Write-Host "  - Make sure you have a recent NVIDIA driver installed on Windows."
    Write-Host "  - In Docker Desktop: Settings > Resources > WSL Integration, enable your distro."
    Write-Host "  - Docker Desktop's GPU support is automatic with WSL2 - no extra toolkit install needed."
    Write-Host "You can still build images now and fix GPU access before you run training."
}

Write-Host ""
Write-Host "== Building images (this can take a while the first time) =="
docker compose build

Write-Host ""
Write-Host "Done. To start:" -ForegroundColor Green
Write-Host "  Training : double-click start-training.bat  (resumes run3 with the same settings as before)"
Write-Host "  Inference: double-click start-inference.bat (showcase site + chat at http://localhost:8000)"
Read-Host "Press Enter to close"
