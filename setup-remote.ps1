#Requires -RunAsAdministrator
# One-shot remote-access setup: OpenSSH Server + Mac pubkey + Tailscale check.
# Run as Administrator. Safe to re-run (idempotent).

$ErrorActionPreference = 'Stop'
$MacPubKey = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJsXsNzU8fGnrZxoFWr0LRp9MUsiF1odnZmlyrimwJOF adam@MBP19.local'

Write-Host "==> Installing OpenSSH Server capability..." -ForegroundColor Cyan
$cap = Get-WindowsCapability -Online -Name OpenSSH.Server*
if ($cap.State -ne 'Installed') {
    Add-WindowsCapability -Online -Name $cap.Name | Out-Null
    Write-Host "    installed."
} else {
    Write-Host "    already installed."
}

Write-Host "==> Starting sshd and enabling autostart..." -ForegroundColor Cyan
Set-Service -Name sshd -StartupType Automatic
Start-Service sshd
Get-Service sshd | Format-Table Name, Status, StartType

Write-Host "==> Ensuring firewall rule for sshd (TCP 22)..." -ForegroundColor Cyan
if (-not (Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' `
        -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
    Write-Host "    rule created."
} else {
    Write-Host "    rule already exists."
}

Write-Host "==> Installing Mac pubkey for admin SSH..." -ForegroundColor Cyan
$adminKeys = 'C:\ProgramData\ssh\administrators_authorized_keys'
if (-not (Test-Path $adminKeys)) {
    New-Item -Path $adminKeys -ItemType File -Force | Out-Null
}
$existing = Get-Content $adminKeys -ErrorAction SilentlyContinue
if ($existing -notcontains $MacPubKey) {
    Add-Content -Path $adminKeys -Value $MacPubKey
    Write-Host "    key added."
} else {
    Write-Host "    key already present."
}

Write-Host "==> Fixing administrators_authorized_keys ACL (SYSTEM + Administrators only)..." -ForegroundColor Cyan
icacls $adminKeys /inheritance:r | Out-Null
icacls $adminKeys /grant 'SYSTEM:F' 'BUILTIN\Administrators:F' | Out-Null
Write-Host "    ACL set."

Write-Host "==> Setting default SSH shell to PowerShell..." -ForegroundColor Cyan
$pwshPath = (Get-Command powershell.exe).Source
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell `
    -Value $pwshPath -PropertyType String -Force | Out-Null
Write-Host "    DefaultShell = $pwshPath"

Write-Host "==> Tailscale status:" -ForegroundColor Cyan
$ts = Get-Command tailscale -ErrorAction SilentlyContinue
if ($ts) {
    & tailscale status
    Write-Host ""
    Write-Host "==> This host on the tailnet:" -ForegroundColor Cyan
    & tailscale ip -4
} else {
    Write-Host "    tailscale CLI not on PATH; check 'C:\Program Files\Tailscale\tailscale.exe'." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "DONE. From the Mac, try:  ssh Adam@<this-host-tailscale-name>" -ForegroundColor Green
