# Windows convenience wrapper. Training runs inside the user's default WSL distro.
# Usage: powershell -File .\run_all.ps1 --workdir /home/USER/nd2-runs/auto5m
$ErrorActionPreference = "Stop"
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL Ubuntu is required. This launcher does not change your OS or NVIDIA driver."
}
$LinuxScript = & wsl.exe --exec wslpath -u (Join-Path $PSScriptRoot 'run_all.sh')
if ($LASTEXITCODE -ne 0) { throw "Could not resolve this directory inside WSL." }
& wsl.exe --exec bash $LinuxScript @args
exit $LASTEXITCODE
