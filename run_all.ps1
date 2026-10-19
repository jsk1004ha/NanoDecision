# Windows entry point. Training executes inside the user's existing WSL Ubuntu.
# WSL and the NVIDIA driver must already be installed by the user.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$linuxPath = (& wsl.exe wslpath -a $here).Trim()
if ($LASTEXITCODE -ne 0) { throw "WSL is unavailable. Install/configure WSL Ubuntu first." }
& wsl.exe --cd $linuxPath bash ./run_all.sh @args
exit $LASTEXITCODE
