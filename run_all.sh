#!/usr/bin/env bash
# NanoDecision2 0.3.1. Does not install/replace your OS or NVIDIA driver.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${ND2_PYTHON:-python3}"
VENV="${ND2_VENV:-$ROOT/.venv031}"
SMOKE=0
for arg in "$@"; do [[ "$arg" == "--smoke" ]] && SMOKE=1; done
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 2; }
command -v "$PYTHON" >/dev/null || fail 'Python 3.11-3.13 is required. Ubuntu 24.04: sudo apt install python3 python3-venv python3-dev'
"$PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,13), "Bootstrap requires Python 3.11-3.13"' || exit 2
command -v "${CXX:-c++}" >/dev/null || fail 'A C++17 compiler is required. Ubuntu: sudo apt install build-essential python3-dev'
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV" || fail 'Cannot create venv. Ubuntu: sudo apt install python3-venv'
fi
VPY="$VENV/bin/python"
# Save build identity to avoid rebuilding/changing a native binary on resume.
SIGNATURE="$("$VPY" - "$ROOT" "$SMOKE" <<'PY'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]);h=hashlib.sha256(sys.argv[2].encode())
for name in ('pyproject.toml','setup.py','requirements-auto-lock.txt','run_all.sh'):
    h.update((root/name).read_bytes())
for pattern in ('nanodecision2/*.py','nanodecision2/recipes/*.json','native/*.hpp','native/*.cpp'):
    for p in sorted(root.glob(pattern)):h.update(p.relative_to(root).as_posix().encode());h.update(p.read_bytes())
print(h.hexdigest())
PY
)"
MARKER="$VENV/.nd2_bootstrap_complete"
if [[ ! -f "$MARKER" ]] || [[ "$(cat "$MARKER")" != "$SIGNATURE" ]]; then
  "$VPY" -m pip install 'pip>=24,<26' 'setuptools>=70,<82' wheel
  INDEX='https://download.pytorch.org/whl/cu128'
  [[ "$SMOKE" == 1 ]] && INDEX='https://download.pytorch.org/whl/cpu'
  "$VPY" -m pip install --upgrade --force-reinstall 'torch==2.10.0' --index-url "$INDEX"
  "$VPY" -m pip install -r requirements-auto-lock.txt
  # BLAS is optional. Only enable after both its header and linker were tested.
  export ND2_BLAS=0
  PROBE="$(mktemp -d)"
  printf '#include <cblas.h>\nint main(){float a=1,b=1,c=0;cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,1,1,1,1,&a,1,&b,1,0,&c,1);return c!=1;}\n' > "$PROBE/probe.cpp"
  if "${CXX:-c++}" "$PROBE/probe.cpp" -lblas -o "$PROBE/probe" >/dev/null 2>&1 && "$PROBE/probe"; then export ND2_BLAS=1; fi
  rm -rf -- "$PROBE"
  echo "Building native engine (system BLAS=$ND2_BLAS)."
  "$VPY" -m pip install --no-build-isolation --no-deps --force-reinstall .
  "$VPY" -m pip check
  "$VPY" -m pip freeze > "$VENV/installed_versions.txt"
  printf '%s\n' "$SIGNATURE" > "$MARKER"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec "$VENV/bin/nd2" auto "$@"
