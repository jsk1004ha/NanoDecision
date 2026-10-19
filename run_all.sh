#!/usr/bin/env bash
# Linux/WSL launcher. Creates an isolated venv; never edits drivers or OS packages.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${ND2_VENV:-$ROOT/.venv031}"
SMOKE=0
for arg in "$@"; do if [[ "$arg" == "--smoke" ]]; then SMOKE=1; fi; done
trap 'printf "\nExecution stopped. Re-run the SAME command to resume. Check the last error above.\n" >&2' ERR
"$PYTHON_BIN" - <<'PY'
import sys
if not (3,11)<=sys.version_info[:2]<=(3,13):
    raise SystemExit('Use Python 3.11–3.13 for the pinned 0.3.1 training environment.')
PY
if ! command -v c++ >/dev/null 2>&1; then
  printf 'C++17 compiler missing. On Ubuntu install build-essential and python3-dev first.\n' >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then "$PYTHON_BIN" -m venv "$VENV"; fi
export ND2_BLAS="${ND2_BLAS:-1}"
BUILD_ID="$("$VENV/bin/python" - "$ROOT" "$SMOKE" "$ND2_BLAS" <<'PY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1]);h=hashlib.sha256()
for name in ['requirements-auto-lock.txt','pyproject.toml','setup.py']:
 h.update((root/name).read_bytes())
for folder,patterns in [('nanodecision2',['*.py']),('nanodecision2/recipes',['*.json']),('native',['*.hpp','*.cpp'])]:
 for pattern in patterns:
  for p in sorted((root/folder).glob(pattern)):
   h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
h.update(sys.version.encode());h.update(sys.argv[2].encode());h.update(sys.argv[3].encode());print(h.hexdigest())
PY
)"
STAMP="$VENV/.nd2-install-id"
if [[ ! -f "$STAMP" ]] || [[ "$(cat "$STAMP")" != "$BUILD_ID" ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip
  # CPU fixture and CUDA training use different installation fingerprints.
  if [[ "$SMOKE" == 1 ]]; then
    "$VENV/bin/python" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cpu
  else
    "$VENV/bin/python" -m pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128
  fi
  "$VENV/bin/python" -m pip install -r requirements-auto-lock.txt
  "$VENV/bin/python" -m pip install --no-deps --no-build-isolation .
  "$VENV/bin/python" - <<'PY'
import nanodecision2,torch
from nanodecision2 import _native
assert nanodecision2.__version__=='0.3.1'
print('Installed NanoDecision2',nanodecision2.__version__,'PyTorch',torch.__version__)
PY
  printf '%s\n' "$BUILD_ID" > "$STAMP"
fi
exec "$VENV/bin/python" -u -m nanodecision2.autopipeline "$@"
