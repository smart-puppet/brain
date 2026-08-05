#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  PYTHON="$(command -v python)"
else
  echo "No .venv found in $ROOT — create one and run: pip install -e ." >&2
  exit 1
fi

if ! "$PYTHON" -c "import puppet" 2>/dev/null; then
  echo "Package 'puppet' not importable. From $ROOT run: .venv/bin/pip install -e ." >&2
  exit 1
fi

# Parakeet and llama.cpp each ship their own libggml — never put parakeet's
# ggml dirs on LD_LIBRARY_PATH or llama will abort (ABI mismatch).
if ! "$PYTHON" -c "import puppet_parakeet" 2>/dev/null; then
  echo "puppet_parakeet failed to import. Run: ./scripts/build_parakeet.sh" >&2
  exit 1
fi

exec "$PYTHON" -m puppet.main --config "${PUPPET_CONFIG:-config}" "$@"
