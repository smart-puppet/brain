#!/usr/bin/env bash
# Build llama-cpp-python against the PrismML llama.cpp fork (Ternary-Bonsai Q2_0).
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_llama_build.sh"
llama_build_init

PRISM_LLAMA_ROOT="${PRISM_LLAMA_ROOT:-$ROOT/../PrismML-Eng/Jun12-2026}"
PRISM_SRC="$(cd "$PRISM_LLAMA_ROOT" && pwd)"
PRISM_LOCK="$ROOT/vendor/native/prism-llama.lock"
MODEL="${MODEL:-$ROOT/models/llm/Ternary-Bonsai-4B-Q2_0.gguf}"

if [[ ! -f "$PRISM_SRC/include/llama.h" ]]; then
  echo "Prism fork not found at $PRISM_SRC" >&2
  echo "Set PRISM_LLAMA_ROOT to your PrismML-Eng/Jun12-2026 checkout." >&2
  exit 1
fi

llama_require_venv
llama_warn_overwrite prism
llama_ensure_cpp_python

echo "==> Prism fork: $PRISM_SRC"
if [[ -f "$PRISM_LOCK" ]]; then
  # shellcheck disable=SC1090
  source "$PRISM_LOCK"
  echo "    pinned commit: ${PRISM_LLAMA_COMMIT:-unknown}"
fi

echo "==> Linking vendor/llama.cpp -> Prism fork"
rm -rf "$VENDOR_DIR/vendor/llama.cpp"
ln -sf "$PRISM_SRC" "$VENDOR_DIR/vendor/llama.cpp"

echo "==> Patching llama-cpp-python bindings for Prism API"
"$VENV/bin/python" "$ROOT/scripts/patch_llama_prism_bindings.py"

llama_pip_install
llama_write_binding_lock prism
llama_smoke_test

echo ""
echo "Done. Set config/llm.yaml binding: prism and run ./scripts/test_llm.py"
