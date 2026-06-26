#!/usr/bin/env bash
# Build llama-cpp-python against the PrismML llama.cpp fork (Ternary-Bonsai Q2_0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT/vendor/llama-cpp-python"
LOCK_FILE="$ROOT/vendor/native/prism-llama.lock"
VENV="${VENV:-$ROOT/.venv}"

CUDA_ARCH="${CUDA_ARCH:-87}"   # Jetson Orin
CUDACXX="${CUDACXX:-/usr/local/cuda-12.6/bin/nvcc}"
PRISM_LLAMA_ROOT="${PRISM_LLAMA_ROOT:-$ROOT/../PrismML-Eng/Jun12-2026}"
PRISM_SRC="$(cd "$PRISM_LLAMA_ROOT" && pwd)"
MODEL="${MODEL:-$ROOT/models/llm/Ternary-Bonsai-4B-Q2_0.gguf}"

if [[ ! -f "$PRISM_SRC/include/llama.h" ]]; then
  echo "Prism fork not found at $PRISM_SRC" >&2
  echo "Set PRISM_LLAMA_ROOT to your PrismML-Eng/Jun12-2026 checkout." >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Virtualenv not found at $VENV — create it first (python -m venv .venv)." >&2
  exit 1
fi

echo "==> Prism fork: $PRISM_SRC"
if [[ -f "$LOCK_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$LOCK_FILE"
  echo "    pinned commit: ${PRISM_LLAMA_COMMIT:-unknown}"
fi

echo "==> Linking vendor/llama.cpp -> Prism fork"
rm -rf "$VENDOR_DIR/vendor/llama.cpp"
ln -sf "$PRISM_SRC" "$VENDOR_DIR/vendor/llama.cpp"

echo "==> Patching llama-cpp-python bindings for Prism API"
"$VENV/bin/python" "$ROOT/scripts/patch_llama_prism_bindings.py"

echo "==> Building llama-cpp-python (CUDA arch=$CUDA_ARCH)"
export CUDACXX
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
export FORCE_CMAKE=1
"$VENV/bin/pip" install "$VENDOR_DIR" --no-cache-dir --force-reinstall

echo "==> Verifying model load"
"$VENV/bin/python" - <<PY
import sys
from pathlib import Path

model = Path("${MODEL}")
if not model.is_file():
    print(f"Model not found ({model}); skipping inference smoke test.")
    sys.exit(0)

from llama_cpp import Llama

llm = Llama(
    model_path=str(model),
    n_ctx=512,
    n_gpu_layers=-1,
    n_batch=128,
    verbose=False,
)
out = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Say hi in one word."}],
    stream=False,
    max_tokens=4,
    temperature=0.0,
)
text = out["choices"][0]["message"]["content"]
print("smoke ok:", repr(text))
PY

echo ""
echo "Done. Set config/llm.yaml to backend: llama and run ./scripts/test_llm.py"
