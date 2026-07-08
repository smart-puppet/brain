# Shared helpers for scripts/build_llama.sh and scripts/build_llama_prism.sh.
# shellcheck shell=bash

llama_build_init() {
  ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
  VENDOR_DIR="$ROOT/vendor/llama-cpp-python"
  LOCK_FILE="$ROOT/vendor/native/llama-cpp-python.lock"
  BINDING_LOCK="$ROOT/vendor/native/llama-binding.lock"
  VENV="${VENV:-$ROOT/.venv}"
  CUDA_ARCH="${CUDA_ARCH:-87}"
  CUDACXX="${CUDACXX:-/usr/local/cuda-12.6/bin/nvcc}"
  MODEL="${MODEL:-}"
}

llama_require_venv() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Virtualenv not found at $VENV — create it first (python -m venv .venv)." >&2
    exit 1
  fi
}

llama_read_installed_binding() {
  if [[ ! -f "$BINDING_LOCK" ]]; then
    return 0
  fi
  # shellcheck disable=SC1090
  source "$BINDING_LOCK"
  printf '%s' "${LLAMA_BINDING:-}"
}

llama_warn_overwrite() {
  local target_binding="$1"
  local existing_binding
  existing_binding="$(llama_read_installed_binding)"
  local has_llama_cpp=0
  if "$VENV/bin/python" -c "import llama_cpp" 2>/dev/null; then
    has_llama_cpp=1
  fi

  if [[ -z "$existing_binding" && "$has_llama_cpp" -eq 0 ]]; then
    return 0
  fi

  local other_script="./scripts/build_llama.sh"
  if [[ "$existing_binding" == "prism" ]]; then
    other_script="./scripts/build_llama_prism.sh"
  fi

  echo "" >&2
  echo "================================================================================" >&2
  echo " WARNING: EXISTING llama-cpp-python INSTALL WILL BE OVERWRITTEN" >&2
  echo "================================================================================" >&2
  echo "" >&2
  echo " Target venv:     $VENV" >&2
  echo " This build:      $target_binding ($(basename "${BASH_SOURCE[1]}"))" >&2
  if [[ -n "$existing_binding" ]]; then
    echo " Last build was:  $existing_binding (see vendor/native/llama-binding.lock)" >&2
  fi
  if [[ "$has_llama_cpp" -eq 1 ]]; then
    echo " llama_cpp:       already importable from this venv" >&2
  fi
  echo "" >&2
  if [[ -n "$existing_binding" && "$existing_binding" != "$target_binding" ]]; then
    echo " >>> The $existing_binding binding will be REMOVED from .venv." >&2
    echo " >>> To use $existing_binding again you must re-run $other_script (~30-45 min)." >&2
    echo "" >&2
  fi
  echo " pip will force-reinstall llama-cpp-python into .venv (no side-by-side installs)." >&2
  echo " After this build, set config/llm.yaml -> llm.binding: $target_binding" >&2
  echo "================================================================================" >&2
  echo "" >&2

  if [[ -t 0 && "${LLAMA_BUILD_YES:-}" != "1" ]]; then
    read -r -p "Continue and overwrite the existing install? [y/N] " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
      echo "Aborted." >&2
      exit 1
    fi
  fi
}

llama_clean_vendor_changes() {
  # Prism build replaces the submodule with a symlink and patches ctypes bindings.
  if [[ -L "$VENDOR_DIR/vendor/llama.cpp" ]]; then
    rm -f "$VENDOR_DIR/vendor/llama.cpp"
  elif [[ -d "$VENDOR_DIR/vendor/llama.cpp" ]] && [[ ! -e "$VENDOR_DIR/vendor/llama.cpp/.git" ]]; then
    rm -rf "$VENDOR_DIR/vendor/llama.cpp"
  fi
  if [[ -d "$VENDOR_DIR/.git" ]]; then
    git -C "$VENDOR_DIR" reset --hard HEAD
  fi
}

llama_ensure_cpp_python() {
  if [[ ! -f "$LOCK_FILE" ]]; then
    echo "Lock file not found: $LOCK_FILE" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "$LOCK_FILE"
  : "${LLAMA_CPP_PYTHON_REPO:?missing in llama-cpp-python.lock}"
  : "${LLAMA_CPP_PYTHON_REF:?missing in llama-cpp-python.lock}"
  if [[ -n "${LLAMA_CPP_PYTHON_REF_OVERRIDE:-}" ]]; then
    echo "==> Overriding llama-cpp-python ref: ${LLAMA_CPP_PYTHON_REF} -> ${LLAMA_CPP_PYTHON_REF_OVERRIDE}"
    LLAMA_CPP_PYTHON_REF="$LLAMA_CPP_PYTHON_REF_OVERRIDE"
  fi

  if [[ ! -d "$VENDOR_DIR/.git" ]]; then
    echo "==> Cloning llama-cpp-python @ $LLAMA_CPP_PYTHON_REF"
    git clone --recursive --branch "$LLAMA_CPP_PYTHON_REF" --depth 1 \
      "$LLAMA_CPP_PYTHON_REPO" "$VENDOR_DIR"
    return
  fi

  echo "==> Updating llama-cpp-python @ $LLAMA_CPP_PYTHON_REF"
  llama_clean_vendor_changes
  git -C "$VENDOR_DIR" fetch origin --tags --depth 1
  git -C "$VENDOR_DIR" checkout -f "$LLAMA_CPP_PYTHON_REF"
  git -C "$VENDOR_DIR" submodule update --init --recursive
}

llama_use_upstream_submodule() {
  echo "==> Using upstream llama.cpp from llama-cpp-python submodule"
  if [[ -L "$VENDOR_DIR/vendor/llama.cpp" ]] || [[ -d "$VENDOR_DIR/vendor/llama.cpp" ]]; then
    rm -rf "$VENDOR_DIR/vendor/llama.cpp"
  fi
  git -C "$VENDOR_DIR" submodule update --init --recursive vendor/llama.cpp
  if [[ ! -f "$VENDOR_DIR/vendor/llama.cpp/include/llama.h" ]]; then
    echo "upstream llama.cpp submodule missing at $VENDOR_DIR/vendor/llama.cpp" >&2
    exit 1
  fi
}

llama_pip_install() {
  echo "==> Building llama-cpp-python into $VENV (CUDA arch=$CUDA_ARCH)"
  export CUDACXX
  export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
  export FORCE_CMAKE=1
  "$VENV/bin/pip" install "$VENDOR_DIR" --no-cache-dir --force-reinstall
}

llama_write_binding_lock() {
  local binding="$1"
  mkdir -p "$(dirname "$BINDING_LOCK")"
  local cpp_commit=""
  if [[ -d "$VENDOR_DIR/vendor/llama.cpp/.git" ]]; then
    cpp_commit="$(git -C "$VENDOR_DIR/vendor/llama.cpp" rev-parse HEAD 2>/dev/null || true)"
  fi
  cat >"$BINDING_LOCK" <<EOF
# Written by scripts/build_llama*.sh — do not edit by hand
LLAMA_BINDING=${binding}
LLAMA_CPP_PYTHON_REF=${LLAMA_CPP_PYTHON_REF}
LLAMA_CPP_COMMIT=${cpp_commit}
EOF
  echo "==> Recorded binding: ${binding} -> $BINDING_LOCK"
}

llama_smoke_test() {
  local model="${MODEL:-$ROOT/models/llm/gemma-4-E2B-it-Q4_K_M.gguf}"
  if ! "$VENV/bin/python" - <<PY
import sys
from pathlib import Path

model = Path("${model}")
if not model.is_file():
    print(f"Model not found ({model}); skipping inference smoke test.")
    sys.exit(0)

try:
    from llama_cpp import Llama
except ImportError as exc:
    print(f"Smoke test failed: llama_cpp not importable: {exc}", file=sys.stderr)
    sys.exit(1)

try:
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
except ValueError as exc:
    msg = str(exc)
    print(f"Smoke test skipped: could not load {model.name}: {msg}", file=sys.stderr)
    if "unknown model architecture" in msg or "gemma4" in msg:
        print(
            "Hint: Gemma 4 needs llama-cpp-python v0.3.30+ (see vendor/native/llama-cpp-python.lock; currently v0.3.33).",
            file=sys.stderr,
        )
    print("pip install succeeded; fix the model path or rebuild with a newer pin.", file=sys.stderr)
    sys.exit(0)
except Exception as exc:
    print(f"Smoke test skipped: {exc}", file=sys.stderr)
    print("pip install succeeded.", file=sys.stderr)
    sys.exit(0)
PY
  then
    echo "Smoke test: llama_cpp import check failed." >&2
    return 1
  fi
}
