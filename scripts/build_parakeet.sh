#!/usr/bin/env bash
# Build parakeet.cpp and the puppet_parakeet Python binding.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINDING_BUILD="$ROOT/build/parakeet"

# Fetch / update vendored parakeet.cpp unless PARAKEET_ROOT points elsewhere
if [[ -n "${PARAKEET_ROOT:-}" ]]; then
  PARAKEET_SRC="$(cd "$PARAKEET_ROOT/.." && pwd)"
  echo "==> Using PARAKEET_ROOT override: $PARAKEET_SRC"
else
  "$ROOT/scripts/fetch_native_deps.sh"
  PARAKEET_SRC="$ROOT/vendor/native/parakeet.cpp"
fi

# CUDA build (default on when nvcc is available). Set PARAKEET_CUDA=0 for CPU-only.
PARAKEET_CUDA="${PARAKEET_CUDA:-auto}"
PARAKEET_CUDA_ARCH="${PARAKEET_CUDA_ARCH:-87}"   # Jetson Orin
CUDACXX="${CUDACXX:-/usr/local/cuda-12.6/bin/nvcc}"

if [[ "$PARAKEET_CUDA" == "auto" ]]; then
  if [[ -x "$CUDACXX" ]]; then
    PARAKEET_CUDA=1
  else
    PARAKEET_CUDA=0
  fi
fi

if [[ "$PARAKEET_CUDA" == "1" ]]; then
  PARAKEET_BUILD="${PARAKEET_BUILD:-$ROOT/build/native/parakeet-cuda}"
else
  PARAKEET_BUILD="${PARAKEET_BUILD:-$ROOT/build/native/parakeet-cpu}"
fi

if [[ ! -d "$PARAKEET_SRC" ]]; then
  echo "parakeet.cpp source not found: $PARAKEET_SRC" >&2
  exit 1
fi

echo "==> Building parakeet.cpp in $PARAKEET_BUILD"
mkdir -p "$PARAKEET_BUILD"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DPARAKEET_SHARED=ON
  -DPARAKEET_BUILD_CLI=ON
)

if [[ "$PARAKEET_CUDA" == "1" ]]; then
  echo "    CUDA: ON (arch=${PARAKEET_CUDA_ARCH}, compiler=${CUDACXX})"
  CMAKE_ARGS+=(
    -DPARAKEET_GGML_CUDA=ON
    -DCMAKE_CUDA_COMPILER="$CUDACXX"
    -DCMAKE_CUDA_ARCHITECTURES="$PARAKEET_CUDA_ARCH"
  )
else
  echo "    CUDA: OFF"
fi

cmake -S "$PARAKEET_SRC" -B "$PARAKEET_BUILD" "${CMAKE_ARGS[@]}"
cmake --build "$PARAKEET_BUILD" -j"$(nproc)"

if [[ ! -f "$PARAKEET_BUILD/libparakeet.so" ]]; then
  echo "libparakeet.so not found in $PARAKEET_BUILD" >&2
  exit 1
fi

echo "==> Building puppet_parakeet pybind11 module"
PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -c "import pybind11" 2>/dev/null; then
  echo "    Installing pybind11 into active Python environment"
  "$PYTHON" -m pip install pybind11
fi
PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
PYTHON_EXECUTABLE="$("$PYTHON" -c 'import sys; print(sys.executable)')"

mkdir -p "$BINDING_BUILD"
cmake -S "$ROOT/bindings/parakeet" -B "$BINDING_BUILD" \
  -DPARAKEET_ROOT="$PARAKEET_BUILD" \
  -DPARAKEET_SRC="$PARAKEET_SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYTHON_EXECUTABLE" \
  -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build "$BINDING_BUILD" -j"$(nproc)"

echo "==> Installing puppet_parakeet into active Python environment"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

cp -v "$PARAKEET_BUILD/libparakeet.so" "$SITE_PACKAGES/"
while IFS= read -r lib; do
  cp -v "$lib" "$SITE_PACKAGES/"
done < <(find "$PARAKEET_BUILD" -name 'libggml*.so' -type f)

MODULE="$(find "$BINDING_BUILD" -maxdepth 1 -name 'puppet_parakeet*.so' -print -quit)"
if [[ -z "$MODULE" ]]; then
  echo "puppet_parakeet module not found in $BINDING_BUILD" >&2
  exit 1
fi
cp -v "$MODULE" "$SITE_PACKAGES/"

python -c "import puppet_parakeet; print('puppet_parakeet OK')"

echo "Done."
