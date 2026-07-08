#!/usr/bin/env bash
# Build llama-cpp-python against upstream ggml-org/llama.cpp (standard Q4/Q8 models).
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_llama_build.sh"
llama_build_init

llama_require_venv
llama_warn_overwrite upstream
llama_ensure_cpp_python
llama_use_upstream_submodule
llama_pip_install
llama_write_binding_lock upstream
llama_smoke_test

echo ""
echo "Done. Set config/llm.yaml binding: upstream and run ./scripts/test_llm.py"
