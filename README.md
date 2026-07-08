# Puppet

Voice chatbot for embedded Linux (Jetson). Single-process orchestrator with modular STT, LLM, and TTS backends.

| Component | Backend | Binding |
|-----------|---------|---------|
| STT | [parakeet.cpp](https://github.com/mudler/parakeet.cpp) | pybind11 (`bindings/parakeet`) |
| LLM | [llama.cpp](https://github.com/ggerganov/llama.cpp) | [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) |
| TTS | [Piper](https://github.com/OHF-Voice/piper1-gpl) | `piper-tts` Python API |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) | onnxruntime (no torch) |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

./scripts/download_models.sh

# Build parakeet.cpp bindings (clones source to vendor/native/ on first run)
./scripts/build_parakeet.sh

# LLM: build llama-cpp-python — WARNING: scripts overwrite each other in .venv (see docs/deployment.md)
# ./scripts/build_llama.sh          # upstream ggml-org/llama.cpp (Gemma, Q4, etc.)
# ./scripts/build_llama_prism.sh    # Prism fork for Ternary-Bonsai Q2_0

# Run
puppet --config config/
# French or German:
puppet --language fr
puppet --language de
```

Configuration: see **[config/README.md](config/README.md)**. Pick mic profile in `config/default.yaml` (`profile: respeaker`), language in `config/language.yaml` or `puppet --language fr`.

## Layout

```
config/          YAML settings (merged at startup)
src/puppet/      Application code
bindings/        Native extensions (parakeet.cpp)
vendor/native/   Pinned native sources (parakeet.cpp cloned here)
tests/           Unit and functional tests
docs/            Architecture and pipeline notes
scripts/         Build and run helpers
models/          Model files (GGUF, ONNX)
```

See [docs/architecture.md](docs/architecture.md) for design details.  
Native builds (parakeet.cpp, llama-cpp-python + CUDA): [docs/deployment.md](docs/deployment.md).
