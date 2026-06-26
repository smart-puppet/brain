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
./scripts/build_native.sh

# LLM: Ternary-Bonsai via Prism llama-server (see docs/deployment.md)
# Optional manual server: ./scripts/build_llama_prism.sh

# Run
puppet --config config/
# French or German:
puppet --language fr
puppet --language de
```

Set the default language in `config/language.yaml` (`language.active: en|fr|de`). Each profile sets STT locale, Piper voice, and LLM system prompt.

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
