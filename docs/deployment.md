# Deployment (no Docker)

## 1. System dependencies

```bash
sudo apt install build-essential cmake python3-venv python3-dev \
  portaudio19-dev libespeak-ng-dev
```

Silero VAD uses **onnxruntime** only (same runtime as Piper TTS). No torch. Download all required models:

```bash
./scripts/download_models.sh
```

## 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 3. Native builds

### parakeet.cpp (STT)

Requires `pybind11` for the Python binding. `build_parakeet.sh` installs it automatically if missing, or you can pre-install:

```bash
pip install pybind11
# or: pip install -e ".[dev]"
```

```bash
# parakeet.cpp + pybind11 binding (clones to vendor/native/, CUDA on by default)
./scripts/build_parakeet.sh

# CPU-only parakeet build
PARAKEET_CUDA=0 ./scripts/build_parakeet.sh
```

`build_parakeet.sh` sets `-DPARAKEET_GGML_CUDA=ON`, `-DPARAKEET_SHARED=ON`, and `CMAKE_CUDA_ARCHITECTURES=87` by default. Override with:

```bash
PARAKEET_CUDA_ARCH=87 CUDACXX=/usr/local/cuda-12.6/bin/nvcc ./scripts/build_parakeet.sh
```

### llama-cpp-python (LLM, CUDA)

> **WARNING — one binding per venv**
>
> `build_llama.sh` and `build_llama_prism.sh` both install into the **same** `.venv` via `pip install --force-reinstall`. They do **not** coexist: running one script **replaces** the other's `llama_cpp` package entirely (~30–45 min to rebuild when switching back).
>
> Each script prints a prominent overwrite warning before starting if `llama_cpp` is already installed. From an interactive terminal you must confirm with `y`; non-interactive runs skip the prompt — set `LLAMA_BUILD_YES=1` to acknowledge explicitly.
>
> `llm.binding` in `config/llm.yaml` does **not** select between two installed libraries — it only records which binding you intend to use and must match `vendor/native/llama-binding.lock` (written by whichever script you ran last).

`llama-cpp-python` is an **optional** dependency (`pip install -e ".[llm]"` or `pip install -e ".[dev]"` does **not** install it, and a plain `pip install llama-cpp-python` often gives a **CPU-only** wheel). On Jetson you must **build from source** with CUDA enabled.

| Script | llama.cpp source | Use for |
|--------|------------------|---------|
| `./scripts/build_llama.sh` | upstream (ggml-org, via llama-cpp-python submodule) | Gemma, Bonsai Q1/Q4, standard GGUF |
| `./scripts/build_llama_prism.sh` | PrismML fork | Ternary-Bonsai Q2_0 |

**What gets overwritten**

| Location | Shared? | Effect of re-running either build script |
|----------|---------|------------------------------------------|
| `.venv/lib/.../llama_cpp/` | Yes | Previous `llama_cpp` wheel is **removed and replaced** |
| `vendor/llama-cpp-python/` | Yes | Git tree reset; Prism patches/symlink discarded before rebuild |
| `vendor/native/llama-binding.lock` | Yes | Updated to `upstream` or `prism` (last script wins) |

**Prerequisites (both LLM builds)**

- Pinned `llama-cpp-python` tag in `vendor/native/llama-cpp-python.lock` (currently **v0.3.33**, latest stable; Gemma 4 needs v0.3.30+). Override at build time with `LLAMA_CPP_PYTHON_REF_OVERRIDE=v0.3.34 ./scripts/build_llama.sh` when a newer tag ships.

- CUDA toolkit with `nvcc` on `PATH` (Jetson: usually `/usr/local/cuda-12.6/bin/nvcc`)
- Active project venv: `source .venv/bin/activate`
- Enough disk space and time (first build: ~30–45 min on Orin)

**Find your GPU compute capability**

```bash
# Orin Nano / Orin NX / Orin AGX → 87
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

#### Upstream llama.cpp (Gemma, standard Q4/Q8)

```bash
source .venv/bin/activate
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CUDA_ARCH=87 ./scripts/build_llama.sh
```

`config/llm.yaml`:

```yaml
backend: llama
binding: upstream
model_path: models/llm/gemma-4-E2B-it-Q4_K_M.gguf
```

#### Ternary-Bonsai (Q2_0, Prism fork)

Standard upstream llama.cpp does **not** work with Ternary-Bonsai `Q2_0` models. Build against the PrismML fork:

```bash
# Build Prism fork once (CUDA Orin example)
git clone https://github.com/PrismML-Eng/llama.cpp ../PrismML-Eng/Jun12-2026
cd ../PrismML-Eng/Jun12-2026
cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j

# Build llama-cpp-python against that fork (~30–45 min on Orin)
PRISM_LLAMA_ROOT=../PrismML-Eng/Jun12-2026 ./scripts/build_llama_prism.sh
```

`config/llm.yaml`:

```yaml
backend: llama
binding: prism
model_path: models/llm/Ternary-Bonsai-4B-Q2_0.gguf
```

**Switching bindings**

There is no fast switch — you must **re-run the full build** for the binding you want, then update `llm.binding`. Puppet checks the lock file at startup and errors if config and venv disagree.

```bash
# Example: currently on prism, want upstream again
./scripts/build_llama.sh          # overwrites prism install in .venv
# edit config/llm.yaml: binding: upstream
```

Verify:

```bash
./scripts/test_llm.py
```

| Variable | Purpose |
|----------|---------|
| `CUDACXX` | Path to `nvcc` (must match your JetPack CUDA) |
| `CUDA_ARCH` / `CMAKE_CUDA_ARCHITECTURES` | GPU arch (Orin = `87`) |
| `FORCE_CMAKE=1` | Forces a source build instead of a prebuilt CPU wheel (set by build scripts) |
| `PRISM_LLAMA_ROOT` | Path to Prism fork checkout (prism build only) |
| `LLAMA_BUILD_YES=1` | Skip interactive overwrite confirmation (scripts still print the warning) |
| `LLAMA_CPP_PYTHON_REF_OVERRIDE` | Use a different llama-cpp-python git tag than the lock file (e.g. when testing a new release) |

**Other platforms**

- x86_64 NVIDIA GPU: set `CUDA_ARCH` to your card (e.g. `75` for Turing, `86` for Ampere desktop, `89` for Ada).
- If your CUDA install is elsewhere, point `CUDACXX` at that `nvcc` (e.g. `/usr/local/cuda/bin/nvcc`).

**Verify CUDA is enabled**

```bash
python -c "import llama_cpp; print('OK:', llama_cpp.__file__)"
ls .venv/lib/python*/site-packages/llama_cpp/lib/libggml-cuda.so
```

The second command should list `libggml-cuda.so`. If only `libggml-cpu.so` appears, reinstall with the appropriate `./scripts/build_llama*.sh`.

**Smoke test with a GGUF model**

```bash
python -c "
from llama_cpp import Llama
llm = Llama(
    model_path='models/llm/model.gguf',
    n_ctx=8096,
    n_gpu_layers=-1,
    type_k=2,   # q4_0
    type_v=2,   # q4_0
    flash_attn=True,
    verbose=True,
)
print(llm('Hello', max_tokens=16))
"
```

With `verbose=True`, look for CUDA / `ggml_cuda` lines in the log (not CPU BLAS only). In `config/llm.yaml`, defaults match `llama.cpp -c 8096 -ctk q4_0 -ctv q4_0 -fa on` with `n_gpu_layers: -1` to offload all layers to the GPU.

**Troubleshooting**

| Symptom | Fix |
|---------|-----|
| `No module named 'llama_cpp'` | Run `./scripts/build_llama.sh` or `./scripts/build_llama_prism.sh` inside `.venv` |
| `unknown model architecture: 'gemma4'` | Rebuild with `./scripts/build_llama.sh` — pin must be v0.3.30+ (`vendor/native/llama-cpp-python.lock`) |
| `llm.binding is … but the active venv was built for …` | Re-run the matching build script or change `llm.binding` |
| Inference is slow, high CPU usage | Rebuild with `GGML_CUDA=on`; confirm `libggml-cuda.so` exists |
| CMake cannot find CUDA | Set `CUDACXX` to the full path to `nvcc` |
| Build fails on wrong arch | Set `CUDA_ARCH` from `nvidia-smi` compute cap (Orin = `87`) |
| Out of memory at runtime | Use a smaller / more quantized GGUF (`Q4_K_M`, etc.) in `config/llm.yaml` |

## 4. Models

```bash
./scripts/download_models.sh
```

The downloader is idempotent: it skips files already present.

Default targets:

- `models/vad/silero_vad.onnx`
- `models/stt/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf`
- `models/llm/Bonsai-8B-Q1_0.gguf`
- `models/tts/en_US-ryan-medium.onnx` (+ `.json`) — English
- `models/tts/fr_FR-siwis-medium.onnx` (+ `.json`) — French
- `models/tts/de_DE-thorsten-medium.onnx` (+ `.json`) — German

### Language

Edit `config/language.yaml` to set `language.active` (`en`, `fr`, or `de`). Each profile configures:

- STT locale (`stt_language` → parakeet `stream_begin_lang`)
- Piper TTS model paths
- LLM system prompt (reply language)

Override at runtime:

```bash
puppet --language fr
# or
PUPPET_LANGUAGE__ACTIVE=de puppet
```

You can override any source/path with environment variables, for example:

```bash
LLM_URL="https://huggingface.co/your-org/your-model/resolve/main/model.gguf" \
LLM_PATH="models/llm/custom.gguf" \
./scripts/download_models.sh
```

## 5. Microphone / STT audio

Parakeet expects **mono PCM at 16 kHz** as float32 in `[-1, 1]` internally.
PyAudio capture is fixed at **16 kHz, `paInt16` (16-bit signed PCM), mono** — no resampling.

On startup you should see:

```
Mic opened: 'USB Audio Device' (index=2) 16000 Hz int16 mono, chunk=320 samples (20 ms)
```

**Diagnose capture + STT:**

```bash
python scripts/check_mic.py --list-devices
python scripts/check_mic.py --device 2 --seconds 5
```

Pick a device with `16k_int16=yes`. While speaking, `peak` should be **> 0.05**.
If open fails, set `audio.input_device` in `config/default.yaml`.

## 6. Run

```bash
./scripts/run_puppet.sh
# or test capture for 5 seconds:
./scripts/run_puppet.sh --once 5
```

## 7. systemd (optional)

Copy `deploy/systemd/puppet.service` to `/etc/systemd/system/`, adjust paths, then:

```bash
sudo systemctl enable --now puppet
```
