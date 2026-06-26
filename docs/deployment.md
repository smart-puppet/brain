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

Requires `pybind11` for the Python binding. `build_native.sh` installs it automatically if missing, or you can pre-install:

```bash
pip install pybind11
# or: pip install -e ".[dev]"
```

```bash
# parakeet.cpp + pybind11 binding (clones to vendor/native/, CUDA on by default)
./scripts/build_native.sh

# CPU-only parakeet build
PARAKEET_CUDA=0 ./scripts/build_native.sh
```

`build_native.sh` sets `-DPARAKEET_GGML_CUDA=ON`, `-DPARAKEET_SHARED=ON`, and `CMAKE_CUDA_ARCHITECTURES=87` by default. Override with:

```bash
PARAKEET_CUDA_ARCH=87 CUDACXX=/usr/local/cuda-12.6/bin/nvcc ./scripts/build_native.sh
```

### llama-cpp-python (LLM, CUDA)

`llama-cpp-python` is an **optional** dependency (`pip install -e ".[llm]"` or `pip install -e ".[dev]"` does **not** install it, and a plain `pip install llama-cpp-python` often gives a **CPU-only** wheel). On Jetson you must **build from source** with CUDA enabled.

#### Ternary-Bonsai (Q2_0, recommended on Orin)

Standard `pip install llama-cpp-python` does **not** work with Ternary-Bonsai `Q2_0` models. Build **in-process** bindings against the PrismML fork (faster than `llama-server` HTTP):

```bash
# Build Prism fork once (CUDA Orin example)
git clone https://github.com/PrismML-Eng/llama.cpp ../PrismML-Eng/Jun12-2026
cd ../PrismML-Eng/Jun12-2026
cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j

# Build llama-cpp-python against that fork (~30–45 min on Orin)
PRISM_LLAMA_ROOT=../PrismML-Eng/Jun12-2026 ./scripts/build_llama_prism.sh
```

`config/llm.yaml` defaults to in-process bindings:

```yaml
backend: llama
model_path: models/llm/Ternary-Bonsai-4B-Q2_0.gguf
n_ctx: 8192
```

Verify:

```bash
./scripts/test_llm.py
```

**Alternative:** `backend: llama_server` with the Prism `llama-server` binary (slower due to HTTP overhead). See commented options in `config/llm.yaml`.

#### Standard llama.cpp (Bonsai-8B-Q1_0)

**Prerequisites**

- CUDA toolkit with `nvcc` on `PATH` (Jetson: usually `/usr/local/cuda-12.6/bin/nvcc`)
- Active project venv: `source .venv/bin/activate`
- Enough disk space and time (first build: ~30–45 min on Orin)

**Find your GPU compute capability**

```bash
# Orin Nano / Orin NX / Orin AGX → 87
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**Install with CUDA (Jetson Orin, CUDA 12.6, arch 87)**

```bash
source .venv/bin/activate

CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" \
FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```

Or install the project LLM extra with the same CUDA flags:

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" \
FORCE_CMAKE=1 pip install -e ".[llm]" --no-cache-dir
```

| Variable | Purpose |
|----------|---------|
| `CUDACXX` | Path to `nvcc` (must match your JetPack CUDA) |
| `CMAKE_ARGS` | `-DGGML_CUDA=on` enables GPU; `-DCMAKE_CUDA_ARCHITECTURES` must match your SoC |
| `FORCE_CMAKE=1` | Forces a source build instead of a prebuilt CPU wheel |
| `--no-cache-dir` | Avoids reusing a cached CPU-only wheel |

**Other platforms**

- x86_64 NVIDIA GPU: set `CMAKE_CUDA_ARCHITECTURES` to your card (e.g. `75` for Turing, `86` for Ampere desktop, `89` for Ada).
- If your CUDA install is elsewhere, point `CUDACXX` at that `nvcc` (e.g. `/usr/local/cuda/bin/nvcc`).

**Verify CUDA is enabled**

```bash
python -c "import llama_cpp; print('OK:', llama_cpp.__file__)"
ls .venv/lib/python*/site-packages/llama_cpp/lib/libggml-cuda.so
```

The second command should list `libggml-cuda.so`. If only `libggml-cpu.so` appears, reinstall with `FORCE_CMAKE=1` and the `CMAKE_ARGS` above.

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
| `No module named 'llama_cpp'` | Run the install command above inside `.venv` |
| Inference is slow, high CPU usage | Rebuild with `GGML_CUDA=on`; confirm `libggml-cuda.so` exists |
| CMake cannot find CUDA | Set `CUDACXX` to the full path to `nvcc` |
| Build fails on wrong arch | Set `CMAKE_CUDA_ARCHITECTURES` from `nvidia-smi` compute cap (Orin = `87`) |
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
